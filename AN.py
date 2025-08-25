
# run_patchcore_al_fixed.py
import os, copy, random
from glob import glob
from typing import List, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# Optional deps
try:
    import timm
    HAVE_TIMM = True
except Exception:
    HAVE_TIMM = False

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


# --------------------------- utils ---------------------------
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def build_loader_from_indices(dataset, indices, batch=32, shuffle=True, num_workers=2, pin_memory=True):
    from torch.utils.data import DataLoader, Subset
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=batch, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

def _l2n(t):
    eps=1e-6
    n = torch.norm(t, p=2, dim=1, keepdim=True).clamp_min(eps)
    return t / n


# --------------------------- data ---------------------------
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class AnomalyDataset(Dataset):
    """
    Loads images from two folders:
      - normal_dir (label 0)
      - anomaly_dir (label 1)
    """
    def __init__(self, normal_dir: str, anomaly_dir: str, size: int = 256, color: str = "L"):
        self.size = size
        self.color = color.lower()

        def _collect(folder, label):
            if not folder or not os.path.isdir(folder):
                return []
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
            files = []
            for e in exts:
                files += glob(os.path.join(folder, "**", e), recursive=True)
            files = sorted(list(set(files)))
            return [(p, label) for p in files]

        samples = []
        samples += _collect(normal_dir, 0)
        samples += _collect(anomaly_dir, 1)
        if len(samples) == 0:
            raise RuntimeError("No images found in the provided directories.")
        self.samples = samples
        self.labels = [lab for (_, lab) in samples]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        img = img.convert("RGB") if self.color.lower()=="rgb" else img.convert("L")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img).astype("float32") / 255.0
        if arr.ndim == 2: arr = arr[:, :, None]
        if arr.shape[2] == 4: arr = arr[:, :, :3]
        arr = np.transpose(arr, (2, 0, 1)).copy()  # C,H,W
        tensor = torch.from_numpy(arr).float()
        return tensor, int(label)

def split_normal_30(dataset: AnomalyDataset, seed: int = 42) -> Tuple[List[int], List[int]]:
    normals = [i for i, lab in enumerate(dataset.labels) if lab == 0]
    rnd = random.Random(seed); rnd.shuffle(normals)
    n_seed = max(1, int(round(0.30 * len(normals))))
    idx_seed = normals[:n_seed]
    idx_rest_normals = normals[n_seed:]
    return idx_seed, idx_rest_normals


# --------------------------- PatchCore backbone/adapter ---------------------------
class PCBackbone(nn.Module):
    """
    Frozen multi-scale feature extractor.
    If timm is available, uses features_only model (e.g., resnet50) and returns maps from out_indices.
    Otherwise, a tiny CNN fallback.
    """
    def __init__(self, model_name="resnet50", out_indices=(2,3), device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.use_timm = HAVE_TIMM
        if self.use_timm:
            self.model = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=out_indices)
            self.model.eval().to(self.device)
            for p in self.model.parameters(): p.requires_grad_(False)
        else:
            # tiny fallback (grayscale)
            self.c1 = nn.Conv2d(1, 32, 3, 2, 1)
            self.c2 = nn.Conv2d(32,64, 3, 2, 1)
            self.c3 = nn.Conv2d(64,128,3, 2, 1)
            self.c4 = nn.Conv2d(128,256,3, 2, 1)
            self.out_indices = out_indices
            self.to(self.device)

    @torch.no_grad()
    def forward(self, x01_1chw: torch.Tensor):
        if self.use_timm:
            x = x01_1chw.repeat(1,3,1,1) if x01_1chw.size(1)==1 else x01_1chw
            feats = self.model(x)  # list of maps
            return feats
        else:
            x = x01_1chw
            f1 = F.relu(self.c1(x)); f2 = F.relu(self.c2(f1))
            f3 = F.relu(self.c3(f2)); f4 = F.relu(self.c4(f3))
            feats = [f2, f3, f4]
            return [feats[i] for i in self.out_indices]

class PCAdapter(nn.Module):
    """Tiny learnable head per selected scale."""
    def __init__(self, in_chs: list, out_dim=256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)
        ) for c in in_chs])

    def forward(self, feats):
        return [p(f) for p, f in zip(self.proj, feats)]


# --------------------------- SWAG ---------------------------
class SWAGAdapter:
    def __init__(self, adapter: nn.Module, noise_scale=0.01, max_snaps=30, device="cuda"):
        self.proto = adapter
        self.noise = noise_scale
        self.max_snaps = max_snaps
        self.snaps = []
        self.device = device

    def collect(self, adapter: nn.Module):
        sd = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        if len(self.snaps) >= self.max_snaps:
            self.snaps.pop(0)
        self.snaps.append(sd)

    def _avg(self):
        assert self.snaps, "No SWAG snapshots collected."
        keys = self.snaps[0].keys()
        avg = {}
        for k in keys:
            tensors = [s[k] for s in self.snaps]
            t0 = tensors[0]
            if t0.is_floating_point():
                stk = torch.stack([t.to(torch.float32) for t in tensors], 0)
                avg[k] = stk.mean(0)
            else:
                avg[k] = t0.clone()
        return avg

    @torch.no_grad()
    def sample_adapters(self, K=8):
        avg = self._avg()
        outs = []
        for _ in range(K):
            Ad = copy.deepcopy(self.proto).to(self.device)
            noisy = {}
            for k, v in avg.items():
                noisy[k] = v + torch.randn_like(v) * self.noise if v.is_floating_point() else v
            Ad.load_state_dict(noisy, strict=False)
            Ad.eval()
            outs.append(Ad)
        return outs


# --------------------------- Memory / Coreset ---------------------------
def coreset_greedy(X: torch.Tensor, ratio=0.2, seed=42):
    N = X.size(0); M = max(1, int(ratio*N))
    rng = np.random.RandomState(seed)
    start = int(rng.randint(N))
    chosen = [start]
    d2 = torch.cdist(X[start:start+1], X, p=2).squeeze(0)  # [N]
    for _ in range(1, M):
        idx = int(torch.argmax(d2).item())
        chosen.append(idx)
        d2 = torch.minimum(d2, torch.cdist(X[idx:idx+1], X, p=2).squeeze(0))
    return X[chosen], np.array(chosen, dtype=np.int64)

class MemoryBank:
    def __init__(self, coreset_ratio=0.2, method="auto", large_N=300_000, device="cpu"):
        self.mem = None
        self.ratio = coreset_ratio
        self.method = method
        self.large_N = large_N
        self.device = torch.device(device)

    def build(self, patch_vecs: torch.Tensor):
        N = patch_vecs.size(0)
        X = _l2n(patch_vecs.T).T.contiguous()
        if self.method == "random" or (self.method == "auto" and N > self.large_N):
            M = max(1, int(self.ratio*N))
            idx = torch.randperm(N)[:M]
            self.mem = X[idx].to(self.device, non_blocking=True)
            print(f"[PatchCore] Memory (random) {self.mem.size(0)}/{N}")
        else:
            Xd = X.to(self.device, non_blocking=True)
            self.mem, _ = coreset_greedy(Xd, ratio=self.ratio)
            self.mem = self.mem.contiguous()
            print(f"[PatchCore] Memory (greedy) {self.mem.size(0)}/{N}")

    @torch.no_grad()
    def knn_dist(self, Q: torch.Tensor, k=3, q_chunk=4096, m_chunk=100_000, prefer_gpu=True):
        assert self.mem is not None, "MemoryBank not built."
        mem = self.mem
        Nm = mem.size(0)
        mem_f = mem if mem.dtype == torch.float32 else mem.float()
        outs = []
        for qi in range(0, Q.size(0), q_chunk):
            q = Q[qi:qi+q_chunk]
            q_f = q if q.dtype == torch.float32 else q.float()
            best = torch.full((q_f.size(0), k), float("inf"))
            q_dev = q_f.to(mem_f.device if (prefer_gpu and mem_f.is_cuda) else torch.device("cpu"), non_blocking=True)
            m_dev_target = q_dev.device
            for mi in range(0, Nm, m_chunk):
                m_blk = mem_f[mi:mi+m_chunk]
                try:
                    m_dev = m_blk.to(m_dev_target, non_blocking=True)
                    d_blk = torch.cdist(q_dev, m_dev, p=2)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        m_dev = m_blk.cpu()
                        q_dev = q_f.cpu()
                        best = best.cpu()
                        d_blk = torch.cdist(q_dev, m_dev, p=2)
                        m_dev_target = torch.device("cpu")
                    else:
                        raise
                cat = torch.cat([best.to(d_blk.device), d_blk], dim=1)
                best = torch.topk(cat, k=k, largest=False, dim=1).values
            outs.append(best.mean(dim=1).cpu())
        return torch.cat(outs, dim=0)


# --------------------------- Feature extraction & scoring ---------------------------
def extract_patch_vectors(backbone, adapter, loader, device, target_hw=(16,16), per_image_cap=256):
    backbone.eval(); adapter.eval()
    vecs=[]
    with torch.no_grad():
        for x,_ in loader:
            x = x.to(device, non_blocking=True)
            feats = adapter(backbone(x))
            ups  = [F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P    = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            if per_image_cap is not None:
                B, H, W = x.size(0), target_hw[0], target_hw[1]
                patches_per_img = H*W
                if patches_per_img > per_image_cap:
                    P = P.view(B, patches_per_img, -1)[:, :per_image_cap, :].reshape(-1, P.size(1))
            vecs.append(P.cpu())
    return torch.cat(vecs, 0)

def score_images_patchcore(backbone, adapter, mem, loader, device, top_q=0.05, k=3,
                           out_size=(256,256), collect_images=True):
    backbone.eval(); adapter.eval()
    all_scores, all_heatmaps = [], []
    all_imgs = [] if collect_images else None
    with torch.no_grad():
        for x,_ in loader:
            x = x.to(device)
            feats = adapter(backbone(x))
            Hmax = max(f.size(2) for f in feats); Wmax = max(f.size(3) for f in feats)
            ups = [F.interpolate(f, size=(Hmax,Wmax), mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            d = mem.knn_dist(P, k=k).view(x.size(0), Hmax, Wmax)
            flat = d.view(x.size(0), -1)
            q = max(1, int(top_q*flat.size(1)))
            s = torch.topk(flat, k=q, dim=1).values.mean(1)
            d_up = F.interpolate(d.unsqueeze(1), size=out_size, mode="bilinear", align_corners=False).squeeze(1)
            all_scores.append(s.cpu()); all_heatmaps.append(d_up.cpu())
            if collect_images:
                all_imgs.append(F.interpolate(x, size=out_size, mode="bilinear", align_corners=False).cpu())
    scores = torch.cat(all_scores,0)
    heatmaps = torch.cat(all_heatmaps,0)
    imgs_opt = torch.cat(all_imgs,0) if collect_images else None
    return (scores, heatmaps, imgs_opt) if collect_images else (scores, heatmaps)

@torch.no_grad()
def swag_uncertainty(backbone, swag_adp: SWAGAdapter, mem: MemoryBank, loader, device, K=4, top_q=0.05, k=3):
    all_vars=[]
    for x,_ in loader:
        x = x.to(device)
        s_ks=[]
        Adapters = swag_adp.sample_adapters(K=K)
        for Ad in Adapters:
            feats = Ad(backbone(x))
            Hmax = max(f.size(2) for f in feats); Wmax = max(f.size(3) for f in feats)
            ups = [F.interpolate(f, size=(Hmax,Wmax), mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            d = mem.knn_dist(P, k=k).view(x.size(0), Hmax, Wmax)
            flat = d.view(x.size(0), -1)
            q = max(1, int(top_q*flat.size(1)))
            s = torch.topk(flat, k=q, dim=1).values.mean(1)
            s_ks.append(s.cpu())
        S = torch.stack(s_ks,0)
        all_vars.append(S.var(0))
    return torch.cat(all_vars,0)


# --------------------------- Tiny adapter training ---------------------------
def train_adapter_compact(backbone, adapter, loader, device, epochs=3, lr=1e-4, proto_k=128):
    backbone.eval(); adapter.train()
    with torch.no_grad():
        vecs=[]
        for x,_ in loader:
            x=x.to(device)
            feats = adapter(backbone(x))
            Hmax = max(f.size(2) for f in feats); Wmax = max(f.size(3) for f in feats)
            ups = [F.interpolate(f, (Hmax,Wmax), mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            vecs.append(P.cpu())
        V = torch.cat(vecs,0)
    protos, _ = coreset_greedy(V, ratio=min(1.0, proto_k/max(1,V.size(0))))
    protos = protos.to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    for ep in range(epochs):
        tot=0; n=0
        for x,_ in loader:
            x=x.to(device)
            feats = adapter(backbone(x))
            Hmax = max(f.size(2) for f in feats); Wmax = max(f.size(3) for f in feats)
            ups = [F.interpolate(f, (Hmax,Wmax), mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            D = torch.cdist(P, protos, p=2)
            loss = D.min(1).values.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item(); n+=1
        print(f"[Adapter] ep {ep+1}/{epochs} loss={tot/max(1,n):.4f}")


# --------------------------- Active learning (FIXED) ---------------------------
def _scores_to_numpy(ret):
    if isinstance(ret, (tuple, list)):
        ret = ret[0]
    if hasattr(ret, "detach"):
        return ret.detach().cpu().numpy()
    return np.asarray(ret)

def active_learning_patchcore_swag_fixed(
    backbone, adapter, dataset, seed_indices_normals, pool_indices, device,
    rounds=3, budget=50, mem_coreset=0.2, swag_noise=0.005, swag_K=4,
    top_q=0.05, k=3, p_gate=0.95, save_dir="runs/patchcore_al",
    use_val_guard=False, val_loader=None, val_labels=None, guard_tol=0.01
):
    os.makedirs(save_dir, exist_ok=True)

    # 1) seed adapter train + SWAG
    seed_loader = build_loader_from_indices(dataset, seed_indices_normals, batch=32, shuffle=True)
    train_adapter_compact(backbone, adapter, seed_loader, device, epochs=3, lr=1e-4, proto_k=128)
    swag = SWAGAdapter(adapter, noise_scale=swag_noise, max_snaps=30, device=device)
    swag.collect(copy.deepcopy(adapter).eval()); swag.collect(copy.deepcopy(adapter).eval())

    # 2) frozen scoring memory from seeds (prevents contamination)
    seed_loader_eval = build_loader_from_indices(dataset, seed_indices_normals, batch=32, shuffle=False)
    V_seed = extract_patch_vectors(backbone, adapter, seed_loader_eval, device, target_hw=(16,16), per_image_cap=256)
    mem_seed = MemoryBank(coreset_ratio=mem_coreset, method="auto", device=device)
    mem_seed.build(V_seed)

    # 3) calibrate z-score gates from seeds
    s_seed_np = _scores_to_numpy(score_images_patchcore(backbone, adapter, mem_seed, seed_loader_eval, device, top_q=top_q, k=k))
    u_seed_np = swag_uncertainty(backbone, swag, mem_seed, seed_loader_eval, device, K=swag_K, top_q=top_q, k=k).cpu().numpy()
    mu_s, sd_s = float(np.mean(s_seed_np)), float(np.std(s_seed_np) + 1e-8)
    mu_u, sd_u = float(np.mean(u_seed_np)), float(np.std(u_seed_np) + 1e-8)
    print(f"[Calib] μ_s={mu_s:.4f} σ_s={sd_s:.4f} | μ_u={mu_u:.4f} σ_u={sd_u:.4f}")

    used = set()
    val_auc_prev = -np.inf

    for r in range(1, rounds+1):
        print(f"\n=== Round {r} ===")
        rem_local = [i for i in range(len(pool_indices)) if i not in used]
        if not rem_local:
            print("Pool exhausted."); break
        rem_global = [pool_indices[i] for i in rem_local]
        rem_loader = build_loader_from_indices(dataset, rem_global, batch=32, shuffle=False)

        # Score/uncert with frozen memory
        s_pool_np = _scores_to_numpy(score_images_patchcore(backbone, adapter, mem_seed, rem_loader, device, top_q=top_q, k=k))
        u_pool_np = swag_uncertainty(backbone, swag, mem_seed, rem_loader, device, K=swag_K, top_q=top_q, k=k).cpu().numpy()

        z_s = (s_pool_np - mu_s)/sd_s
        z_u = (u_pool_np - mu_u)/sd_u

        cand = np.nonzero((z_s <= 1.0) & (z_u <= 1.0))[0].tolist()
        if not cand:
            print("[Round] No safe candidates at 1.0σ; relaxing to 1.5σ once.")
            cand = np.nonzero((z_s <= 1.5) & (z_u <= 1.5))[0].tolist()
        take = cand[: min(budget, len(cand))]
        print(f"[Round {r}] Safe candidates: {len(cand)} | Taking: {len(take)} / budget {budget}")

        if not take:
            print("[Round] Nothing to add this round.")
            continue

        for li in take: used.add(rem_local[li])
        new_global = [rem_global[i] for i in take]

        # brief adapter tune on accepted samples
        round_loader = build_loader_from_indices(dataset, new_global, batch=32, shuffle=True)
        train_adapter_compact(backbone, adapter, round_loader, device, epochs=1, lr=5e-5, proto_k=128)
        swag.collect(copy.deepcopy(adapter).eval())

        # Optional: validation guardrail
        if use_val_guard and val_loader is not None and val_labels is not None and HAVE_SKLEARN:
            val_scores = _scores_to_numpy(score_images_patchcore(backbone, adapter, mem_seed, val_loader, device, top_q=top_q, k=k))
            val_auc = roc_auc_score(val_labels, val_scores)
            print(f"[Round {r}] Val AUC: {val_auc:.4f} (prev {val_auc_prev:.4f})")
            if val_auc < val_auc_prev - guard_tol:
                print("[AL] Early stop: validation AUC decreased.")
                break
            val_auc_prev = val_auc

    # Return adapter and the **seed** memory used for scoring
    return adapter, mem_seed


# --------------------------- Eval/plots ---------------------------
# def evaluate_and_plot(labels_all, scores, prefix="", outdir="runs/patchcore_al", target_precision=0.80):
#     os.makedirs(outdir, exist_ok=True)
#     if not HAVE_SKLEARN:
#         print("(Metrics skipped: scikit-learn not available.)")
#         return {}

#     auc = roc_auc_score(labels_all, scores)
#     fpr, tpr, thr = roc_curve(labels_all, scores)
#     ap  = average_precision_score(labels_all, scores)

#     J = tpr - fpr
#     jbest = int(np.nanargmax(J))
#     thr_best = thr[jbest] if np.isfinite(thr[jbest]) else np.quantile(scores, 0.95)
#     preds_y = (scores >= thr_best).astype(np.int32)

#     acc_y = accuracy_score(labels_all, preds_y)
#     prec_y = precision_score(labels_all, preds_y, zero_division=0)
#     rec_y  = recall_score(labels_all, preds_y, zero_division=0)
#     f1_y   = f1_score(labels_all, preds_y, zero_division=0)
#     cm_y   = confusion_matrix(labels_all, preds_y)

#     pr, rc, thr_pr = precision_recall_curve(labels_all, scores)
#     mask = pr[1:] >= target_precision
#     thr_op = thr_pr[mask][-1] if np.any(mask) else thr_pr[np.argmax(pr[1:])]
#     preds_o = (scores >= thr_op).astype(np.int32)

#     acc_o = accuracy_score(labels_all, preds_o)
#     prec_o = precision_score(labels_all, preds_o, zero_division=0)
#     rec_o  = recall_score(labels_all, preds_o, zero_division=0)
#     f1_o   = f1_score(labels_all, preds_o, zero_division=0)
#     cm_o   = confusion_matrix(labels_all, preds_o)

#     print(f"\n=== {prefix} Evaluation ===")
#     print(f"ROC AUC : {auc:.4f}")
#     print(f"PR  AUC : {ap:.4f}")
#     print(f"Thr* (Youden) : {thr_best:.6f}")
#     print(f"  ACC={acc_y:.4f}  Precision={prec_y:.4f}  Recall={rec_y:.4f}  F1={f1_y:.4f}")
#     print("  Confusion (Youden) [tn fp; fn tp]:\n", cm_y)

#     print(f"\nThr@P>={target_precision:.2f} : {thr_op:.6f}")
#     print(f"  ACC={acc_o:.4f}  Precision={prec_o:.4f}  Recall={rec_o:.4f}  F1={f1_o:.4f}")
#     print("  Confusion (OpPt) [tn fp; fn tp]:\n", cm_o)

#     if HAVE_PLT:
#         plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
#         plt.plot([0,1],[0,1],"k--"); plt.legend(); plt.title(f"ROC ({prefix})"); plt.xlabel("FPR"); plt.ylabel("TPR")
#         plt.savefig(os.path.join(outdir, f"roc_{prefix}.png"), dpi=120); plt.close()

#         plt.figure(); plt.plot(rc, pr, label=f"AP={ap:.4f}")
#         plt.title(f"Precision-Recall ({prefix})"); plt.xlabel("Recall"); plt.ylabel("Precision")
#         plt.savefig(os.path.join(outdir, f"pr_{prefix}.png"), dpi=120); plt.close()

#     return {
#         "auc": auc, "ap": ap, "thr_youden": thr_best, "thr_opt": thr_op,
#         "acc_y": acc_y, "prec_y": prec_y, "rec_y": rec_y, "f1_y": f1_y,
#         "acc_o": acc_o, "prec_o": prec_o, "rec_o": rec_o, "f1_o": f1_o
#     }

def evaluate_and_plot(labels_all, scores, prefix="", outdir="runs/patchcore_al", target_precision=0.80):
    os.makedirs(outdir, exist_ok=True)
    if not HAVE_SKLEARN:
        print("(Metrics skipped: scikit-learn not available.)")
        return {}

    # --- core metrics ---
    auc = roc_auc_score(labels_all, scores)
    fpr, tpr, thr = roc_curve(labels_all, scores)
    ap  = average_precision_score(labels_all, scores)

    # Youden
    J = tpr - fpr
    jbest = int(np.nanargmax(J))
    thr_best = thr[jbest] if np.isfinite(thr[jbest]) else np.quantile(scores, 0.95)
    preds_y = (scores >= thr_best).astype(np.int32)
    acc_y = accuracy_score(labels_all, preds_y)
    prec_y = precision_score(labels_all, preds_y, zero_division=0)
    rec_y  = recall_score(labels_all, preds_y, zero_division=0)
    f1_y   = f1_score(labels_all, preds_y, zero_division=0)
    cm_y   = confusion_matrix(labels_all, preds_y)

    # Operating point by target precision
    pr, rc, thr_pr = precision_recall_curve(labels_all, scores)
    mask = pr[1:] >= target_precision
    thr_op = thr_pr[mask][-1] if np.any(mask) else thr_pr[np.argmax(pr[1:])]
    preds_o = (scores >= thr_op).astype(np.int32)
    acc_o = accuracy_score(labels_all, preds_o)
    prec_o = precision_score(labels_all, preds_o, zero_division=0)
    rec_o  = recall_score(labels_all, preds_o, zero_division=0)
    f1_o   = f1_score(labels_all, preds_o, zero_division=0)
    cm_o   = confusion_matrix(labels_all, preds_o)

    # --- prints ---
    print(f"\n=== {prefix} Evaluation ===")
    print(f"ROC AUC : {auc:.4f}")
    print(f"PR  AUC : {ap:.4f}")
    print(f"Thr* (Youden) : {thr_best:.6f}")
    print(f"  ACC={acc_y:.4f}  Precision={prec_y:.4f}  Recall={rec_y:.4f}  F1={f1_y:.4f}")
    print("  Confusion (Youden) [tn fp; fn tp]:\n", cm_y)

    print(f"\nThr@P>={target_precision:.2f} : {thr_op:.6f}")
    print(f"  ACC={acc_o:.4f}  Precision={prec_o:.4f}  Recall={rec_o:.4f}  F1={f1_o:.4f}")
    print("  Confusion (OpPt) [tn fp; fn tp]:\n", cm_o)

    # --- plots (ROC, PR, Confusion Matrices) ---
    if HAVE_PLT:
        # ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0,1],[0,1],"k--")
        plt.legend(); plt.title(f"ROC ({prefix})"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.savefig(os.path.join(outdir, f"roc_{prefix}.png"), dpi=120); plt.close()

        # PR
        plt.figure()
        plt.plot(rc, pr, label=f"AP={ap:.4f}")
        plt.title(f"Precision-Recall ({prefix})"); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.savefig(os.path.join(outdir, f"pr_{prefix}.png"), dpi=120); plt.close()

        # Confusion matrix helper
        def _plot_cm(cm, title, fname):
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Normal (0)', 'Anomaly (1)'])
            plt.yticks(tick_marks, ['Normal (0)', 'Anomaly (1)'])
            thresh = cm.max() / 2.0 if cm.size else 0.5
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, fname), dpi=140)
            plt.close()

        # Plot both confusion matrices
        _plot_cm(cm_y, f"Confusion (Youden) — {prefix}", f"cm_youden_{prefix}.png")
        _plot_cm(cm_o, f"Confusion (OpPt P≥{target_precision:.2f}) — {prefix}", f"cm_opt_{prefix}.png")

    return {
        "auc": auc, "ap": ap, "thr_youden": thr_best, "thr_opt": thr_op,
        "acc_y": acc_y, "prec_y": prec_y, "rec_y": rec_y, "f1_y": f1_y, "cm_y": cm_y,
        "acc_o": acc_o, "prec_o": prec_o, "rec_o": rec_o, "f1_o": f1_o, "cm_o": cm_o,
    }

# --------------------------- MAIN ---------------------------
if __name__ == "__main__":
    set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ====== EDIT THESE PATHS (absolute paths recommended) ======
    normal_dir_train  = "Data_XRAY/train/NORMAL"#"chest_xray/train/NORMAL"#"/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/train/NORMAL"
    anomaly_dir_train = "Data_XRAY/train/COVID19"#"chest_xray/train/PNEUMONIA"#"/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/train/PNEUMONIA"
    normal_dir_test   = "Data_XRAY/test/NORMAL" #"/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/test/NORMAL"
    anomaly_dir_test  ="Data_XRAY/test/COVID19"#"/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/test/PNEUMONIA"



#.     normal_dir_train  = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/Data_XRAY/train/NORMAL"#"/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/train/NORMAL"
#     anomaly_dir_train = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/Data_XRAY/train/PNEUMONIA"#"/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/train/PNEUMONIA"
#     normal_dir_test   = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/Data_XRAY/test/NORMAL"##"/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/test/NORMAL"
#     anomaly_dir_test  = "Data_XRAY/test/PNEUMONIA"#"/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/test/PNEUMONIA"


    # ===========================================================

    size = 256
    batch = 32
    DATASET_LABEL_ONE_IS_ANOMALY = True
    TARGET_PRECISION = 0.70

    # Datasets
    dataset = AnomalyDataset(normal_dir_train, anomaly_dir_train, size=size, color="L")
    idx_seed, idx_rest_normals = split_normal_30(dataset, seed=42)
    idx_anoms = [i for i in range(len(dataset)) if dataset.labels[i] == 1]
    pool_indices = idx_rest_normals + idx_anoms

    test_dataset = AnomalyDataset(normal_dir_test, anomaly_dir_test, size=size, color="L")
    test_loader  = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    # Backbone + adapter
    backbone = PCBackbone(model_name="resnet50", out_indices=(2,3), device=device)
    with torch.no_grad():
        x0, _ = dataset[0]
        x0 = x0.unsqueeze(0).to(device)
        feats0 = backbone(x0)
        in_chs = [f.size(1) for f in feats0]
    adapter = PCAdapter(in_chs=in_chs, out_dim=256).to(device)

    # -------- Baseline memory from whole train set (for baseline evaluation) --------
    print("\n=== Baseline evaluation (before AL) ===")
    train_loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    feats_all = []
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            feats = adapter(backbone(x))
            Hmax = max(f.size(2) for f in feats); Wmax = max(f.size(3) for f in feats)
            ups = [F.interpolate(f, size=(Hmax,Wmax), mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            feats_all.append(P.cpu())
    feats_all = torch.cat(feats_all,0).contiguous().float()

    memory_baseline = MemoryBank(coreset_ratio=0.3, method="auto", device=device)
    memory_baseline.build(feats_all)

    s_base, heat_base, imgs_base = score_images_patchcore(
        backbone, adapter, memory_baseline, test_loader, device,
        top_q=0.05, k=3, out_size=(256,256), collect_images=True
    )

    labels_base = []
    for _, y in test_loader: labels_base.extend(y.tolist())
    labels_base = np.asarray(labels_base, dtype=np.int32)
    if not DATASET_LABEL_ONE_IS_ANOMALY: labels_base = 1 - labels_base

    eval_base = evaluate_and_plot(
        labels_base, s_base.numpy(), prefix="baseline",
        outdir="runs/patchcore_al/baseline", target_precision=TARGET_PRECISION
    )

    os.makedirs("runs/patchcore_al/baseline_vis", exist_ok=True)
    if HAVE_PLT:
        try:
            # simple preview
            idx_vis = 0
            img_v = imgs_base[idx_vis].permute(1,2,0).numpy()
            hm_v  = heat_base[idx_vis].numpy()
            h = (hm_v - hm_v.min())/(hm_v.max()-hm_v.min()+1e-8)
            cmap = plt.get_cmap('jet')
            overlay = (1-0.5)*np.repeat(img_v[..., :1], 3, axis=2) + 0.5*cmap(h)[..., :3]
            plt.imsave("runs/patchcore_al/baseline_vis/example.png", overlay)
            print("Saved:", "runs/patchcore_al/baseline_vis/example.png")
        except Exception as e:
            print("Baseline vis failed:", e)

    # -------- Active learning (fixed) --------
    adapter_trained, memory_seed = active_learning_patchcore_swag_fixed(
        backbone, adapter, dataset,
        seed_indices_normals=idx_seed,
        pool_indices=pool_indices,
        device=device,
        rounds=5,          # you can lower to 3 if needed
        budget=50,         # smaller budget per round
        mem_coreset=0.3,
        swag_noise=0.005,
        swag_K=4,
        top_q=0.05,
        k=3,
        p_gate=0.95,
        save_dir="runs/patchcore_al",
        use_val_guard=False,   # set True if you provide val_loader & val_labels
        val_loader=None,
        val_labels=None,
        guard_tol=0.01
    )

    # -------- Post-AL evaluation (with FROZEN seed memory) --------
    print("\n=== Post-AL evaluation (after AL) ===")
    s_post, heat_post, imgs_post = score_images_patchcore(
        backbone, adapter_trained, memory_seed, test_loader, device,
        top_q=0.05, k=3, out_size=(256,256), collect_images=True
    )

    labels_post = []
    for _, y in test_loader: labels_post.extend(y.tolist())
    labels_post = np.asarray(labels_post, dtype=np.int32)
    if not DATASET_LABEL_ONE_IS_ANOMALY: labels_post = 1 - labels_post

    eval_post = evaluate_and_plot(
        labels_post, s_post.numpy(), prefix="postAL",
        outdir="runs/patchcore_al/postAL", target_precision=TARGET_PRECISION
    )

    # -------- Summary --------
    def fmt(v): 
        try: return f"{v:.4f}"
        except: return "nan"
    print("\n=== Summary ===")
    print("Baseline:  AUC:", fmt(eval_base.get("auc")), " AP:", fmt(eval_base.get("ap")), " F1@Youden:", fmt(eval_base.get("f1_y")))
    print("Post-AL :  AUC:", fmt(eval_post.get("auc")), " AP:", fmt(eval_post.get("ap")), " F1@Youden:", fmt(eval_post.get("f1_y")))

# run_patchcore_al_fixed.py
import os, copy, random
from glob import glob
from typing import List, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# Optional deps
try:
    import timm
    HAVE_TIMM = True
except Exception:
    HAVE_TIMM = False

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


# --------------------------- utils ---------------------------
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def build_loader_from_indices(dataset, indices, batch=32, shuffle=True, num_workers=2, pin_memory=True):
    from torch.utils.data import DataLoader, Subset
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=batch, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

def _l2n(t):
    eps=1e-6
    n = torch.norm(t, p=2, dim=1, keepdim=True).clamp_min(eps)
    return t / n


# --------------------------- data ---------------------------
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class AnomalyDataset(Dataset):
    """
    Loads images from two folders:
      - normal_dir (label 0)
      - anomaly_dir (label 1)
    """
    def __init__(self, normal_dir: str, anomaly_dir: str, size: int = 256, color: str = "L"):
        self.size = size
        self.color = color.lower()

        def _collect(folder, label):
            if not folder or not os.path.isdir(folder):
                return []
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
            files = []
            for e in exts:
                files += glob(os.path.join(folder, "**", e), recursive=True)
            files = sorted(list(set(files)))
            return [(p, label) for p in files]

        samples = []
        samples += _collect(normal_dir, 0)
        samples += _collect(anomaly_dir, 1)
        if len(samples) == 0:
            raise RuntimeError("No images found in the provided directories.")
        self.samples = samples
        self.labels = [lab for (_, lab) in samples]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        img = img.convert("RGB") if self.color.lower()=="rgb" else img.convert("L")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img).astype("float32") / 255.0
        if arr.ndim == 2: arr = arr[:, :, None]
        if arr.shape[2] == 4: arr = arr[:, :, :3]
        arr = np.transpose(arr, (2, 0, 1)).copy()  # C,H,W
        tensor = torch.from_numpy(arr).float()
        return tensor, int(label)

def split_normal_30(dataset: AnomalyDataset, seed: int = 42) -> Tuple[List[int], List[int]]:
    normals = [i for i, lab in enumerate(dataset.labels) if lab == 0]
    rnd = random.Random(seed); rnd.shuffle(normals)
    n_seed = max(1, int(round(0.30 * len(normals))))
    idx_seed = normals[:n_seed]
    idx_rest_normals = normals[n_seed:]
    return idx_seed, idx_rest_normals


# --------------------------- PatchCore backbone/adapter ---------------------------
class PCBackbone(nn.Module):
    """
    Frozen multi-scale feature extractor.
    If timm is available, uses features_only model (e.g., resnet50) and returns maps from out_indices.
    Otherwise, a tiny CNN fallback.
    """
    def __init__(self, model_name="resnet50", out_indices=(2,3), device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.use_timm = HAVE_TIMM
        if self.use_timm:
            self.model = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=out_indices)
            self.model.eval().to(self.device)
            for p in self.model.parameters(): p.requires_grad_(False)
        else:
            # tiny fallback (grayscale)
            self.c1 = nn.Conv2d(1, 32, 3, 2, 1)
            self.c2 = nn.Conv2d(32,64, 3, 2, 1)
            self.c3 = nn.Conv2d(64,128,3, 2, 1)
            self.c4 = nn.Conv2d(128,256,3, 2, 1)
            self.out_indices = out_indices
            self.to(self.device)

    @torch.no_grad()
    def forward(self, x01_1chw: torch.Tensor):
        if self.use_timm:
            x = x01_1chw.repeat(1,3,1,1) if x01_1chw.size(1)==1 else x01_1chw
            feats = self.model(x)  # list of maps
            return feats
        else:
            x = x01_1chw
            f1 = F.relu(self.c1(x)); f2 = F.relu(self.c2(f1))
            f3 = F.relu(self.c3(f2)); f4 = F.relu(self.c4(f3))
            feats = [f2, f3, f4]
            return [feats[i] for i in self.out_indices]

class PCAdapter(nn.Module):
    """Tiny learnable head per selected scale."""
    def __init__(self, in_chs: list, out_dim=256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)
        ) for c in in_chs])

    def forward(self, feats):
        return [p(f) for p, f in zip(self.proj, feats)]


# --------------------------- SWAG ---------------------------
class SWAGAdapter:
    def __init__(self, adapter: nn.Module, noise_scale=0.01, max_snaps=30, device="cuda"):
        self.proto = adapter
        self.noise = noise_scale
        self.max_snaps = max_snaps
        self.snaps = []
        self.device = device

    def collect(self, adapter: nn.Module):
        sd = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        if len(self.snaps) >= self.max_snaps:
            self.snaps.pop(0)
        self.snaps.append(sd)

    def _avg(self):
        assert self.snaps, "No SWAG snapshots collected."
        keys = self.snaps[0].keys()
        avg = {}
        for k in keys:
            tensors = [s[k] for s in self.snaps]
            t0 = tensors[0]
            if t0.is_floating_point():
                stk = torch.stack([t.to(torch.float32) for t in tensors], 0)
                avg[k] = stk.mean(0)
            else:
                avg[k] = t0.clone()
        return avg

    @torch.no_grad()
    def sample_adapters(self, K=8):
        avg = self._avg()
        outs = []
        for _ in range(K):
            Ad = copy.deepcopy(self.proto).to(self.device)
            noisy = {}
            for k, v in avg.items():
                noisy[k] = v + torch.randn_like(v) * self.noise if v.is_floating_point() else v
            Ad.load_state_dict(noisy, strict=False)
            Ad.eval()
            outs.append(Ad)
        return outs


# --------------------------- Memory / Coreset ---------------------------
def coreset_greedy(X: torch.Tensor, ratio=0.2, seed=42):
    N = X.size(0); M = max(1, int(ratio*N))
    rng = np.random.RandomState(seed)
    start = int(rng.randint(N))
    chosen = [start]
    d2 = torch.cdist(X[start:start+1], X, p=2).squeeze(0)  # [N]
    for _ in range(1, M):
        idx = int(torch.argmax(d2).item())
        chosen.append(idx)
        d2 = torch.minimum(d2, torch.cdist(X[idx:idx+1], X, p=2).squeeze(0))
    return X[chosen], np.array(chosen, dtype=np.int64)

class MemoryBank:
    def __init__(self, coreset_ratio=0.2, method="auto", large_N=300_000, device="cpu"):
        self.mem = None
        self.ratio = coreset_ratio
        self.method = method
        self.large_N = large_N
        self.device = torch.device(device)

    def build(self, patch_vecs: torch.Tensor):
        N = patch_vecs.size(0)
        X = _l2n(patch_vecs.T).T.contiguous()
        if self.method == "random" or (self.method == "auto" and N > self.large_N):
            M = max(1, int(self.ratio*N))
            idx = torch.randperm(N)[:M]
            self.mem = X[idx].to(self.device, non_blocking=True)
            print(f"[PatchCore] Memory (random) {self.mem.size(0)}/{N}")
        else:
            Xd = X.to(self.device, non_blocking=True)
            self.mem, _ = coreset_greedy(Xd, ratio=self.ratio)
            self.mem = self.mem.contiguous()
            print(f"[PatchCore] Memory (greedy) {self.mem.size(0)}/{N}")

    @torch.no_grad()
    def knn_dist(self, Q: torch.Tensor, k=3, q_chunk=4096, m_chunk=100_000, prefer_gpu=True):
        assert self.mem is not None, "MemoryBank not built."
        mem = self.mem
        Nm = mem.size(0)
        mem_f = mem if mem.dtype == torch.float32 else mem.float()
        outs = []
        for qi in range(0, Q.size(0), q_chunk):
            q = Q[qi:qi+q_chunk]
            q_f = q if q.dtype == torch.float32 else q.float()
            best = torch.full((q_f.size(0), k), float("inf"))
            q_dev = q_f.to(mem_f.device if (prefer_gpu and mem_f.is_cuda) else torch.device("cpu"), non_blocking=True)
            m_dev_target = q_dev.device
            for mi in range(0, Nm, m_chunk):
                m_blk = mem_f[mi:mi+m_chunk]
                try:
                    m_dev = m_blk.to(m_dev_target, non_blocking=True)
                    d_blk = torch.cdist(q_dev, m_dev, p=2)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        m_dev = m_blk.cpu()
                        q_dev = q_f.cpu()
                        best = best.cpu()
                        d_blk = torch.cdist(q_dev, m_dev, p=2)
                        m_dev_target = torch.device("cpu")
                    else:
                        raise
                cat = torch.cat([best.to(d_blk.device), d_blk], dim=1)
                best = torch.topk(cat, k=k, largest=False, dim=1).values
            outs.append(best.mean(dim=1).cpu())
        return torch.cat(outs, dim=0)


# --------------------------- Feature extraction & scoring ---------------------------
def extract_patch_vectors(backbone, adapter, loader, device, target_hw=(16,16), per_image_cap=256):
    backbone.eval(); adapter.eval()
    vecs=[]
    with torch.no_grad():
        for x,_ in loader:
            x = x.to(device, non_blocking=True)
            feats = adapter(backbone(x))
            ups  = [F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P    = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            if per_image_cap is not None:
                B, H, W = x.size(0), target_hw[0], target_hw[1]
                patches_per_img = H*W
                if patches_per_img > per_image_cap:
                    P = P.view(B, patches_per_img, -1)[:, :per_image_cap, :].reshape(-1, P.size(1))
            vecs.append(P.cpu())
    return torch.cat(vecs, 0)

def score_images_patchcore(backbone, adapter, mem, loader, device, top_q=0.05, k=3,
                           out_size=(256,256), collect_images=True):
    backbone.eval(); adapter.eval()
    all_scores, all_heatmaps = [], []
    all_imgs = [] if collect_images else None
    with torch.no_grad():
        for x,_ in loader:
            x = x.to(device)
            feats = adapter(backbone(x))
            Hmax = max(f.size(2) for f in feats); Wmax = max(f.size(3) for f in feats)
            ups = [F.interpolate(f, size=(Hmax,Wmax), mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            d = mem.knn_dist(P, k=k).view(x.size(0), Hmax, Wmax)
            flat = d.view(x.size(0), -1)
            q = max(1, int(top_q*flat.size(1)))
            s = torch.topk(flat, k=q, dim=1).values.mean(1)
            d_up = F.interpolate(d.unsqueeze(1), size=out_size, mode="bilinear", align_corners=False).squeeze(1)
            all_scores.append(s.cpu()); all_heatmaps.append(d_up.cpu())
            if collect_images:
                all_imgs.append(F.interpolate(x, size=out_size, mode="bilinear", align_corners=False).cpu())
    scores = torch.cat(all_scores,0)
    heatmaps = torch.cat(all_heatmaps,0)
    imgs_opt = torch.cat(all_imgs,0) if collect_images else None
    return (scores, heatmaps, imgs_opt) if collect_images else (scores, heatmaps)

@torch.no_grad()
def swag_uncertainty(backbone, swag_adp: SWAGAdapter, mem: MemoryBank, loader, device, K=4, top_q=0.05, k=3):
    all_vars=[]
    for x,_ in loader:
        x = x.to(device)
        s_ks=[]
        Adapters = swag_adp.sample_adapters(K=K)
        for Ad in Adapters:
            feats = Ad(backbone(x))
            Hmax = max(f.size(2) for f in feats); Wmax = max(f.size(3) for f in feats)
            ups = [F.interpolate(f, size=(Hmax,Wmax), mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            d = mem.knn_dist(P, k=k).view(x.size(0), Hmax, Wmax)
            flat = d.view(x.size(0), -1)
            q = max(1, int(top_q*flat.size(1)))
            s = torch.topk(flat, k=q, dim=1).values.mean(1)
            s_ks.append(s.cpu())
        S = torch.stack(s_ks,0)
        all_vars.append(S.var(0))
    return torch.cat(all_vars,0)


# --------------------------- Tiny adapter training ---------------------------
def train_adapter_compact(backbone, adapter, loader, device, epochs=3, lr=1e-4, proto_k=128):
    backbone.eval(); adapter.train()
    with torch.no_grad():
        vecs=[]
        for x,_ in loader:
            x=x.to(device)
            feats = adapter(backbone(x))
            Hmax = max(f.size(2) for f in feats); Wmax = max(f.size(3) for f in feats)
            ups = [F.interpolate(f, (Hmax,Wmax), mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            vecs.append(P.cpu())
        V = torch.cat(vecs,0)
    protos, _ = coreset_greedy(V, ratio=min(1.0, proto_k/max(1,V.size(0))))
    protos = protos.to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    for ep in range(epochs):
        tot=0; n=0
        for x,_ in loader:
            x=x.to(device)
            feats = adapter(backbone(x))
            Hmax = max(f.size(2) for f in feats); Wmax = max(f.size(3) for f in feats)
            ups = [F.interpolate(f, (Hmax,Wmax), mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            D = torch.cdist(P, protos, p=2)
            loss = D.min(1).values.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item(); n+=1
        print(f"[Adapter] ep {ep+1}/{epochs} loss={tot/max(1,n):.4f}")


# --------------------------- Active learning (FIXED) ---------------------------
def _scores_to_numpy(ret):
    if isinstance(ret, (tuple, list)):
        ret = ret[0]
    if hasattr(ret, "detach"):
        return ret.detach().cpu().numpy()
    return np.asarray(ret)

def active_learning_patchcore_swag_fixed(
    backbone, adapter, dataset, seed_indices_normals, pool_indices, device,
    rounds=3, budget=50, mem_coreset=0.2, swag_noise=0.005, swag_K=4,
    top_q=0.05, k=3, p_gate=0.95, save_dir="runs/patchcore_al",
    use_val_guard=False, val_loader=None, val_labels=None, guard_tol=0.01
):
    os.makedirs(save_dir, exist_ok=True)

    # 1) seed adapter train + SWAG
    seed_loader = build_loader_from_indices(dataset, seed_indices_normals, batch=32, shuffle=True)
    train_adapter_compact(backbone, adapter, seed_loader, device, epochs=3, lr=1e-4, proto_k=128)
    swag = SWAGAdapter(adapter, noise_scale=swag_noise, max_snaps=30, device=device)
    swag.collect(copy.deepcopy(adapter).eval()); swag.collect(copy.deepcopy(adapter).eval())

    # 2) frozen scoring memory from seeds (prevents contamination)
    seed_loader_eval = build_loader_from_indices(dataset, seed_indices_normals, batch=32, shuffle=False)
    V_seed = extract_patch_vectors(backbone, adapter, seed_loader_eval, device, target_hw=(16,16), per_image_cap=256)
    mem_seed = MemoryBank(coreset_ratio=mem_coreset, method="auto", device=device)
    mem_seed.build(V_seed)

    # 3) calibrate z-score gates from seeds
    s_seed_np = _scores_to_numpy(score_images_patchcore(backbone, adapter, mem_seed, seed_loader_eval, device, top_q=top_q, k=k))
    u_seed_np = swag_uncertainty(backbone, swag, mem_seed, seed_loader_eval, device, K=swag_K, top_q=top_q, k=k).cpu().numpy()
    mu_s, sd_s = float(np.mean(s_seed_np)), float(np.std(s_seed_np) + 1e-8)
    mu_u, sd_u = float(np.mean(u_seed_np)), float(np.std(u_seed_np) + 1e-8)
    print(f"[Calib] μ_s={mu_s:.4f} σ_s={sd_s:.4f} | μ_u={mu_u:.4f} σ_u={sd_u:.4f}")

    used = set()
    val_auc_prev = -np.inf

    for r in range(1, rounds+1):
        print(f"\n=== Round {r} ===")
        rem_local = [i for i in range(len(pool_indices)) if i not in used]
        if not rem_local:
            print("Pool exhausted."); break
        rem_global = [pool_indices[i] for i in rem_local]
        rem_loader = build_loader_from_indices(dataset, rem_global, batch=32, shuffle=False)

        # Score/uncert with frozen memory
        s_pool_np = _scores_to_numpy(score_images_patchcore(backbone, adapter, mem_seed, rem_loader, device, top_q=top_q, k=k))
        u_pool_np = swag_uncertainty(backbone, swag, mem_seed, rem_loader, device, K=swag_K, top_q=top_q, k=k).cpu().numpy()

        z_s = (s_pool_np - mu_s)/sd_s
        z_u = (u_pool_np - mu_u)/sd_u

        cand = np.nonzero((z_s <= 1.0) & (z_u <= 1.0))[0].tolist()
        if not cand:
            print("[Round] No safe candidates at 1.0σ; relaxing to 1.5σ once.")
            cand = np.nonzero((z_s <= 1.5) & (z_u <= 1.5))[0].tolist()
        take = cand[: min(budget, len(cand))]
        print(f"[Round {r}] Safe candidates: {len(cand)} | Taking: {len(take)} / budget {budget}")

        if not take:
            print("[Round] Nothing to add this round.")
            continue

        for li in take: used.add(rem_local[li])
        new_global = [rem_global[i] for i in take]

        # brief adapter tune on accepted samples
        round_loader = build_loader_from_indices(dataset, new_global, batch=32, shuffle=True)
        train_adapter_compact(backbone, adapter, round_loader, device, epochs=1, lr=5e-5, proto_k=128)
        swag.collect(copy.deepcopy(adapter).eval())

        # Optional: validation guardrail
        if use_val_guard and val_loader is not None and val_labels is not None and HAVE_SKLEARN:
            val_scores = _scores_to_numpy(score_images_patchcore(backbone, adapter, mem_seed, val_loader, device, top_q=top_q, k=k))
            val_auc = roc_auc_score(val_labels, val_scores)
            print(f"[Round {r}] Val AUC: {val_auc:.4f} (prev {val_auc_prev:.4f})")
            if val_auc < val_auc_prev - guard_tol:
                print("[AL] Early stop: validation AUC decreased.")
                break
            val_auc_prev = val_auc

    # Return adapter and the **seed** memory used for scoring
    return adapter, mem_seed


# --------------------------- Eval/plots ---------------------------
# def evaluate_and_plot(labels_all, scores, prefix="", outdir="runs/patchcore_al", target_precision=0.80):
#     os.makedirs(outdir, exist_ok=True)
#     if not HAVE_SKLEARN:
#         print("(Metrics skipped: scikit-learn not available.)")
#         return {}

#     auc = roc_auc_score(labels_all, scores)
#     fpr, tpr, thr = roc_curve(labels_all, scores)
#     ap  = average_precision_score(labels_all, scores)

#     J = tpr - fpr
#     jbest = int(np.nanargmax(J))
#     thr_best = thr[jbest] if np.isfinite(thr[jbest]) else np.quantile(scores, 0.95)
#     preds_y = (scores >= thr_best).astype(np.int32)

#     acc_y = accuracy_score(labels_all, preds_y)
#     prec_y = precision_score(labels_all, preds_y, zero_division=0)
#     rec_y  = recall_score(labels_all, preds_y, zero_division=0)
#     f1_y   = f1_score(labels_all, preds_y, zero_division=0)
#     cm_y   = confusion_matrix(labels_all, preds_y)

#     pr, rc, thr_pr = precision_recall_curve(labels_all, scores)
#     mask = pr[1:] >= target_precision
#     thr_op = thr_pr[mask][-1] if np.any(mask) else thr_pr[np.argmax(pr[1:])]
#     preds_o = (scores >= thr_op).astype(np.int32)

#     acc_o = accuracy_score(labels_all, preds_o)
#     prec_o = precision_score(labels_all, preds_o, zero_division=0)
#     rec_o  = recall_score(labels_all, preds_o, zero_division=0)
#     f1_o   = f1_score(labels_all, preds_o, zero_division=0)
#     cm_o   = confusion_matrix(labels_all, preds_o)

#     print(f"\n=== {prefix} Evaluation ===")
#     print(f"ROC AUC : {auc:.4f}")
#     print(f"PR  AUC : {ap:.4f}")
#     print(f"Thr* (Youden) : {thr_best:.6f}")
#     print(f"  ACC={acc_y:.4f}  Precision={prec_y:.4f}  Recall={rec_y:.4f}  F1={f1_y:.4f}")
#     print("  Confusion (Youden) [tn fp; fn tp]:\n", cm_y)

#     print(f"\nThr@P>={target_precision:.2f} : {thr_op:.6f}")
#     print(f"  ACC={acc_o:.4f}  Precision={prec_o:.4f}  Recall={rec_o:.4f}  F1={f1_o:.4f}")
#     print("  Confusion (OpPt) [tn fp; fn tp]:\n", cm_o)

#     if HAVE_PLT:
#         plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
#         plt.plot([0,1],[0,1],"k--"); plt.legend(); plt.title(f"ROC ({prefix})"); plt.xlabel("FPR"); plt.ylabel("TPR")
#         plt.savefig(os.path.join(outdir, f"roc_{prefix}.png"), dpi=120); plt.close()

#         plt.figure(); plt.plot(rc, pr, label=f"AP={ap:.4f}")
#         plt.title(f"Precision-Recall ({prefix})"); plt.xlabel("Recall"); plt.ylabel("Precision")
#         plt.savefig(os.path.join(outdir, f"pr_{prefix}.png"), dpi=120); plt.close()

#     return {
#         "auc": auc, "ap": ap, "thr_youden": thr_best, "thr_opt": thr_op,
#         "acc_y": acc_y, "prec_y": prec_y, "rec_y": rec_y, "f1_y": f1_y,
#         "acc_o": acc_o, "prec_o": prec_o, "rec_o": rec_o, "f1_o": f1_o
#     }

def evaluate_and_plot(labels_all, scores, prefix="", outdir="runs/patchcore_al", target_precision=0.80):
    os.makedirs(outdir, exist_ok=True)
    if not HAVE_SKLEARN:
        print("(Metrics skipped: scikit-learn not available.)")
        return {}

    # --- core metrics ---
    auc = roc_auc_score(labels_all, scores)
    fpr, tpr, thr = roc_curve(labels_all, scores)
    ap  = average_precision_score(labels_all, scores)

    # Youden
    J = tpr - fpr
    jbest = int(np.nanargmax(J))
    thr_best = thr[jbest] if np.isfinite(thr[jbest]) else np.quantile(scores, 0.95)
    preds_y = (scores >= thr_best).astype(np.int32)
    acc_y = accuracy_score(labels_all, preds_y)
    prec_y = precision_score(labels_all, preds_y, zero_division=0)
    rec_y  = recall_score(labels_all, preds_y, zero_division=0)
    f1_y   = f1_score(labels_all, preds_y, zero_division=0)
    cm_y   = confusion_matrix(labels_all, preds_y)

    # Operating point by target precision
    pr, rc, thr_pr = precision_recall_curve(labels_all, scores)
    mask = pr[1:] >= target_precision
    thr_op = thr_pr[mask][-1] if np.any(mask) else thr_pr[np.argmax(pr[1:])]
    preds_o = (scores >= thr_op).astype(np.int32)
    acc_o = accuracy_score(labels_all, preds_o)
    prec_o = precision_score(labels_all, preds_o, zero_division=0)
    rec_o  = recall_score(labels_all, preds_o, zero_division=0)
    f1_o   = f1_score(labels_all, preds_o, zero_division=0)
    cm_o   = confusion_matrix(labels_all, preds_o)

    # --- prints ---
    print(f"\n=== {prefix} Evaluation ===")
    print(f"ROC AUC : {auc:.4f}")
    print(f"PR  AUC : {ap:.4f}")
    print(f"Thr* (Youden) : {thr_best:.6f}")
    print(f"  ACC={acc_y:.4f}  Precision={prec_y:.4f}  Recall={rec_y:.4f}  F1={f1_y:.4f}")
    print("  Confusion (Youden) [tn fp; fn tp]:\n", cm_y)

    print(f"\nThr@P>={target_precision:.2f} : {thr_op:.6f}")
    print(f"  ACC={acc_o:.4f}  Precision={prec_o:.4f}  Recall={rec_o:.4f}  F1={f1_o:.4f}")
    print("  Confusion (OpPt) [tn fp; fn tp]:\n", cm_o)

    # --- plots (ROC, PR, Confusion Matrices) ---
    if HAVE_PLT:
        # ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0,1],[0,1],"k--")
        plt.legend(); plt.title(f"ROC ({prefix})"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.savefig(os.path.join(outdir, f"roc_{prefix}.png"), dpi=120); plt.close()

        # PR
        plt.figure()
        plt.plot(rc, pr, label=f"AP={ap:.4f}")
        plt.title(f"Precision-Recall ({prefix})"); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.savefig(os.path.join(outdir, f"pr_{prefix}.png"), dpi=120); plt.close()

        # Confusion matrix helper
        def _plot_cm(cm, title, fname):
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Normal (0)', 'Anomaly (1)'])
            plt.yticks(tick_marks, ['Normal (0)', 'Anomaly (1)'])
            thresh = cm.max() / 2.0 if cm.size else 0.5
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, fname), dpi=140)
            plt.close()

        # Plot both confusion matrices
        _plot_cm(cm_y, f"Confusion (Youden) — {prefix}", f"cm_youden_{prefix}.png")
        _plot_cm(cm_o, f"Confusion (OpPt P≥{target_precision:.2f}) — {prefix}", f"cm_opt_{prefix}.png")

    return {
        "auc": auc, "ap": ap, "thr_youden": thr_best, "thr_opt": thr_op,
        "acc_y": acc_y, "prec_y": prec_y, "rec_y": rec_y, "f1_y": f1_y, "cm_y": cm_y,
        "acc_o": acc_o, "prec_o": prec_o, "rec_o": rec_o, "f1_o": f1_o, "cm_o": cm_o,
    }

# --------------------------- MAIN ---------------------------
if __name__ == "__main__":
    set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ====== EDIT THESE PATHS (absolute paths recommended) ======
    normal_dir_train  = "Data_XRAY/train/NORMAL"#"chest_xray/train/NORMAL"#"/chest_xray/train/NORMAL"
    anomaly_dir_train = "Data_XRAY/train/COVID19"#"chest_xray/train/PNEUMONIA"#"/chest_xray/train/PNEUMONIA"
    normal_dir_test   = "Data_XRAY/test/NORMAL" #"/chest_xray/test/NORMAL"
    anomaly_dir_test  ="Data_XRAY/test/COVID19"#"/chest_xray/test/PNEUMONIA"



    # ===========================================================

    size = 256
    batch = 32
    DATASET_LABEL_ONE_IS_ANOMALY = True
    TARGET_PRECISION = 0.70

    # Datasets
    dataset = AnomalyDataset(normal_dir_train, anomaly_dir_train, size=size, color="L")
    idx_seed, idx_rest_normals = split_normal_30(dataset, seed=42)
    idx_anoms = [i for i in range(len(dataset)) if dataset.labels[i] == 1]
    pool_indices = idx_rest_normals + idx_anoms

    test_dataset = AnomalyDataset(normal_dir_test, anomaly_dir_test, size=size, color="L")
    test_loader  = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    # Backbone + adapter
    backbone = PCBackbone(model_name="resnet50", out_indices=(2,3), device=device)
    with torch.no_grad():
        x0, _ = dataset[0]
        x0 = x0.unsqueeze(0).to(device)
        feats0 = backbone(x0)
        in_chs = [f.size(1) for f in feats0]
    adapter = PCAdapter(in_chs=in_chs, out_dim=256).to(device)

    # -------- Baseline memory from whole train set (for baseline evaluation) --------
    print("\n=== Baseline evaluation (before AL) ===")
    train_loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    feats_all = []
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            feats = adapter(backbone(x))
            Hmax = max(f.size(2) for f in feats); Wmax = max(f.size(3) for f in feats)
            ups = [F.interpolate(f, size=(Hmax,Wmax), mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            feats_all.append(P.cpu())
    feats_all = torch.cat(feats_all,0).contiguous().float()

    memory_baseline = MemoryBank(coreset_ratio=0.3, method="auto", device=device)
    memory_baseline.build(feats_all)

    s_base, heat_base, imgs_base = score_images_patchcore(
        backbone, adapter, memory_baseline, test_loader, device,
        top_q=0.05, k=3, out_size=(256,256), collect_images=True
    )

    labels_base = []
    for _, y in test_loader: labels_base.extend(y.tolist())
    labels_base = np.asarray(labels_base, dtype=np.int32)
    if not DATASET_LABEL_ONE_IS_ANOMALY: labels_base = 1 - labels_base

    eval_base = evaluate_and_plot(
        labels_base, s_base.numpy(), prefix="baseline",
        outdir="runs/patchcore_al/baseline", target_precision=TARGET_PRECISION
    )

    os.makedirs("runs/patchcore_al/baseline_vis", exist_ok=True)
    if HAVE_PLT:
        try:
            # simple preview
            idx_vis = 0
            img_v = imgs_base[idx_vis].permute(1,2,0).numpy()
            hm_v  = heat_base[idx_vis].numpy()
            h = (hm_v - hm_v.min())/(hm_v.max()-hm_v.min()+1e-8)
            cmap = plt.get_cmap('jet')
            overlay = (1-0.5)*np.repeat(img_v[..., :1], 3, axis=2) + 0.5*cmap(h)[..., :3]
            plt.imsave("runs/patchcore_al/baseline_vis/example.png", overlay)
            print("Saved:", "runs/patchcore_al/baseline_vis/example.png")
        except Exception as e:
            print("Baseline vis failed:", e)

    # -------- Active learning (fixed) --------
    adapter_trained, memory_seed = active_learning_patchcore_swag_fixed(
        backbone, adapter, dataset,
        seed_indices_normals=idx_seed,
        pool_indices=pool_indices,
        device=device,
        rounds=5,          # you can lower to 3 if needed
        budget=50,         # smaller budget per round
        mem_coreset=0.3,
        swag_noise=0.005,
        swag_K=4,
        top_q=0.05,
        k=3,
        p_gate=0.95,
        save_dir="runs/patchcore_al",
        use_val_guard=False,   # set True if you provide val_loader & val_labels
        val_loader=None,
        val_labels=None,
        guard_tol=0.01
    )

    # -------- Post-AL evaluation (with FROZEN seed memory) --------
    print("\n=== Post-AL evaluation (after AL) ===")
    s_post, heat_post, imgs_post = score_images_patchcore(
        backbone, adapter_trained, memory_seed, test_loader, device,
        top_q=0.05, k=3, out_size=(256,256), collect_images=True
    )

    labels_post = []
    for _, y in test_loader: labels_post.extend(y.tolist())
    labels_post = np.asarray(labels_post, dtype=np.int32)
    if not DATASET_LABEL_ONE_IS_ANOMALY: labels_post = 1 - labels_post

    eval_post = evaluate_and_plot(
        labels_post, s_post.numpy(), prefix="postAL",
        outdir="runs/patchcore_al/postAL", target_precision=TARGET_PRECISION
    )

    # -------- Summary --------
    def fmt(v): 
        try: return f"{v:.4f}"
        except: return "nan"
    print("\n=== Summary ===")
    print("Baseline:  AUC:", fmt(eval_base.get("auc")), " AP:", fmt(eval_base.get("ap")), " F1@Youden:", fmt(eval_base.get("f1_y")))
    print("Post-AL :  AUC:", fmt(eval_post.get("auc")), " AP:", fmt(eval_post.get("ap")), " F1@Youden:", fmt(eval_post.get("f1_y")))
