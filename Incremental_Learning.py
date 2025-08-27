# run_patchcore_al_fixed.py
# Fair baseline + improved active learning (boundary ranking + growing memory)

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


# --------------------------- Metrics / plots ---------------------------
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


# --------------------------- AL metric (improved) ---------------------------
@torch.no_grad()
def _metric_score_to_maximize(
    backbone, adapter, mem_normal, seed_loader_eval, device,
    dataset=None, pool_indices_for_metric=None, pool_sample=512,
    val_loader=None, val_labels=None
):
    """
    Returns a scalar to MAXIMIZE:

    1) If you have labels, use validation ROC-AUC.
    2) Else, unsupervised separation proxy: mean(pool_scores) - mean(seed_scores)
       on a small, fixed subset of the pool (if provided), otherwise -mean(seed_scores).
    """
    # (1) Supervised: proper validation AUC
    if val_loader is not None and val_labels is not None and HAVE_SKLEARN:
        s_val = score_images_patchcore(backbone, adapter, mem_normal, val_loader, device, top_q=0.05, k=3)[0]
        auc = roc_auc_score(val_labels, s_val.cpu().numpy())
        return float(auc)

    # (2) Unsupervised proxy
    s_seed = score_images_patchcore(backbone, adapter, mem_normal, seed_loader_eval, device, top_q=0.05, k=3)[0]
    mean_seed = float(torch.mean(s_seed).item())

    if dataset is not None and pool_indices_for_metric is not None and len(pool_indices_for_metric) > 0:
        sel = pool_indices_for_metric
        if len(sel) > pool_sample:
            rnd = np.random.RandomState(123)
            sel = list(rnd.choice(sel, size=pool_sample, replace=False))
        pool_loader_eval = build_loader_from_indices(dataset, sel, batch=64, shuffle=False)
        s_pool = score_images_patchcore(backbone, adapter, mem_normal, pool_loader_eval, device, top_q=0.05, k=3)[0]
        mean_pool = float(torch.mean(s_pool).item())
        return float(mean_pool - mean_seed)

    # Fallback: prefer lower seed score
    return float(-mean_seed)


# --------------------------- Active learning (improved) ---------------------------
def _scores_to_numpy(ret):
    if isinstance(ret, (tuple, list)):
        ret = ret[0]
    if hasattr(ret, "detach"):
        return ret.detach().cpu().numpy()
    return np.asarray(ret)

def active_learning_patchcore_swag_fixed(
    backbone, adapter, dataset, seed_indices_normals, pool_indices, device,
    rounds=5, budget=100 ,mem_coreset=0.3, swag_noise=0.02, swag_K=4,
    top_q=0.03, k=3, p_gate=0.95, save_dir="runs/patchcore_al_best",
    use_val_guard=False, val_loader=None, val_labels=None, guard_tol=0.01,
    strict_normal_only=True, resume_each_round_from="best_so_far",
    rank_mode="boundary",                    # "boundary" (recommended) or "uncert"
    metric_pool_indices=None, pool_sample_for_metric=512
):
    """
    Active learning with:
      - boundary-based ranking inside a safe gate
      - normals-only fine-tuning (optional)
      - memory grows with accepted normals (seed ∪ accepted)
      - per-round checkpointing + resume-from-best
      - robust unsupervised metric (seed vs. pool separation)
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = os.path.join(save_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Seed loaders
    seed_loader      = build_loader_from_indices(dataset, seed_indices_normals, batch=32, shuffle=True)
    seed_loader_eval = build_loader_from_indices(dataset, seed_indices_normals, batch=64, shuffle=False)

    # --- Warm-up adapter on seed normals
    train_adapter_compact(backbone, adapter, seed_loader, device, epochs=3, lr=1e-4, proto_k=128)

    # --- SWAG bootstrap
    swag = SWAGAdapter(adapter, noise_scale=swag_noise, max_snaps=30, device=device)
    swag.collect(copy.deepcopy(adapter).eval()); swag.collect(copy.deepcopy(adapter).eval())

    # --- Build initial memory from seed normals
    V_seed = extract_patch_vectors(backbone, adapter, seed_loader_eval, device, target_hw=(16,16), per_image_cap=256)
    mem_normal = MemoryBank(coreset_ratio=mem_coreset, method="auto", device=device)
    mem_normal.build(V_seed)

    # --- Calibrate on seeds
    s_seed_np = _scores_to_numpy(score_images_patchcore(backbone, adapter, mem_normal, seed_loader_eval, device, top_q=top_q, k=k))
    u_seed_np = swag_uncertainty(backbone, swag, mem_normal, seed_loader_eval, device, K=swag_K, top_q=top_q, k=k).cpu().numpy()
    mu_s, sd_s = float(np.mean(s_seed_np)), float(np.std(s_seed_np) + 1e-8)
    mu_u, sd_u = float(np.mean(u_seed_np)), float(np.std(u_seed_np) + 1e-8)
    use_uncert_gate = (sd_u > 1e-6)   # ignore u if numerically zero
    print(f"[Calib] μ_s={mu_s:.4f} σ_s={sd_s:.6f} | μ_u={mu_u:.6f} σ_u={sd_u:.6f} | use_u_gate={use_uncert_gate}")

    # --- Best tracking
    best_path = os.path.join(ckpt_dir, "best_overall.pth")
    best_score = _metric_score_to_maximize(
        backbone, adapter, mem_normal, seed_loader_eval, device,
        dataset=dataset, pool_indices_for_metric=metric_pool_indices, pool_sample=pool_sample_for_metric,
        val_loader=val_loader, val_labels=val_labels
    )
    torch.save({"state_dict": adapter.state_dict()}, best_path)
    print(f"[CKPT] initial best_overall = {best_score:.6f}")

    # --- State across rounds
    used_local = set()
    accepted_normals_all: List[int] = []
    val_auc_prev = -np.inf

    for r in range(1, rounds+1):
        print(f"\n=== Round {r} ===")

        # Resume policy
        if resume_each_round_from == "best_so_far" and os.path.isfile(best_path):
            adapter.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
            print("[CKPT] resumed from best_overall")
        elif resume_each_round_from == "last":
            last_path = os.path.join(ckpt_dir, f"round{r-1}_last.pth")
            if os.path.isfile(last_path):
                adapter.load_state_dict(torch.load(last_path, map_location=device)["state_dict"])
                print(f"[CKPT] resumed from {last_path}")

        # Rebuild memory from seed ∪ accepted_normals (adapter may have changed)
        mem_ids = seed_indices_normals + accepted_normals_all
        mem_loader_eval = build_loader_from_indices(dataset, mem_ids, batch=64, shuffle=False)
        V_mem = extract_patch_vectors(backbone, adapter, mem_loader_eval, device, target_hw=(16,16), per_image_cap=256)
        mem_normal.build(V_mem)

        # Remaining pool
        rem_local  = [i for i in range(len(pool_indices)) if i not in used_local]
        if not rem_local:
            print("Pool exhausted."); break
        rem_global = [pool_indices[i] for i in rem_local]
        rem_loader = build_loader_from_indices(dataset, rem_global, batch=64, shuffle=False)

        # Score + (optional) uncertainty on pool
        s_pool_np = _scores_to_numpy(score_images_patchcore(backbone, adapter, mem_normal, rem_loader, device, top_q=top_q, k=k))
        if use_uncert_gate:
            u_pool_np = swag_uncertainty(backbone, swag, mem_normal, rem_loader, device, K=swag_K, top_q=top_q, k=k).cpu().numpy()
        else:
            u_pool_np = np.zeros_like(s_pool_np)

        z_s = (s_pool_np - mu_s)/sd_s
        z_u = (u_pool_np - mu_u)/(sd_u if sd_u>0 else 1.0)

        # Safe gate
        if use_uncert_gate:
            safe_idx = np.nonzero((z_s <= 1.0) & (z_u <= 1.0))[0]
            if safe_idx.size == 0:
                print("[Round] no safe candidates at 1.0σ; relaxing to 1.5σ once.")
                safe_idx = np.nonzero((z_s <= 1.5) & (z_u <= 1.5))[0]
        else:
            safe_idx = np.nonzero(z_s <= 1.0)[0]
            if safe_idx.size == 0:
                print("[Round] no safe candidates at 1.0σ; relaxing to 1.5σ once.")
                safe_idx = np.nonzero(z_s <= 1.5)[0]

        if safe_idx.size == 0:
            print("[Round] still none; skipping.")
            continue

        # Ranking inside safe region
        if rank_mode == "boundary":
            order = np.argsort(-s_pool_np[safe_idx])  # high score → close to boundary
        else:  # "uncert"
            order = np.argsort(-z_u[safe_idx])       # higher uncertainty first
        take_pool_idx = safe_idx[order][: min(budget, safe_idx.size)]
        print(f"[Round {r}] Safe={safe_idx.size} | Taking={take_pool_idx.size} / budget {budget}")

        # Bookkeeping: mark used
        for i_pool in take_pool_idx:
            used_local.add(rem_local[int(i_pool)])
        selected_globals = [rem_global[i] for i in take_pool_idx]

        # Only normals for adapter updates (recommended for anomaly detection)
        if strict_normal_only:
            n_before = len(selected_globals)
            selected_globals = [g for g in selected_globals if dataset.labels[g] == 0]
            print(f"[Round {r}] selected={n_before} → normals_only={len(selected_globals)}")
            if len(selected_globals) == 0:
                # save "last" and continue
                torch.save({"state_dict": adapter.state_dict()}, os.path.join(ckpt_dir, f"round{r}_last.pth"))
                print("[Round] no normals to fine-tune on; skipping update.")
                continue

        # Fine-tune briefly on accepted normals
        round_loader = build_loader_from_indices(dataset, selected_globals, batch=32, shuffle=True)
        train_adapter_compact(backbone, adapter, round_loader, device, epochs=5, lr=3e-5, proto_k=128)
        swag.collect(copy.deepcopy(adapter).eval())

        # Grow accepted set & rebuild memory (seed ∪ accepted_normals)
        accepted_normals_all.extend(selected_globals)
        mem_ids = seed_indices_normals + accepted_normals_all
        mem_loader_eval = build_loader_from_indices(dataset, mem_ids, batch=64, shuffle=False)
        V_mem = extract_patch_vectors(backbone, adapter, mem_loader_eval, device, target_hw=(16,16), per_image_cap=256)
        mem_normal.build(V_mem)

        # Save "last" for this round
        torch.save({"state_dict": adapter.state_dict()}, os.path.join(ckpt_dir, f"round{r}_last.pth"))

        # Metric & best-overall
        metric_now = _metric_score_to_maximize(
            backbone, adapter, mem_normal, seed_loader_eval, device,
            dataset=dataset, pool_indices_for_metric=metric_pool_indices,
            pool_sample=pool_sample_for_metric, val_loader=val_loader, val_labels=val_labels
        )
        if metric_now > best_score:
            best_score = metric_now
            torch.save({"state_dict": adapter.state_dict()}, best_path)
            print(f"[CKPT] new best_overall = {best_score:.6f} at round {r}")

        # Optional validation guard
        if use_val_guard and val_loader is not None and val_labels is not None and HAVE_SKLEARN:
            val_scores = _scores_to_numpy(score_images_patchcore(backbone, adapter, mem_normal, val_loader, device, top_q=top_q, k=k))
            val_auc = roc_auc_score(val_labels, val_scores)
            print(f"[Round {r}] Val AUC: {val_auc:.4f} (prev {val_auc_prev:.4f})")
            if val_auc < val_auc_prev - guard_tol:
                print("[AL] Early stop: validation AUC decreased.")
                break
            val_auc_prev = val_auc

    # Restore best for final evaluation
    adapter.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    print(f"[CKPT] restored best_overall = {best_score:.6f} for final eval.")
    return adapter, mem_normal

def active_learning_oracle_free(
    backbone, adapter, dataset, seed_indices_normals, pool_indices, device,
    rounds=5, budget=50, mem_coreset=0.3, swag_noise=0.02, swag_K=8,
    top_q=0.03, k=3, save_dir="runs/patchcore_oraclefree",
    tau_z_initial=1.0, tau_z_relaxed=1.5,
    rank_mode="safe",              # "safe" (low z_s & z_u first) or "boundary" (high s first)
    ws=0.7, wu=0.3,                # weights for z_s and z_u (only used in "safe" mode)
    use_staging=True, staging_patience=2  # must pass the gate for 2 rounds before promotion
):
    """
    Oracle-free AL:
      - No label checks on the pool.
      - Dual z-score gates: accept only low distance & low SWAG uncertainty.
      - Optional 'staging' — item must be safe in >=2 rounds before being added to memory & used for fine-tuning.
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = os.path.join(save_dir, "ckpts"); os.makedirs(ckpt_dir, exist_ok=True)

    # --- seed loaders & warm-up ---
    seed_loader      = build_loader_from_indices(dataset, seed_indices_normals, batch=32, shuffle=True)
    seed_loader_eval = build_loader_from_indices(dataset, seed_indices_normals, batch=64, shuffle=False)
    train_adapter_compact(backbone, adapter, seed_loader, device, epochs=3, lr=1e-4, proto_k=128)

    # --- SWAG ---
    swag = SWAGAdapter(adapter, noise_scale=swag_noise, max_snaps=30, device=device)
    swag.collect(copy.deepcopy(adapter).eval()); swag.collect(copy.deepcopy(adapter).eval())

    # --- initial memory from seeds ---
    V_seed = extract_patch_vectors(backbone, adapter, seed_loader_eval, device, target_hw=(16,16), per_image_cap=256)
    mem_normal = MemoryBank(coreset_ratio=mem_coreset, method="auto", device=device)
    mem_normal.build(V_seed)

    # --- calibrate gates on seed normals ---
    s_seed_np = score_images_patchcore(backbone, adapter, mem_normal, seed_loader_eval, device, top_q=top_q, k=k)[0].cpu().numpy()
    u_seed_np = swag_uncertainty(backbone, swag, mem_normal, seed_loader_eval, device, K=swag_K, top_q=top_q, k=k).cpu().numpy()
    mu_s, sd_s = float(np.mean(s_seed_np)), float(np.std(s_seed_np) + 1e-8)
    mu_u, sd_u = float(np.mean(u_seed_np)), float(np.std(u_seed_np) + 1e-8)
    use_u = (sd_u > 1e-6)
    print(f"[Calib] mu_s={mu_s:.6f} sd_s={sd_s:.6f} | mu_u={mu_u:.6f} sd_u={sd_u:.6f} | use_u={use_u}")

    # --- state ---
    used_local = set()
    accepted = []               # promoted (unlabeled) items → memory
    staged   = {}               # global_idx -> consecutive safe passes
    tau_z = tau_z_initial

    for r in range(1, rounds+1):
        print(f"\n=== Round {r} (tau_z={tau_z}) ===")

        # rebuild memory from seeds ∪ accepted
        mem_ids = seed_indices_normals + accepted
        mem_loader_eval = build_loader_from_indices(dataset, mem_ids, batch=64, shuffle=False)
        V_mem = extract_patch_vectors(backbone, adapter, mem_loader_eval, device, target_hw=(16,16), per_image_cap=256)
        mem_normal.build(V_mem)

        # remaining pool
        rem_local  = [i for i in range(len(pool_indices)) if i not in used_local]
        if not rem_local:
            print("Pool exhausted."); break
        rem_global = [pool_indices[i] for i in rem_local]
        rem_loader = build_loader_from_indices(dataset, rem_global, batch=64, shuffle=False)

        # scores & (optional) uncertainty
        s_pool = score_images_patchcore(backbone, adapter, mem_normal, rem_loader, device, top_q=top_q, k=k)[0].cpu().numpy()
        if use_u:
            u_pool = swag_uncertainty(backbone, swag, mem_normal, rem_loader, device, K=swag_K, top_q=top_q, k=k).cpu().numpy()
        else:
            u_pool = np.zeros_like(s_pool)

        z_s = (s_pool - mu_s) / (sd_s if sd_s > 0 else 1.0)
        z_u = (u_pool - mu_u) / (sd_u if sd_u > 0 else 1.0)

        # dual safe gate (relax once if empty)
        safe_idx = np.nonzero((z_s <= tau_z) & ((z_u <= tau_z) if use_u else True))[0]
        if safe_idx.size == 0 and tau_z < tau_z_relaxed:
            print("[Round] relaxing gate once.")
            tau_z = tau_z_relaxed
            safe_idx = np.nonzero((z_s <= tau_z) & ((z_u <= tau_z) if use_u else True))[0]

        if safe_idx.size == 0:
            print("[Round] still none; skipping.")
            continue

        # ranking inside safe set
        if rank_mode == "safe":
            rank_score = ws * z_s[safe_idx] + (wu * z_u[safe_idx] if use_u else 0.0)  # lower is safer
            order = np.argsort(rank_score)
        else:  # "boundary"
            order = np.argsort(-s_pool[safe_idx])  # high score first
        take_idx = safe_idx[order][:min(budget, safe_idx.size)]

        # mark used (we won't revisit them)
        for i_pool in take_idx:
            used_local.add(rem_local[int(i_pool)])
        selected_globals = [rem_global[i] for i in take_idx]

        # staging (no labels): require consistency across rounds before promotion
        to_finetune = []
        if use_staging:
            for g in selected_globals:
                staged[g] = staged.get(g, 0) + 1
                if staged[g] >= staging_patience:
                    to_finetune.append(g)
                    accepted.append(g)
                    del staged[g]
        else:
            to_finetune = selected_globals
            accepted.extend(selected_globals)

        # short fine-tune on presumed normals (oracle-free)
        if len(to_finetune) > 0:
            round_loader = build_loader_from_indices(dataset, to_finetune, batch=32, shuffle=True)
            train_adapter_compact(backbone, adapter, round_loader, device, epochs=1, lr=3e-5, proto_k=128)
            swag.collect(copy.deepcopy(adapter).eval())

        # save a round checkpoint
        torch.save({"state_dict": adapter.state_dict()}, os.path.join(ckpt_dir, f"round{r}_last.pth"))

    print("[AL] oracle-free finished.")
    return adapter, mem_normal

# --------------------------- MAIN ---------------------------
if __name__ == "__main__":
    set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ====== EDIT THESE PATHS (absolute paths recommended) ======
    normal_dir_train  = "chest_xray/train/NORMAL"
    anomaly_dir_train = "chest_xray/train/PNEUMONIA"
    normal_dir_test   = "chest_xray/test/NORMAL"
    anomaly_dir_test  = "chest_xray/test/PNEUMONIA"

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

    # Backbone + adapter shape discovery
    backbone = PCBackbone(model_name="resnet50", out_indices=(2,3), device=device)
    with torch.no_grad():
        x0, _ = dataset[0]
        x0 = x0.unsqueeze(0).to(device)
        feats0 = backbone(x0)
        in_chs = [f.size(1) for f in feats0]

    # ----- FAIR BASELINE on a separate adapter (seed-only warm-up + seed memory) -----
    print("\n=== Baseline (seed-only, warmed adapter, seed memory) ===")
    adapter_base = PCAdapter(in_chs=in_chs, out_dim=256).to(device)

    seed_loader = build_loader_from_indices(dataset, idx_seed, batch=batch, shuffle=True)
    train_adapter_compact(backbone, adapter_base, seed_loader, device, epochs=3, lr=1e-4, proto_k=128)

    seed_loader_eval = build_loader_from_indices(dataset, idx_seed, batch=batch, shuffle=False)
    V_seed = extract_patch_vectors(backbone, adapter_base, seed_loader_eval, device,
                                   target_hw=(16,16), per_image_cap=256)
    memory_seed_baseline = MemoryBank(coreset_ratio=0.3, method="auto", device=device)
    memory_seed_baseline.build(V_seed)

    s_base, heat_base, imgs_base = score_images_patchcore(
        backbone, adapter_base, memory_seed_baseline, test_loader, device,
        top_q=0.05, k=3, out_size=(256,256), collect_images=True
    )

    labels_base = []
    for _, y in test_loader: labels_base.extend(y.tolist())
    labels_base = np.asarray(labels_base, dtype=np.int32)
    if not DATASET_LABEL_ONE_IS_ANOMALY: labels_base = 1 - labels_base

    outdir_base = "runs/patchcore_al_best/baseline_adapter"
    eval_base = evaluate_and_plot(
        labels_base, s_base.numpy(), prefix="baseline",
        outdir=outdir_base, target_precision=TARGET_PRECISION
    )

    os.makedirs(os.path.join(outdir_base, "baseline_vis"), exist_ok=True)
    if HAVE_PLT:
        try:
            idx_vis = 0
            img_v = imgs_base[idx_vis].permute(1,2,0).numpy()
            hm_v  = heat_base[idx_vis].numpy()
            h = (hm_v - hm_v.min())/(hm_v.max()-hm_v.min()+1e-8)
            cmap = plt.get_cmap('jet')
            overlay = (1-0.5)*np.repeat(img_v[..., :1], 3, axis=2) + 0.5*cmap(h)[..., :3]
            overlay = np.clip(overlay, 0.0, 1.0)  # <-- fix for saving
            vis_path = os.path.join(outdir_base, "baseline_vis", "example.png")
            plt.imsave(vis_path, overlay)
            print("Saved:", vis_path)
        except Exception as e:
            print("Baseline vis failed:", e)

    # ----- ACTIVE LEARNING (improved) -----
    adapter_al = PCAdapter(in_chs=in_chs, out_dim=256).to(device)
    # small fixed subset of pool for the unsupervised metric (stabilizes score)
    metric_pool_indices = pool_indices[:1500] if len(pool_indices) > 0 else []

    adapter_trained, memory_normal = active_learning_patchcore_swag_fixed(
        backbone, adapter_al, dataset,
        seed_indices_normals=idx_seed,
        pool_indices=pool_indices,
        device=device,
        rounds=5,
        budget=50,
        mem_coreset=0.3,
        swag_noise=0.02,          # ↑ to make variance non-trivial
        swag_K=4,
        top_q=0.03,               # ↓ a bit to encourage recall
        k=3,
        save_dir="runs/patchcore_al_best",
        strict_normal_only=False,
        resume_each_round_from="best_so_far",
        rank_mode="boundary",
        metric_pool_indices=metric_pool_indices,
        pool_sample_for_metric=512
    )
    
    # ----- ACTIVE LEARNING (oracle-free) -----
    # adapter_al = PCAdapter(in_chs=in_chs, out_dim=256).to(device)

    # adapter_trained, memory_normal = active_learning_oracle_free(
    #     backbone, adapter_al, dataset,
    #     seed_indices_normals=idx_seed,
    #     pool_indices=pool_indices,      # pool can be unlabeled
    #     device=device,
    #     rounds=5,
    #     budget=50,
    #     mem_coreset=0.3,
    #     swag_noise=0.02, swag_K=8,      # stronger SWAG to avoid variance collapse
    #     top_q=0.03, k=3,
    #     save_dir="runs/patchcore_oraclefree",
    #     rank_mode="safe", ws=0.7, wu=0.3,
    #     use_staging=True, staging_patience=2
    # )


    # -------- Post-AL evaluation (with grown memory) --------
    print("\n=== Post-AL evaluation (after AL) ===")
    s_post, heat_post, imgs_post = score_images_patchcore(
        backbone, adapter_trained, memory_normal, test_loader, device,
        top_q=0.05, k=3, out_size=(256,256), collect_images=True
    )

    labels_post = []
    for _, y in test_loader: labels_post.extend(y.tolist())
    labels_post = np.asarray(labels_post, dtype=np.int32)
    if not DATASET_LABEL_ONE_IS_ANOMALY: labels_post = 1 - labels_post

    eval_post = evaluate_and_plot(
        labels_post, s_post.numpy(), prefix="postAL",
        outdir="runs/patchcore_al_best/postAL", target_precision=TARGET_PRECISION
    )

    # -------- Summary --------
    def fmt(v):
        try: return f"{v:.4f}"
        except: return "nan"
    print("\n=== Summary ===")
    print("Baseline:  AUC:", fmt(eval_base.get("auc")), " AP:", fmt(eval_base.get("ap")), " F1@Youden:", fmt(eval_base.get("f1_y")))
    print("Post-AL :  AUC:", fmt(eval_post.get("auc")), " AP:", fmt(eval_post.get("ap")), " F1@Youden:", fmt(eval_post.get("f1_y")))
