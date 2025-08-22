# run_patchcore.py
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
        roc_auc_score, roc_curve,
        accuracy_score, precision_score, recall_score, f1_score,
        precision_recall_curve,
    )
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False
# --------------------------- main ---------------------------
# import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset

# --------------------------- utils ---------------------------

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def build_loader_from_indices(dataset, indices, batch=32, shuffle=True, num_workers=2, pin_memory=True):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

# --------------------------- data ---------------------------

class AnomalyDataset(Dataset):
    """
    Loads images from two folders:
      - normal_dir (label 0)
      - anomaly_dir (label 1)
    Returns tensors in [0,1], shape [C,H,W], C=1 (grayscale) or 3 (rgb).
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        if self.color == "rgb":
            img = img.convert("RGB")
        else:
            img = img.convert("L")
        img = img.resize((self.size, self.size), Image.BILINEAR)

        arr = np.asarray(img).astype("float32") / 255.0
        if arr.ndim == 2:         # grayscale → H,W → H,W,1
            arr = arr[:, :, None]
        if arr.shape[2] == 4:     # drop alpha
            arr = arr[:, :, :3]
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

# --------------------------- PatchCore stack ---------------------------

def _l2n(t):  # L2 norm on channel dim
    eps=1e-6
    n = torch.norm(t, p=2, dim=1, keepdim=True).clamp_min(eps)
    return t / n

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
        # x expected in [0,1]
        if self.use_timm:
            # timm backbones expect 3ch; repeat if grayscale
            if x01_1chw.size(1) == 1:
                x = x01_1chw.repeat(1,3,1,1)
            else:
                x = x01_1chw
            feats = self.model(x)  # list of maps
            return feats
        else:
            x = x01_1chw
            f1 = F.relu(self.c1(x)); f2 = F.relu(self.c2(f1))
            f3 = F.relu(self.c3(f2)); f4 = F.relu(self.c4(f3))
            feats = [f2, f3, f4]
            return [feats[i] for i in self.out_indices]

class PCAdapter(nn.Module):
    """
    Tiny learnable head per selected scale. We'll apply SWAG to this.
    """
    def __init__(self, in_chs: list, out_dim=256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)
        ) for c in in_chs])

    def forward(self, feats):
        return [p(f) for p, f in zip(self.proj, feats)]

class SWAGAdapter:
    def __init__(self, adapter: nn.Module, noise_scale=0.01, max_snaps=30, device="cuda"):
        self.proto = adapter
        self.noise = noise_scale
        self.max_snaps = max_snaps
        self.snaps = []
        self.device = device

    def collect(self, adapter: nn.Module):
        if len(self.snaps) >= self.max_snaps: self.snaps.pop(0)
        self.snaps.append({k: v.detach().cpu().clone() for k,v in adapter.state_dict().items()})

    def _avg(self):
        assert len(self.snaps)>0, "No SWAG snapshots collected."
        keys = self.snaps[0].keys()
        return {k: torch.stack([s[k] for s in self.snaps], 0).mean(0) for k in keys}

    @torch.no_grad()
    def sample_adapters(self, K=8):
        avg = self._avg()
        outs=[]
        for _ in range(K):
            Ad = copy.deepcopy(self.proto).to(self.device)
            noisy = {k: v + torch.randn_like(v)*self.noise for k,v in avg.items()}
            Ad.load_state_dict(noisy, strict=False); Ad.eval()
            outs.append(Ad)
        return outs

def coreset_greedy(X: torch.Tensor, ratio=0.2, seed=42):
    # simple farthest-first traversal; runs on whatever device X is on
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
        self.method = method  # "auto" | "random" | "greedy"
        self.large_N = large_N
        self.device = torch.device(device)

    def build(self, patch_vecs: torch.Tensor):
        N = patch_vecs.size(0)
        X = _l2n(patch_vecs.T).T.contiguous()
        if self.method == "random" or (self.method=="auto" and N > self.large_N):
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
    def knn_dist(self, Q: torch.Tensor, k=3, chunk=8192):
        mem = self.mem
        outs=[]
        for i in range(0, Q.size(0), chunk):
            q = Q[i:i+chunk].to(mem.device, non_blocking=True)
            D = torch.cdist(q, mem, p=2)
            outs.append(D.topk(k, largest=False, dim=1).values.mean(1).cpu())
        return torch.cat(outs, 0)

# --------------------------- Feature extraction & scoring ---------------------------

def extract_patch_vectors(backbone, adapter, loader, device, target_hw=(16,16), per_image_cap=256):
    """
    Downsample each scale to target_hw to keep memory small; return all patch vectors (on CPU).
    """
    backbone.eval(); adapter.eval()
    vecs=[]
    with torch.no_grad():
        for x,_ in loader:
            x = x.to(device, non_blocking=True)
            feats = adapter(backbone(x))  # list [B,D,Hs,Ws]
            ups  = [F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)                   # [B, Dsum, H, W] small grid
            P    = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))  # [B*H*W, Dsum]
            if per_image_cap is not None:
                B, H, W = x.size(0), target_hw[0], target_hw[1]
                patches_per_img = H*W
                if patches_per_img > per_image_cap:
                    # uniform slice per image (already small, so keep simple)
                    P = P.view(B, patches_per_img, -1)[:, :per_image_cap, :].reshape(-1, P.size(1))
            vecs.append(P.cpu())
    return torch.cat(vecs, 0)

def score_images_patchcore(backbone, adapter, mem: MemoryBank, loader, device, top_q=0.10, k=3):
    backbone.eval(); adapter.eval()
    scores=[]; heatmaps=[]
    with torch.no_grad():
        for x,_ in loader:
            x = x.to(device)
            feats = adapter(backbone(x))  # list [B,D,H,W]
            Hmax = max(f.size(2) for f in feats); Wmax = max(f.size(3) for f in feats)
            ups = [F.interpolate(f, size=(Hmax,Wmax), mode="bilinear", align_corners=False) for f in feats]
            Fcat = torch.cat(ups, dim=1)
            P = _l2n(Fcat).permute(0,2,3,1).reshape(-1, Fcat.size(1))
            d = mem.knn_dist(P, k=k).view(x.size(0), Hmax, Wmax)
            flat = d.view(x.size(0), -1)
            q = max(1, int(top_q*flat.size(1)))
            s = torch.topk(flat, k=q, dim=1).values.mean(1)
            scores.append(s.cpu()); heatmaps.append(d.cpu())
    return torch.cat(scores,0), torch.cat(heatmaps,0)

@torch.no_grad()
def swag_uncertainty(backbone, swag_adp: SWAGAdapter, mem: MemoryBank, loader, device, K=8, top_q=0.10, k=3):
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

# --------------------------- Active learning loop ---------------------------

def active_learning_patchcore_swag(
    backbone, adapter, dataset, seed_indices_normals, pool_indices, device,
    rounds=3, budget=1000, mem_coreset=0.2, swag_noise=0.01, swag_K=8,
    p_gate=0.95, top_q=0.10, k=3, save_dir="runs/patchcore_al"
):
    os.makedirs(save_dir, exist_ok=True)
    seed_loader = build_loader_from_indices(dataset, seed_indices_normals, batch=32, shuffle=True)

    # 1) adapter train + SWAG seed
    train_adapter_compact(backbone, adapter, seed_loader, device, epochs=3, lr=1e-4, proto_k=128)
    swag = SWAGAdapter(adapter, noise_scale=swag_noise, max_snaps=30, device=device)
    swag.collect(copy.deepcopy(adapter).eval()); swag.collect(copy.deepcopy(adapter).eval())

    # 2) memory on seeds (downsampled vectors for speed)
    seed_loader_eval = build_loader_from_indices(dataset, seed_indices_normals, batch=32, shuffle=False)
    V_seed = extract_patch_vectors(backbone, adapter, seed_loader_eval, device, target_hw=(16,16), per_image_cap=256)
    mem = MemoryBank(coreset_ratio=mem_coreset, method="auto", device=device)
    mem.build(V_seed)

    # 3) calibrate gates on seeds
    s_seed= score_images_patchcore(backbone, adapter, mem, seed_loader_eval, device, top_q=top_q, k=k)
    u_seed    = swag_uncertainty(backbone, swag, mem, seed_loader_eval, device, K=swag_K, top_q=top_q, k=k)
    tau_s = np.percentile(s_seed.numpy(), p_gate*100)
    tau_u = np.percentile(u_seed.numpy(), p_gate*100)
    print(f"[Calib] score<= {tau_s:.4f} (p{int(p_gate*100)}), var<= {tau_u:.4f} (p{int(p_gate*100)})")

    used = set()
    for r in range(1, rounds+1):
        print(f"\n=== Round {r} ===")
        rem_local = [i for i in range(len(pool_indices)) if i not in used]
        if not rem_local: print("Pool exhausted."); break
        rem_global = [pool_indices[i] for i in rem_local]
        rem_loader = build_loader_from_indices(dataset, rem_global, batch=32, shuffle=False)

        s_pool, _ = score_images_patchcore(backbone, adapter, mem, rem_loader, device, top_q=top_q, k=k)
        u_pool    = swag_uncertainty(backbone, swag, mem, rem_loader, device, K=swag_K, top_q=top_q, k=k)

        pass_s = (s_pool <= tau_s); pass_u = (u_pool <= tau_u)
        accept_idx  = torch.nonzero(pass_s & pass_u, as_tuple=False).squeeze(1).tolist()

        # borderline: fails exactly one gate within +10%
        margin=0.10
        border=[]
        for i in range(len(rem_global)):
            cs = (not bool(pass_s[i])) and (s_pool[i].item()/(tau_s+1e-8)-1.0) <= margin
            cu = (not bool(pass_u[i])) and (u_pool[i].item()/(tau_u+1e-8)-1.0) <= margin
            if (cs + cu) == 1: border.append(i)

        print(f"[Round {r}] Accept {len(accept_idx)} | Borderline {len(border)} | Pool {len(rem_global)}")

        cand = accept_idx + border
        if not cand:
            print("[Round] No candidates this round; relaxing var gate to pool p95.")
            tau_u = np.percentile(u_pool.numpy(), 95)
            continue

        fused = 0.7*(s_pool.numpy()[cand]/(tau_s+1e-8)) + 0.3*(u_pool.numpy()[cand]/(tau_u+1e-8))
        order = np.argsort(fused)
        take  = [cand[i] for i in order[:min(budget, len(cand))]]

        for li in take: used.add(rem_local[li])
        new_global = [rem_global[i] for i in take]
        print(f"[Round {r}] Selected {len(new_global)} pseudo-normals.")

        # brief adapter tune + new memory
        round_loader = build_loader_from_indices(dataset, new_global, batch=32, shuffle=True)
        train_adapter_compact(backbone, adapter, round_loader, device, epochs=1, lr=5e-5, proto_k=128)
        swag.collect(copy.deepcopy(adapter).eval())

        merged = seed_indices_normals + new_global
        merged_loader = build_loader_from_indices(dataset, merged, batch=32, shuffle=False)
        V_new = extract_patch_vectors(backbone, adapter, merged_loader, device, target_hw=(16,16), per_image_cap=256)
        mem.build(V_new)

    return adapter, mem



def active_learning_patchcore_swag(
    backbone, adapter, dataset, seed_indices_normals, pool_indices, device,
    rounds=3, budget=1000, mem_coreset=0.2, swag_noise=0.01, swag_K=8,
    p_gate=0.95, top_q=0.10, k=3, save_dir="runs/patchcore_al"
):
    os.makedirs(save_dir, exist_ok=True)
    seed_loader = build_loader_from_indices(dataset, seed_indices_normals, batch=32, shuffle=True)

    # 1) adapter train + SWAG seed
    train_adapter_compact(backbone, adapter, seed_loader, device, epochs=3, lr=1e-4, proto_k=128)
    swag = SWAGAdapter(adapter, noise_scale=swag_noise, max_snaps=30, device=device)
    swag.collect(copy.deepcopy(adapter).eval()); swag.collect(copy.deepcopy(adapter).eval())

    # 2) memory on seeds (downsampled vectors for speed)
    seed_loader_eval = build_loader_from_indices(dataset, seed_indices_normals, batch=32, shuffle=False)
    V_seed = extract_patch_vectors(backbone, adapter, seed_loader_eval, device, target_hw=(16,16), per_image_cap=256)
    mem = MemoryBank(coreset_ratio=mem_coreset, method="auto", device=device)
    mem.build(V_seed)

    # 3) calibrate gates on seeds (robust to 2- or 3-return scorer)
    ret_seed  = score_images_patchcore(backbone, adapter, mem, seed_loader_eval, device, top_q=top_q, k=k)
    s_seed_np = _scores_to_numpy(ret_seed)  # 1D numpy array
    u_seed_np = swag_uncertainty(backbone, swag, mem, seed_loader_eval, device, K=swag_K, top_q=top_q, k=k)\
                    .detach().cpu().numpy()

    tau_s = np.percentile(s_seed_np, p_gate*100)
    tau_u = np.percentile(u_seed_np, p_gate*100)
    print(f"[Calib] score<= {tau_s:.4f} (p{int(p_gate*100)}), var<= {tau_u:.4f} (p{int(p_gate*100)})")

    used = set()
    for r in range(1, rounds+1):
        print(f"\n=== Round {r} ===")
        rem_local = [i for i in range(len(pool_indices)) if i not in used]
        if not rem_local:
            print("Pool exhausted.")
            break

        rem_global = [pool_indices[i] for i in rem_local]
        rem_loader = build_loader_from_indices(dataset, rem_global, batch=32, shuffle=False)

        # --- robust scoring (handles 2/3 returns)
        ret_pool  = score_images_patchcore(backbone, adapter, mem, rem_loader, device, top_q=top_q, k=k)
        s_pool_np = _scores_to_numpy(ret_pool)  # numpy [N]
        u_pool_np = swag_uncertainty(backbone, swag, mem, rem_loader, device, K=swag_K, top_q=top_q, k=k)\
                        .detach().cpu().numpy()

        # Gates in numpy
        pass_s = (s_pool_np <= tau_s)
        pass_u = (u_pool_np <= tau_u)
        accept_idx = np.nonzero(pass_s & pass_u)[0].tolist()

        # borderline: fails exactly one gate within +10%
        margin = 0.10
        border = []
        for i in range(len(rem_global)):
            cs = (not pass_s[i]) and ((s_pool_np[i]/(tau_s+1e-8) - 1.0) <= margin)
            cu = (not pass_u[i]) and ((u_pool_np[i]/(tau_u+1e-8) - 1.0) <= margin)
            if (cs + cu) == 1:
                border.append(i)

        print(f"[Round {r}] Accept {len(accept_idx)} | Borderline {len(border)} | Pool {len(rem_global)}")

        cand = accept_idx + border
        if not cand:
            print("[Round] No candidates this round; relaxing var gate to pool p95.")
            tau_u = np.percentile(u_pool_np, 95)
            continue

        fused = 0.7*(s_pool_np[cand]/(tau_s+1e-8)) + 0.3*(u_pool_np[cand]/(tau_u+1e-8))
        order = np.argsort(fused)
        take  = [cand[i] for i in order[:min(budget, len(cand))]]

        for li in take:
            used.add(rem_local[li])
        new_global = [rem_global[i] for i in take]
        print(f"[Round {r}] Selected {len(new_global)} pseudo-normals.")

        # brief adapter tune + new memory
        round_loader = build_loader_from_indices(dataset, new_global, batch=32, shuffle=True)
        train_adapter_compact(backbone, adapter, round_loader, device, epochs=20, lr=5e-5, proto_k=128)
        swag.collect(copy.deepcopy(adapter).eval())

        merged = seed_indices_normals + new_global
        merged_loader = build_loader_from_indices(dataset, merged, batch=32, shuffle=False)
        V_new = extract_patch_vectors(backbone, adapter, merged_loader, device, target_hw=(16,16), per_image_cap=256)
        mem.build(V_new)

    return adapter, mem

class SWAGAdapter:
    def __init__(self, adapter: nn.Module, noise_scale=0.01, max_snaps=30, device="cuda"):
        self.proto = adapter
        self.noise = noise_scale
        self.max_snaps = max_snaps
        self.snaps = []
        self.device = device

    def collect(self, adapter: nn.Module):
        # store a CPU copy of the full state_dict (weights + buffers)
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
                # stack and mean in float32
                stk = torch.stack([t.to(torch.float32) for t in tensors], 0)
                avg[k] = stk.mean(0)
            else:
                # non-float buffers (e.g., num_batches_tracked) – just take the first
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
                if v.is_floating_point():
                    noisy[k] = v + torch.randn_like(v) * self.noise
                else:
                    noisy[k] = v  # keep integer/bool buffers unchanged
            Ad.load_state_dict(noisy, strict=False)
            Ad.eval()
            outs.append(Ad)
        return outs


# ---------------------------
# Visualization utilities
# ---------------------------
def _scores_to_numpy(ret):
    """
    Accepts output from score_images_patchcore which might be:
      - Tensor of scores
      - (scores, heatmaps) tuple
      - (scores, heatmaps, imgs) tuple
      - list/ndarray
    Returns a 1D numpy array of scores.
    """
    # If it's a (scores, ...) or [scores, ...], take the first element
    if isinstance(ret, (tuple, list)):
        ret = ret[0]
    # Torch tensor -> numpy
    if hasattr(ret, "detach"):
        return ret.detach().cpu().numpy()
    # Already numpy/list-like
    return np.asarray(ret)

def plot_with_heatmap(img_tensor, heatmap_tensor, idx=0, size=(256,256), alpha=0.5, save_path=None, dpi=100):
    """
    img_tensor: [N, C, H, W], values ideally in [0,1]
    heatmap_tensor: [N, Hh, Wh] (already 256x256 in our scorer; will still re-check/resize)
    idx: sample index
    """
    import numpy as np

    # --- pick sample
    img = img_tensor[idx].detach().cpu()               # [C,H,W]
    hmap = heatmap_tensor[idx].detach().cpu()          # [Hh,Wh]

    # --- ensure size=(H,W)
    if tuple(hmap.shape[-2:]) != tuple(size):
        hmap = F.interpolate(hmap.unsqueeze(0).unsqueeze(0), size, mode="bilinear",
                             align_corners=False)[0,0]
    if tuple(img.shape[-2:]) != tuple(size):
        img = F.interpolate(img.unsqueeze(0), size, mode="bilinear",
                            align_corners=False)[0]

    # normalize heatmap to [0,1]
    hmin, hmax = float(hmap.min()), float(hmap.max())
    if hmax - hmin < 1e-12:
        hmap_norm = torch.zeros_like(hmap)
    else:
        hmap_norm = (hmap - hmin) / (hmax - hmin)

    # image to (H,W,C) and ensure RGB
    img_np = img.permute(1,2,0).numpy()                # [H,W,C]
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    img_np = np.clip(img_np, 0.0, 1.0)
    hmap_np = hmap_norm.numpy()

    # figure sizing (exact pixels if saving)
    if save_path:
        inches_w = (2*size[1]) / dpi
        inches_h = (size[0]) / dpi
        fig = plt.figure(figsize=(inches_w, inches_h), dpi=dpi)
    else:
        fig = plt.figure(figsize=(6,3))

    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(img_np, interpolation="nearest")
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(img_np, interpolation="nearest")
    ax2.imshow(hmap_np, cmap="jet", alpha=alpha, interpolation="bilinear")
    ax2.set_title("Heatmap Overlay")
    ax2.axis("off")

    plt.tight_layout(pad=0.1)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
    else:
        plt.show()


def save_side_by_side_cv2(img_tensor, heatmap_tensor, idx=0, size=(256,256), out_path="vis.png", alpha=0.5):
    """
    Writes a 256x512 PNG: [Image | HeatmapOverlay]
    """
    if not HAVE_CV2:
        raise RuntimeError("OpenCV (cv2) not available. Install opencv-python to use this saver.")

    img = img_tensor[idx].detach().cpu()         # [C,H,W]
    hmap = heatmap_tensor[idx].detach().cpu()    # [Hh,Wh]

    # resize to size
    if tuple(img.shape[-2:]) != tuple(size):
        img = F.interpolate(img.unsqueeze(0), size, mode="bilinear",
                            align_corners=False)[0]
    if tuple(hmap.shape[-2:]) != tuple(size):
        hmap = F.interpolate(hmap.unsqueeze(0).unsqueeze(0), size, mode="bilinear",
                             align_corners=False)[0,0]

    # image -> RGB uint8
    img_np = img.permute(1,2,0).numpy()
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    img_np = np.clip(img_np, 0.0, 1.0)
    img_255 = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_255, cv2.COLOR_RGB2BGR)

    # heatmap -> color
    h = hmap.numpy()
    h = (h - h.min()) / (h.max() - h.min() + 1e-12)
    h_255 = (h * 255).astype(np.uint8)
    h_cmap = cv2.applyColorMap(h_255, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_bgr, 1.0, h_cmap, alpha, 0.0)
    side_by_side = np.concatenate([img_bgr, overlay], axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, side_by_side)


#---------------------------
#PatchCore scorer (modified)
#---------------------------

def score_images_patchcore(backbone, adapter, mem, loader, device, top_q=0.10, k=3,
                           out_size=(256,256), collect_images=True):
    """
    Returns:
      scores:   [N] tensor of top-q mean distances per image
      heatmaps: [N, out_size[0], out_size[1]] heatmaps (upsampled to out_size)
      imgs_opt: (optional) if collect_images=True, returns [N, C, out_size[0], out_size[1]]
                with input images resized to out_size for easy visualization

    Assumes:
      - adapter(backbone(x)) -> list of [B, D_l, H_l, W_l] features
      - mem.knn_dist(P, k)   -> distances for each patch vector P, shape [num_patches]
      - _l2n(tensor)         -> your L2-normalize helper (not defined here)
    """
    backbone.eval()
    adapter.eval()

    all_scores = []
    all_heatmaps = []
    all_imgs = [] if collect_images else None

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)  # [B,C,H,W]
            feats_list = adapter(backbone(x))  # list of [B,D,H_l,W_l]
            Hmax = max(f.size(2) for f in feats_list)
            Wmax = max(f.size(3) for f in feats_list)

            # upsample features to common spatial size
            ups = [F.interpolate(f, size=(Hmax, Wmax), mode="bilinear", align_corners=False) for f in feats_list]
            Fcat = torch.cat(ups, dim=1)  # [B, sumD, Hmax, Wmax]

            # normalize, flatten patches
            P = _l2n(Fcat).permute(0, 2, 3, 1).reshape(-1, Fcat.size(1))  # [B*Hmax*Wmax, sumD]

            # distances for each patch, then reshape to [B, Hmax, Wmax]
            d = mem.knn_dist(P, k=k).view(x.size(0), Hmax, Wmax)  # [B,Hmax,Wmax]

            # anomaly score = mean of top-q patch distances
            flat = d.view(x.size(0), -1)
            q = max(1, int(top_q * flat.size(1)))
            s = torch.topk(flat, k=q, dim=1).values.mean(1)  # [B]

            # upsample heatmaps to out_size (e.g., 256x256)
            d_up = F.interpolate(d.unsqueeze(1), size=out_size, mode="bilinear",
                                 align_corners=False).squeeze(1)  # [B,H,W]

            all_scores.append(s.detach().cpu())
            all_heatmaps.append(d_up.detach().cpu())

            if collect_images:
                # resize input images to out_size (for consistent side-by-side plots)
                x_small = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False).detach().cpu()
                all_imgs.append(x_small)

    scores = torch.cat(all_scores, dim=0)               # [N]
    heatmaps = torch.cat(all_heatmaps, dim=0)           # [N,H,W]
    imgs_opt = torch.cat(all_imgs, dim=0) if collect_images else None

    if collect_images:
        return scores, heatmaps, imgs_opt
    else:
        return scores, heatmaps


# # ---------------------------
# # Main (your original flow, with small changes)
# # ---------------------------

# if __name__ == "__main__":
#     # Your environment should provide these:
#     # - set_seed
#     # - AnomalyDataset, split_normal_30
#     # - PCBackbone, PCAdapter
#     # - active_learning_patchcore_swag
#     # - MemoryBank, _l2n, DataLoader
#     # - HAVE_SKLEARN / sklearn metrics if available
#     # - HAVE_PLT (matplotlib)
#     # (They are not redefined here.)

#     set_seed(123)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)

#     # ==== EDIT THESE PATHS ====
#     normal_dir_train  = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/train/NORMAL"
#     anomaly_dir_train = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/train/PNEUMONIA"
#     normal_dir_test   = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/test/NORMAL"
#     anomaly_dir_test  = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/test/PNEUMONIA"
#     # ==========================

#     size = 256
#     batch = 32

#     dataset = AnomalyDataset(normal_dir_train, anomaly_dir_train, size=size, color="L")
#     idx_seed, idx_rest_normals = split_normal_30(dataset, seed=42)
#     idx_anoms = [i for i in range(len(dataset)) if dataset.labels[i] == 1]
#     pool_indices = idx_rest_normals + idx_anoms

#     # Backbone + adapter
#     backbone = PCBackbone(model_name="resnet50", out_indices=(2,3), device=device)
#     with torch.no_grad():
#         x0, _ = dataset[0]; x0 = x0.unsqueeze(0).to(device)
#         feats0 = backbone(x0)
#         in_chs = [f.size(1) for f in feats0]
#     adapter = PCAdapter(in_chs=in_chs, out_dim=256).to(device)

#     # Active learning
#     adapter_trained, memory = active_learning_patchcore_swag(
#         backbone, adapter, dataset,
#         seed_indices_normals=idx_seed,
#         pool_indices=pool_indices,
#         device=device,
#         rounds=5, budget=2000, mem_coreset=0.3,
#         swag_noise=0.01, swag_K=8, p_gate=0.90, top_q=0.10, k=3,
#         save_dir="runs/patchcore_al"
#     )

#     # -------- Evaluation on held-out test set --------
#     assert os.path.isdir(normal_dir_test) and os.path.isdir(anomaly_dir_test), "Test folders not found."

#     test_dataset = AnomalyDataset(normal_dir_test, anomaly_dir_test, size=size, color="L")
#     test_loader  = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

#     # Modified: collect_images=True to get 256x256 inputs aligned with heatmaps
#     s_test, heat, imgs256 = score_images_patchcore(
#         backbone, adapter_trained, memory, test_loader, device,
#         top_q=0.10, k=3, out_size=(256,256), collect_images=True
#     )

#     # Gather labels
#     labels_all = []
#     for _, y in test_loader:
#         labels_all.extend(y.tolist())
#     scores = s_test.numpy()

#     # ---- Metrics: AUC, ACC, Precision, Recall, F1 ----
#     if HAVE_SKLEARN and len(set(labels_all)) > 1:
#         from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
#         from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#         auc = roc_auc_score(labels_all, scores)
#         fpr, tpr, thr = roc_curve(labels_all, scores)
#         # Youden's J to pick a threshold
#         J = tpr - fpr
#         jbest = int(np.argmax(J))
#         thr_best = thr[jbest] if jbest < len(thr) else 0.5
#         preds = (scores >= thr_best).astype(np.int32)

#         acc  = accuracy_score(labels_all, preds)
#         prec = precision_score(labels_all, preds, zero_division=0)
#         rec  = recall_score(labels_all, preds, zero_division=0)
#         f1   = f1_score(labels_all, preds, zero_division=0)

#         print("\n=== Evaluation (held-out test set) ===")
#         print(f"AUC-ROC : {auc:.4f}")
#         print(f"Thr*    : {thr_best:.6f} (Youden)")
#         print(f"ACC     : {acc:.4f}")
#         print(f"Precision: {prec:.4f}")
#         print(f"Recall   : {rec:.4f}")
#         print(f"F1       : {f1:.4f}")

#         # Optional curves
#         HAVE_PLT = True
#         os.makedirs("runs/patchcore_al", exist_ok=True)
#         plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
#         plt.plot([0,1],[0,1],"k--"); plt.legend(); plt.title("ROC"); plt.xlabel("FPR"); plt.ylabel("TPR")
#         plt.savefig("runs/patchcore_al/roc_patchcore.png", dpi=120); plt.close()

#         pr, rc, _ = precision_recall_curve(labels_all, scores)
#         plt.figure(); plt.plot(rc, pr); plt.title("Precision-Recall"); plt.xlabel("Recall"); plt.ylabel("Precision")
#         plt.savefig("runs/patchcore_al/pr_patchcore.png", dpi=120); plt.close()
#     else:
#         print("(Metrics skipped: scikit-learn not available or labels not mixed.)")

#     # ---------------------------
#     # Visualizations (256x256)
#     # ---------------------------
#     os.makedirs("runs/patchcore_al/test_vis", exist_ok=True)

#     # Show one example inline (matplotlib)
#     try:
#         plot_with_heatmap(imgs256, heat, idx=0, size=(256,256), alpha=0.5,
#                           save_path="runs/patchcore_al/test_vis/example_matplotlib.png", dpi=128)
#         print("Saved:", "runs/patchcore_al/test_vis/example_matplotlib.png")
#     except Exception as e:
#         print("Matplotlib plot failed:", e)

#     # Save exact-pixel side-by-side via OpenCV
#     if HAVE_CV2:
#         try:
#             save_side_by_side_cv2(imgs256, heat, idx=0, size=(256,256),
#                                   out_path="runs/patchcore_al/test_vis/example_cv2.png", alpha=0.5)
#             print("Saved:", "runs/patchcore_al/test_vis/example_cv2.png")
#         except Exception as e:
#             print("cv2 save failed:", e)

#     # (Optional) save a few raw colorized heatmaps alone (already 256x256)
#     if HAVE_CV2:
#         for i in range(min(10, heat.size(0))):
#             h = heat[i]
#             h = (h - h.min())/(h.max()-h.min()+1e-8)
#             hm = (h.numpy()*255).astype(np.uint8)
#             hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
#             cv2.imwrite(f"runs/patchcore_al/test_vis/heat_{i}.png", hm)
# -*- coding: utf-8 -*-
"""
Full end-to-end script:
- Baseline (pre-AL) evaluation + plots
- Active learning training
- Post-AL evaluation + plots
- Heatmap visualizations

Assumes your environment provides:
  - set_seed
  - AnomalyDataset, split_normal_30
  - PCBackbone, PCAdapter
  - active_learning_patchcore_swag
  - MemoryBank with init_from_feats(...)
  - _l2n (L2-normalize helper), DataLoader
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Optional libs (guarded)
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

# Try to detect sklearn
try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


# ---------------------------------------------------------------------
# Provided scoring function (slightly tidied – same logic as your version)
# ---------------------------------------------------------------------
def score_images_patchcore(backbone, adapter, mem, loader, device, top_q=0.10, k=3,
                           out_size=(256, 256), collect_images=True):
    """
    Returns:
      scores:   [N] tensor of top-q mean distances per image (higher => more anomalous)
      heatmaps: [N, out_size[0], out_size[1]]
      imgs_opt: [N, C, out_size[0], out_size[1]] if collect_images=True

    Assumes:
      - adapter(backbone(x)) -> list of [B, D_l, H_l, W_l] features
      - mem.knn_dist(P, k)   -> distances for each patch vector P, shape [num_patches_total]
      - _l2n(tensor)         -> L2-normalize helper
    """
    backbone.eval()
    adapter.eval()

    all_scores = []
    all_heatmaps = []
    all_imgs = [] if collect_images else None

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)  # [B,C,H,W]
            feats_list = adapter(backbone(x))  # list of [B,D,H_l,W_l]
            Hmax = max(f.size(2) for f in feats_list)
            Wmax = max(f.size(3) for f in feats_list)

            # upsample features to common spatial size
            ups = [F.interpolate(f, size=(Hmax, Wmax), mode="bilinear", align_corners=False) for f in feats_list]
            Fcat = torch.cat(ups, dim=1)  # [B, sumD, Hmax, Wmax]

            # normalize, flatten patches
            P = _l2n(Fcat).permute(0, 2, 3, 1).reshape(-1, Fcat.size(1))  # [B*Hmax*Wmax, sumD]

            # distances for each patch, then reshape to [B, Hmax, Wmax]
            d = mem.knn_dist(P, k=k).view(x.size(0), Hmax, Wmax)  # [B,Hmax,Wmax]

            # anomaly score = mean of top-q patch distances
            flat = d.view(x.size(0), -1)
            q = max(1, int(top_q * flat.size(1)))
            s = torch.topk(flat, k=q, dim=1).values.mean(1)  # [B]

            # upsample heatmaps to out_size (e.g., 256x256)
            d_up = F.interpolate(d.unsqueeze(1), size=out_size, mode="bilinear",
                                 align_corners=False).squeeze(1)  # [B,H,W]

            all_scores.append(s.detach().cpu())
            all_heatmaps.append(d_up.detach().cpu())

            if collect_images:
                # resize input images to out_size (for consistent side-by-side plots)
                x_small = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False).detach().cpu()
                all_imgs.append(x_small)

    scores = torch.cat(all_scores, dim=0)               # [N]
    heatmaps = torch.cat(all_heatmaps, dim=0)           # [N,H,W]
    imgs_opt = torch.cat(all_imgs, dim=0) if collect_images else None

    if collect_images:
        return scores, heatmaps, imgs_opt
    else:
        return scores, heatmaps


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------
def plot_with_heatmap(imgs, heats, idx=0, size=(256, 256), alpha=0.5, save_path=None, dpi=128):
    """
    imgs: [N,C,H,W] tensor on CPU, values in [0,1] or [0,255]
    heats: [N,H,W] tensor on CPU (already resized to `size`)
    """
    if not HAVE_PLT:
        print("Matplotlib not available; skipping plot_with_heatmap.")
        return

    img = imgs[idx].numpy()
    heat = heats[idx].numpy()

    # Normalize img to [0,1]
    if img.max() > 1.0:
        img = img / 255.0

    # To HxW or HxWx3
    if img.shape[0] == 1:
        base = np.repeat(img[0:1], 3, axis=0)  # 3xH xW
    elif img.shape[0] == 3:
        base = img
    else:
        # Unexpected channels -> take first and repeat
        base = np.repeat(img[0:1], 3, axis=0)
    base = np.transpose(base, (1, 2, 0))  # HxWx3

    # Normalize heat to [0,1] and apply colormap
    h = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    cmap = plt.get_cmap('jet')
    heat_rgb = cmap(h)[..., :3]  # HxWx3

    overlay = (1 - alpha) * base + alpha * heat_rgb
    overlay = np.clip(overlay, 0, 1)

    fig = plt.figure(figsize=(8, 4), dpi=dpi)
    ax1 = plt.subplot(1, 2, 1); ax1.imshow(base); ax1.set_title("Image"); ax1.axis('off')
    ax2 = plt.subplot(1, 2, 2); ax2.imshow(overlay); ax2.set_title("Heatmap Overlay"); ax2.axis('off')

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print("Saved:", save_path)
    plt.close(fig)


def save_side_by_side_cv2(imgs, heats, idx=0, size=(256, 256), out_path=None, alpha=0.5):
    if not HAVE_CV2:
        print("cv2 not available; skipping save_side_by_side_cv2.")
        return

    img = imgs[idx].numpy()
    heat = heats[idx].numpy()

    # Prepare grayscale base image
    if img.shape[0] == 3:
        base = np.transpose(img, (1, 2, 0))
        if base.max() <= 1.0:
            base = (base * 255).astype(np.uint8)
        base_gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
        base_rgb = base
    else:
        base_gray = img[0]
        if base_gray.max() <= 1.0:
            base_gray = (base_gray * 255).astype(np.uint8)
        base_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)

    # Heatmap colorization
    h = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    hm_u8 = (h * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(base_rgb, 1 - alpha, heat_color, alpha, 0)

    # Side-by-side
    vis = np.concatenate([base_rgb, overlay], axis=1)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print("Saved:", out_path)


# ---------------------------------------------------------------------
# Metrics/plots helper
# ---------------------------------------------------------------------
def evaluate_and_plot(labels_all, scores, prefix="", outdir="runs/patchcore_al", target_precision=0.80):
    if not HAVE_SKLEARN:
        print("(Metrics skipped: scikit-learn not available.)")
        return {
            "auc": np.nan, "ap": np.nan, "thr_youden": np.nan, "thr_opt": np.nan,
            "acc_y": np.nan, "prec_y": np.nan, "rec_y": np.nan, "f1_y": np.nan,
            "acc_o": np.nan, "prec_o": np.nan, "rec_o": np.nan, "f1_o": np.nan
        }

    os.makedirs(outdir, exist_ok=True)

    auc = roc_auc_score(labels_all, scores)
    fpr, tpr, thr = roc_curve(labels_all, scores)
    ap  = average_precision_score(labels_all, scores)

    # Youden threshold (robust)
    J = tpr - fpr
    jbest = int(np.nanargmax(J))
    thr_best = thr[jbest] if np.isfinite(thr[jbest]) else np.nan
    if not np.isfinite(thr_best):
        thr_best = np.quantile(scores, 0.95)
    preds_y = (scores >= thr_best).astype(np.int32)

    acc_y = accuracy_score(labels_all, preds_y)
    prec_y = precision_score(labels_all, preds_y, zero_division=0)
    rec_y  = recall_score(labels_all, preds_y, zero_division=0)
    f1_y   = f1_score(labels_all, preds_y, zero_division=0)
    cm_y   = confusion_matrix(labels_all, preds_y)

    # Operating point by target precision
    pr, rc, thr_pr = precision_recall_curve(labels_all, scores)
    mask = pr[1:] >= target_precision
    if np.any(mask):
        thr_op = thr_pr[mask][-1]
    else:
        thr_op = thr_pr[np.argmax(pr[1:])]
    preds_o = (scores >= thr_op).astype(np.int32)

    acc_o = accuracy_score(labels_all, preds_o)
    prec_o = precision_score(labels_all, preds_o, zero_division=0)
    rec_o  = recall_score(labels_all, preds_o, zero_division=0)
    f1_o   = f1_score(labels_all, preds_o, zero_division=0)
    cm_o   = confusion_matrix(labels_all, preds_o)

    # Print block
    print(f"\n=== {prefix} Evaluation ===")
    print(f"ROC AUC : {auc:.4f}")
    print(f"PR  AUC : {ap:.4f}")
    print(f"Thr* (Youden) : {thr_best:.6f}")
    print(f"  ACC={acc_y:.4f}  Precision={prec_y:.4f}  Recall={rec_y:.4f}  F1={f1_y:.4f}")
    print("  Confusion (Youden) [tn fp; fn tp]:\n", cm_y)

    print(f"\nThr@P>={target_precision:.2f} : {thr_op:.6f}")
    print(f"  ACC={acc_o:.4f}  Precision={prec_o:.4f}  Recall={rec_o:.4f}  F1={f1_o:.4f}")
    print("  Confusion (OpPt) [tn fp; fn tp]:\n", cm_o)

    # Plots
    if HAVE_PLT:
        plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.legend(); plt.title(f"ROC ({prefix})"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.savefig(os.path.join(outdir, f"roc_{prefix}.png"), dpi=120); plt.close()

        plt.figure(); plt.plot(rc, pr, label=f"AP={ap:.4f}")
        plt.title(f"Precision-Recall ({prefix})"); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.savefig(os.path.join(outdir, f"pr_{prefix}.png"), dpi=120); plt.close()

    return {
        "auc": auc, "ap": ap, "thr_youden": thr_best, "thr_opt": thr_op,
        "acc_y": acc_y, "prec_y": prec_y, "rec_y": rec_y, "f1_y": f1_y,
        "acc_o": acc_o, "prec_o": prec_o, "rec_o": rec_o, "f1_o": f1_o
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
# if __name__ == "__main__":
#     # External objects expected from your environment:
#     # set_seed, AnomalyDataset, split_normal_30
#     # PCBackbone, PCAdapter, active_learning_patchcore_swag
#     # MemoryBank, _l2n  (and DataLoader already imported)

#     # =================== CONFIG ===================
#     set_seed(123)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)

#     # Paths (EDIT THESE)
#     normal_dir_train  = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/train/NORMAL"
#     anomaly_dir_train = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/train/PNEUMONIA"
#     normal_dir_test   = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/test/NORMAL"
#     anomaly_dir_test  = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/test/PNEUMONIA"

#     size = 256
#     batch = 32

#     # Label/score polarity controls
#     # If dataset uses 0=NORMAL, 1=ANOMALY (typical), leave True.
#     # If dataset uses 1=NORMAL, 0=ANOMALY, set False.
#     DATASET_LABEL_ONE_IS_ANOMALY = True

#     # Target precision for an "operating-point" threshold (besides Youden)
#     TARGET_PRECISION = 0.80
#     # ==============================================

#     # ---- Train pool construction ----
#     dataset = AnomalyDataset(normal_dir_train, anomaly_dir_train, size=size, color="L")
#     idx_seed, idx_rest_normals = split_normal_30(dataset, seed=42)
#     idx_anoms = [i for i in range(len(dataset)) if dataset.labels[i] == 1]
#     pool_indices = idx_rest_normals + idx_anoms

#     # ---- Backbone + Adapter ----
#     backbone = PCBackbone(model_name="resnet50", out_indices=(2, 3), device=device)
#     with torch.no_grad():
#         x0, _ = dataset[0]
#         x0 = x0.unsqueeze(0).to(device)
#         feats0 = backbone(x0)
#         in_chs = [f.size(1) for f in feats0]
#     adapter = PCAdapter(in_chs=in_chs, out_dim=256).to(device)

#     # ---- TEST SET LOADER (used for both pre-AL and post-AL) ----
#     assert os.path.isdir(normal_dir_test) and os.path.isdir(anomaly_dir_test), "Test folders not found."
#     test_dataset = AnomalyDataset(normal_dir_test, anomaly_dir_test, size=size, color="L")
#     test_loader  = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

#     # -----------------------------------------------------------------
#     # PRE-AL BASELINE: build naive memory bank from TRAIN set features
#     # -----------------------------------------------------------------
#     print("\n=== Baseline evaluation (before Active Learning) ===")
#     train_loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
#     feats_all = []
#     with torch.no_grad():
#         for x, _ in train_loader:
#             x = x.to(device)
#             feats_list = adapter(backbone(x))
#             Hmax = max(f.size(2) for f in feats_list)
#             Wmax = max(f.size(3) for f in feats_list)
#             ups = [F.interpolate(f, size=(Hmax, Wmax), mode="bilinear", align_corners=False) for f in feats_list]
#             Fcat = torch.cat(ups, dim=1)
#             P = _l2n(Fcat).permute(0, 2, 3, 1).reshape(-1, Fcat.size(1))
#             feats_all.append(P.cpu())
#     feats_all = torch.cat(feats_all, dim=0)

#     # memory_baseline = MemoryBank()
#     # # Adjust args here if your MemoryBank uses a different constructor/init signature
#     # memory_baseline.init_from_feats(feats_all.numpy(), coreset_fraction=0.3, seed=123)

#     # ✅ Corrected initialization (depending on your API)
    
#     memory_baseline = MemoryBank()
#     memory_baseline.fit(feats_all.numpy(), coreset_fraction=0.3, seed=123)  # <--- changed

#     # Score test set (pre-AL)
#     s_base, heat_base, imgs_base = score_images_patchcore(
#         backbone, adapter, memory_baseline, test_loader, device,
#         top_q=0.10, k=3, out_size=(256, 256), collect_images=True
#     )

#     # Labels (normalize to 1=anomaly)
#     labels_base = []
#     for _, y in test_loader:
#         labels_base.extend(y.tolist())
#     labels_base = np.asarray(labels_base, dtype=np.int32)
#     if not DATASET_LABEL_ONE_IS_ANOMALY:
#         labels_base = 1 - labels_base

#     scores_base = s_base.numpy()
#     eval_base = evaluate_and_plot(
#         labels_base, scores_base, prefix="baseline",
#         outdir="runs/patchcore_al/baseline", target_precision=TARGET_PRECISION
#     )

#     # Baseline visualizations
#     os.makedirs("runs/patchcore_al/baseline_vis", exist_ok=True)
#     plot_with_heatmap(imgs_base, heat_base, idx=0, size=(256, 256), alpha=0.5,
#                       save_path="runs/patchcore_al/baseline_vis/example_matplotlib.png", dpi=128)
#     if HAVE_CV2:
#         save_side_by_side_cv2(imgs_base, heat_base, idx=0, size=(256, 256),
#                               out_path="runs/patchcore_al/baseline_vis/example_cv2.png", alpha=0.5)

#     # -------------------------------------------------------------
#     # ACTIVE LEARNING / TRAINING
#     # -------------------------------------------------------------
#     adapter_trained, memory = active_learning_patchcore_swag(
#         backbone, adapter, dataset,
#         seed_indices_normals=idx_seed,
#         pool_indices=pool_indices,
#         device=device,
#         rounds=5, budget=2000, mem_coreset=0.3,
#         swag_noise=0.01, swag_K=8, p_gate=0.90, top_q=0.10, k=3,
#         save_dir="runs/patchcore_al"
#     )

#     # -------------------------------------------------------------
#     # POST-AL EVALUATION
#     # -------------------------------------------------------------
#     print("\n=== Post-AL evaluation (after Active Learning) ===")
#     s_test, heat_test, imgs_test = score_images_patchcore(
#         backbone, adapter_trained, memory, test_loader, device,
#         top_q=0.10, k=3, out_size=(256, 256), collect_images=True
#     )

#     labels_all = []
#     for _, y in test_loader:
#         labels_all.extend(y.tolist())
#     labels_all = np.asarray(labels_all, dtype=np.int32)
#     if not DATASET_LABEL_ONE_IS_ANOMALY:
#         labels_all = 1 - labels_all

#     scores_post = s_test.numpy()
#     eval_post = evaluate_and_plot(
#         labels_all, scores_post, prefix="postAL",
#         outdir="runs/patchcore_al/postAL", target_precision=TARGET_PRECISION
#     )

#     # Post-AL visualizations
#     os.makedirs("runs/patchcore_al/test_vis", exist_ok=True)
#     plot_with_heatmap(imgs_test, heat_test, idx=0, size=(256, 256), alpha=0.5,
#                       save_path="runs/patchcore_al/test_vis/example_matplotlib.png", dpi=128)
#     if HAVE_CV2:
#         save_side_by_side_cv2(imgs_test, heat_test, idx=0, size=(256, 256),
#                               out_path="runs/patchcore_al/test_vis/example_cv2.png", alpha=0.5)

#     # -------------------------------------------------------------
#     # Summary printout (optional)
#     # -------------------------------------------------------------
#     def fmt(d, k): return "nan" if (d.get(k) is None) else f"{d.get(k):.4f}"
#     print("\n=== Summary ===")
#     print("Baseline:  AUC:", fmt(eval_base, "auc"), " AP:", fmt(eval_base, "ap"),
#           " F1@Youden:", fmt(eval_base, "f1_y"))
#     print("Post-AL :  AUC:", fmt(eval_post, "auc"), " AP:", fmt(eval_post, "ap"),
#           " F1@Youden:", fmt(eval_post, "f1_y"))


class MemoryBank:
    def __init__(self, coreset_ratio=0.2, method="auto", large_N=300_000, device="cpu"):
        self.mem = None
        self.ratio = coreset_ratio
        self.method = method  # "auto" | "random" | "greedy"
        self.large_N = large_N
        self.device = torch.device(device)

    def build(self, patch_vecs: torch.Tensor):
        N = patch_vecs.size(0)
        X = _l2n(patch_vecs.T).T.contiguous()
        if self.method == "random" or (self.method == "auto" and N > self.large_N):
            M = max(1, int(self.ratio * N))
            idx = torch.randperm(N)[:M]
            self.mem = X[idx].to(self.device, non_blocking=True)
            print(f"[PatchCore] Memory (random) {self.mem.size(0)}/{N}")
        else:
            Xd = X.to(self.device, non_blocking=True)
            self.mem, _ = coreset_greedy(Xd, ratio=self.ratio)
            self.mem = self.mem.contiguous()
            print(f"[PatchCore] Memory (greedy) {self.mem.size(0)}/{N}")

    @torch.no_grad()
    def knn_dist(
        self,
        Q: torch.Tensor,
        k: int = 3,
        q_chunk: int = 4096,
        m_chunk: int = 100_000,
        prefer_gpu: bool = True,
    ):
        """
        Returns mean k-NN distance per query vector.
        Chunked over BOTH queries and memory to avoid OOM.

        Args:
          Q: [Nq, D] on any device
          k: neighbors
          q_chunk: query batch size
          m_chunk: memory batch size
          prefer_gpu: if True, try GPU; on OOM fall back to CPU for that block
        """
        assert self.mem is not None, "MemoryBank not built."
        mem = self.mem  # [Nm, D]
        Nm = mem.size(0)
        D = mem.size(1)

        # Work in float32 for distance stability
        mem_f = mem if mem.dtype == torch.float32 else mem.float()

        outs = []
        # iterate queries
        for qi in range(0, Q.size(0), q_chunk):
            q = Q[qi:qi + q_chunk]
            q_f = q if q.dtype == torch.float32 else q.float()

            # We'll maintain best-k distances per row incrementally
            # Start with +inf
            best = torch.full((q_f.size(0), k), float("inf"))

            # move q to the working device
            def _to_dev(t, dev):
                return t.to(dev, non_blocking=True)

            # try GPU first for q
            q_dev = _to_dev(q_f, mem_f.device if (prefer_gpu and mem_f.is_cuda) else torch.device("cpu"))

            # iterate memory in chunks
            m_dev_target = q_dev.device  # compute where q is
            for mi in range(0, Nm, m_chunk):
                m_blk = mem_f[mi:mi + m_chunk]

                try:
                    m_dev = _to_dev(m_blk, m_dev_target)
                    # Compute pairwise distances for this block
                    # shape: [q_b, m_b]
                    d_blk = torch.cdist(q_dev, m_dev, p=2)

                except RuntimeError as e:
                    # Fallback to CPU for this block
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        m_dev = m_blk.cpu()
                        q_cpu = q_f.cpu()
                        d_blk = torch.cdist(q_cpu, m_dev, p=2)
                        # ensure best lives on CPU too
                        best = best.cpu()
                        q_dev = q_cpu  # subsequent blocks on CPU
                        m_dev_target = torch.device("cpu")
                    else:
                        raise

                # Update running top-k: concat then topk along mem-dim
                # best: [q_b, k], d_blk: [q_b, m_b]
                cat = torch.cat([best.to(d_blk.device), d_blk], dim=1)
                best = torch.topk(cat, k=k, largest=False, dim=1).values  # [q_b, k]
                # keep best on current device to avoid ping-pong

            # mean over k neighbors
            best_mean = best.mean(dim=1)
            outs.append(best_mean.cpu())

        return torch.cat(outs, dim=0)

if __name__ == "__main__":
    # External objects expected from your environment:
    # set_seed, AnomalyDataset, split_normal_30
    # PCBackbone, PCAdapter, active_learning_patchcore_swag
    # MemoryBank, _l2n  (and DataLoader already imported)

    # =================== CONFIG ===================
    set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Paths (EDIT THESE)
    normal_dir_train  = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/train/NORMAL"
    anomaly_dir_train = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/train/PNEUMONIA"
    normal_dir_test   = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/test/NORMAL"
    anomaly_dir_test  = "/home/usd.local/nand.yadav/rizk_lab/shared/USDN/chest_xray/test/PNEUMONIA"

    size = 256
    batch = 32

    # Label/score polarity controls
    # If dataset uses 0=NORMAL, 1=ANOMALY (typical), leave True.
    # If dataset uses 1=NORMAL, 0=ANOMALY, set False.
    DATASET_LABEL_ONE_IS_ANOMALY = True

    # Target precision for an "operating-point" threshold (besides Youden)
    TARGET_PRECISION = 0.80
    # ==============================================

    # ---- Train pool construction ----
    dataset = AnomalyDataset(normal_dir_train, anomaly_dir_train, size=size, color="L")
    idx_seed, idx_rest_normals = split_normal_30(dataset, seed=42)
    idx_anoms = [i for i in range(len(dataset)) if dataset.labels[i] == 1]
    pool_indices = idx_rest_normals + idx_anoms

    # ---- Backbone + Adapter ----
    backbone = PCBackbone(model_name="resnet50", out_indices=(2, 3), device=device)
    with torch.no_grad():
        x0, _ = dataset[0]
        x0 = x0.unsqueeze(0).to(device)
        feats0 = backbone(x0)
        in_chs = [f.size(1) for f in feats0]
    adapter = PCAdapter(in_chs=in_chs, out_dim=256).to(device)

    # ---- TEST SET LOADER (used for both pre-AL and post-AL) ----
    assert os.path.isdir(normal_dir_test) and os.path.isdir(anomaly_dir_test), "Test folders not found."
    test_dataset = AnomalyDataset(normal_dir_test, anomaly_dir_test, size=size, color="L")
    test_loader  = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    # -----------------------------------------------------------------
    # PRE-AL BASELINE: build naive memory bank from TRAIN set features
    # -----------------------------------------------------------------
    print("\n=== Baseline evaluation (before Active Learning) ===")
    train_loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    feats_all = []
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            feats_list = adapter(backbone(x))
            Hmax = max(f.size(2) for f in feats_list)
            Wmax = max(f.size(3) for f in feats_list)
            ups = [F.interpolate(f, size=(Hmax, Wmax), mode="bilinear", align_corners=False) for f in feats_list]
            Fcat = torch.cat(ups, dim=1)
            P = _l2n(Fcat).permute(0, 2, 3, 1).reshape(-1, Fcat.size(1))
            feats_all.append(P.cpu())
    feats_all = torch.cat(feats_all, dim=0).contiguous().to(dtype=torch.float32)

    # ✅ Build MemoryBank using YOUR API (build)
    memory_baseline = MemoryBank(coreset_ratio=0.3, method="auto", device=str(device))
    memory_baseline.build(feats_all)

    # Score test set (pre-AL)
    s_base, heat_base, imgs_base = score_images_patchcore(
        backbone, adapter, memory_baseline, test_loader, device,
        top_q=0.10, k=3, out_size=(256, 256), collect_images=True
    )

    # Labels (normalize to 1=anomaly)
    labels_base = []
    for _, y in test_loader:
        labels_base.extend(y.tolist())
    labels_base = np.asarray(labels_base, dtype=np.int32)
    if not DATASET_LABEL_ONE_IS_ANOMALY:
        labels_base = 1 - labels_base

    scores_base = s_base.numpy()
    eval_base = evaluate_and_plot(
        labels_base, scores_base, prefix="baseline",
        outdir="runs/patchcore_al/baseline", target_precision=TARGET_PRECISION
    )

    # Baseline visualizations
    os.makedirs("runs/patchcore_al/baseline_vis", exist_ok=True)
    plot_with_heatmap(imgs_base, heat_base, idx=0, size=(256, 256), alpha=0.5,
                      save_path="runs/patchcore_al/baseline_vis/example_matplotlib.png", dpi=128)
    if HAVE_CV2:
        save_side_by_side_cv2(imgs_base, heat_base, idx=0, size=(256, 256),
                              out_path="runs/patchcore_al/baseline_vis/example_cv2.png", alpha=0.5)

    # -------------------------------------------------------------
    # ACTIVE LEARNING / TRAINING
    # -------------------------------------------------------------
    adapter_trained, memory = active_learning_patchcore_swag(
        backbone, adapter, dataset,
        seed_indices_normals=idx_seed,
        pool_indices=pool_indices,
        device=device,
        rounds=5, budget=200, mem_coreset=0.3,
        swag_noise=0.01, swag_K=8, p_gate=0.95, top_q=0.10, k=3,
        save_dir="runs/patchcore_al"
    )

    # -------------------------------------------------------------
    # POST-AL EVALUATION
    # -------------------------------------------------------------
    print("\n=== Post-AL evaluation (after Active Learning) ===")
    s_test, heat_test, imgs_test = score_images_patchcore(
        backbone, adapter_trained, memory, test_loader, device,
        top_q=0.10, k=3, out_size=(256, 256), collect_images=True
    )

    labels_all = []
    for _, y in test_loader:
        labels_all.extend(y.tolist())
    labels_all = np.asarray(labels_all, dtype=np.int32)
    if not DATASET_LABEL_ONE_IS_ANOMALY:
        labels_all = 1 - labels_all

    scores_post = s_test.numpy()
    eval_post = evaluate_and_plot(
        labels_all, scores_post, prefix="postAL",
        outdir="runs/patchcore_al/postAL", target_precision=TARGET_PRECISION
    )

    # Post-AL visualizations
    os.makedirs("runs/patchcore_al/test_vis", exist_ok=True)
    plot_with_heatmap(imgs_test, heat_test, idx=0, size=(256, 256), alpha=0.5,
                      save_path="runs/patchcore_al/test_vis/example_matplotlib.png", dpi=128)
    if HAVE_CV2:
        save_side_by_side_cv2(imgs_test, heat_test, idx=0, size=(256, 256),
                              out_path="runs/patchcore_al/test_vis/example_cv2.png", alpha=0.5)

    # -------------------------------------------------------------
    # Summary printout (optional)
    # -------------------------------------------------------------
    def fmt(d, k): return "nan" if (d.get(k) is None) else f"{d.get(k):.4f}"
    print("\n=== Summary ===")
    print("Baseline:  AUC:", fmt(eval_base, "auc"), " AP:", fmt(eval_base, "ap"),
          " F1@Youden:", fmt(eval_base, "f1_y"))
    print("Post-AL :  AUC:", fmt(eval_post, "auc"), " AP:", fmt(eval_post, "ap"),
          " F1@Youden:", fmt(eval_post, "f1_y"))
