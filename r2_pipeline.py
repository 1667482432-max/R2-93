from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
BS_P, BS_H, BS_V = 2, 16, 8


def bs_fft_torch(channel):
    """2-D BS-array FFT for flattened polarization -> H -> V storage."""
    shaped = channel.reshape(*channel.shape[:-3], BS_P, BS_H, BS_V,
                             channel.shape[-2], channel.shape[-1])
    import torch
    angular = torch.fft.fft2(shaped, dim=(-4, -3), norm="ortho")
    return angular.reshape_as(channel)


def bs_ifft_torch(angular):
    shaped = angular.reshape(*angular.shape[:-3], BS_P, BS_H, BS_V,
                             angular.shape[-2], angular.shape[-1])
    import torch
    channel = torch.fft.ifft2(shaped, dim=(-4, -3), norm="ortho")
    return channel.reshape_as(angular)


def bs_fft_numpy(channel):
    shaped = channel.reshape(*channel.shape[:-3], BS_P, BS_H, BS_V,
                             channel.shape[-2], channel.shape[-1])
    return np.fft.fft2(shaped, axes=(-4, -3), norm="ortho").reshape(channel.shape)


def bs_ifft_numpy(angular):
    shaped = angular.reshape(*angular.shape[:-3], BS_P, BS_H, BS_V,
                             angular.shape[-2], angular.shape[-1])
    return np.fft.ifft2(shaped, axes=(-4, -3), norm="ortho").reshape(angular.shape)


def load_data():
    pos = np.load(ROOT / "Round2_Train_Pos.npy")
    channel = np.load(ROOT / "Round2_Train_Channel.npy", mmap_mode="r")
    energy_path = ROOT / "train_energy.npy"
    if energy_path.exists():
        energy = np.load(energy_path)
    else:
        energy = np.empty(len(channel), np.float64)
        for start in range(0, len(channel), 8):
            x = np.asarray(channel[start : start + 8])
            energy[start : start + len(x)] = np.sum(
                x.real.astype(np.float64) ** 2 + x.imag.astype(np.float64) ** 2,
                axis=(1, 2, 3),
            )
        np.save(energy_path, energy)
    return pos, channel, energy


# Width, height and population recovered from the eight official test islands.
TEST_BLOCKS = [
    (40.3, 45.1, 90), (30.9, 33.3, 55), (29.2, 32.4, 43), (46.5, 34.5, 41),
    (42.9, 41.1, 67), (37.0, 34.4, 63), (21.7, 30.1, 50), (49.3, 32.0, 48),
]


def make_block_split(fold: int = 0, scale: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    pos, _, energy = load_data()
    valid_nonzero = energy > 0
    chosen: list[int] = []
    # Fixed folds are reproducible. Candidate centers are actual sample positions.
    rng = np.random.default_rng(20260813 + fold)
    for side in range(2):
        side_mask = (pos[:, 1] < 0) if side == 0 else (pos[:, 1] > 0)
        candidates = np.flatnonzero(side_mask & valid_nonzero)
        rng.shuffle(candidates)
        occupied = np.zeros(len(pos), dtype=bool)
        occupied[chosen] = True
        for width, height, target in TEST_BLOCKS[side * 4 : side * 4 + 4]:
            width, height = width * scale, height * scale
            target = max(12, round(target * scale * scale))
            best = None
            for center_idx in candidates[:1200]:
                center = pos[center_idx, :2]
                inside = (
                    side_mask & valid_nonzero & ~occupied
                    & (np.abs(pos[:, 0] - center[0]) <= width / 2)
                    & (np.abs(pos[:, 1] - center[1]) <= height / 2)
                )
                ids = np.flatnonzero(inside)
                if len(ids) < max(12, int(target * 0.55)):
                    continue
                # Prefer official-like populations; select all points in the rectangle.
                penalty = abs(len(ids) - target)
                if best is None or penalty < best[0]:
                    best = (penalty, ids)
                    if penalty == 0:
                        break
            if best is None:
                raise RuntimeError("Unable to construct rectangular validation block")
            ids = best[1]
            chosen.extend(ids.tolist())
            # Prevent overlap and enforce a buffer so validation points are a real hole.
            xmin, ymin = pos[ids, :2].min(0) - 3
            xmax, ymax = pos[ids, :2].max(0) + 3
            occupied |= (
                (pos[:, 0] >= xmin) & (pos[:, 0] <= xmax)
                & (pos[:, 1] >= ymin) & (pos[:, 1] <= ymax)
            )
    val = np.unique(np.asarray(chosen, dtype=np.int64))
    train = np.flatnonzero(valid_nonzero & ~np.isin(np.arange(len(pos)), val))
    return train, val


def make_official_region_split() -> tuple[np.ndarray, np.ndarray]:
    """Validation points inside the exact official test rectangles with matched NN CDF."""
    _, _, energy = load_data()
    val_path = ROOT / os.environ.get("R2_VAL_FILE", "official_region_val.npy")
    if not val_path.exists():
        raise FileNotFoundError("Run the official-region split optimizer first")
    val = np.load(val_path).astype(np.int64)
    valid = energy > 0
    if not np.all(valid[val]):
        raise RuntimeError("Official-region validation contains zero-channel outliers")
    train = np.flatnonzero(valid & ~np.isin(np.arange(len(valid)), val))
    return train, val


def describe_split(fold: int):
    from sklearn.cluster import DBSCAN
    pos, _, energy = load_data()
    train, val = make_block_split(fold)
    d = cKDTree(pos[train,:2]).query(pos[val,:2], k=1)[0]
    labels = DBSCAN(eps=10, min_samples=3).fit_predict(pos[val,:2])
    print(json.dumps(dict(fold=fold, n=len(val), distance_q=np.quantile(d,[0,.1,.5,.9,1]).tolist())))
    for label in sorted(set(labels)):
        ids=val[labels==label]
        print(json.dumps(dict(cluster=int(label), n=len(ids),
                              bounds=[pos[ids,:2].min(0).tolist(),pos[ids,:2].max(0).tolist()],
                              energy_median=float(np.median(energy[ids])))))


def neighbor_plan(train_idx: np.ndarray, query_pos: np.ndarray, k: int):
    pos = np.load(ROOT / "Round2_Train_Pos.npy")
    tree = cKDTree(pos[train_idx, :2])
    dist, local = tree.query(query_pos[:, :2], k=k)
    if k == 1:
        dist, local = dist[:, None], local[:, None]
    return dist, train_idx[local]


def delaunay_plan(train_idx: np.ndarray, query_pos: np.ndarray):
    from scipy.spatial import Delaunay
    pos = np.load(ROOT / "Round2_Train_Pos.npy")
    xy = pos[train_idx, :2]
    tri = Delaunay(xy)
    simplex = tri.find_simplex(query_pos[:, :2])
    indices = np.empty((len(query_pos), 3), np.int64)
    weights = np.empty((len(query_pos), 3), np.float64)
    inside = simplex >= 0
    s = simplex[inside]
    delta = query_pos[inside, :2] - tri.transform[s, 2]
    bary = np.einsum("nij,nj->ni", tri.transform[s, :2], delta)
    weights[inside] = np.column_stack([bary, 1 - bary.sum(1)])
    indices[inside] = train_idx[tri.simplices[s]]
    if np.any(~inside):
        dist, ids = neighbor_plan(train_idx, query_pos[~inside], 3)
        w = 1 / np.maximum(dist, .25) ** 2
        weights[~inside] = w / w.sum(1, keepdims=True)
        indices[~inside] = ids
    return indices, weights


def delaunay_oracle(fold: int):
    import torch
    pos, channel, _ = load_data()
    train, val = make_block_split(fold)
    idx, weights = delaunay_plan(train, pos[val])
    device = torch.device("cuda")
    sums = [0., 0.]
    for start in range(0, len(val), 8):
        stop = min(start + 8, len(val))
        t = torch.as_tensor(np.asarray(channel[val[start:stop]]).copy(), device=device)
        tp = torch.abs(bs_fft_torch(t)) ** 2
        td = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
        h = torch.as_tensor(np.asarray(channel[idx[start:stop]]).copy(), device=device)
        pp = torch.abs(bs_fft_torch(h)) ** 2
        pd = torch.abs(torch.fft.fft(h, dim=4, norm="ortho")) ** 2
        pp /= torch.linalg.vector_norm(pp, dim=2, keepdim=True).clamp_min(1e-30)
        pd /= torch.linalg.vector_norm(pd, dim=4, keepdim=True).clamp_min(1e-30)
        w = torch.as_tensor(weights[start:stop], dtype=torch.float32, device=device)
        pp = torch.sum(w[:,:,None,None,None]*pp, dim=1).clamp_min(0)
        pd = torch.sum(w[:,:,None,None,None]*pd, dim=1).clamp_min(0)
        sums[0] += float(((pp*tp).sum(1)/(torch.linalg.vector_norm(pp,dim=1)*torch.linalg.vector_norm(tp,dim=1)).clamp_min(1e-30)).sum())
        sums[1] += float(((pd*td).sum(-1)/(torch.linalg.vector_norm(pd,dim=-1)*torch.linalg.vector_norm(td,dim=-1)).clamp_min(1e-30)).sum())
    c1=sums[0]/(len(val)*4*192); c2=sums[1]/(len(val)*256*4)
    print(json.dumps(dict(method="delaunay_norm", c1_pas=c1,c2_pdp=c2,score=.4*c1+.4*c2+.1)))


def predict_knn(indices: np.ndarray, dist: np.ndarray, method: str, power: float = 2.0):
    channel = np.load(ROOT / "Round2_Train_Channel.npy", mmap_mode="r")
    if method == "nearest":
        return np.asarray(channel[indices[:, 0]])
    w = 1.0 / np.maximum(dist, 0.25) ** power
    w /= w.sum(axis=1, keepdims=True)
    out = np.empty((len(indices), 256, 4, 192), np.complex64)
    for i in range(len(indices)):
        out[i] = np.tensordot(w[i].astype(np.float32), np.asarray(channel[indices[i]]), axes=1)
    return out


def predict_weighted(indices: np.ndarray, weights: np.ndarray):
    channel = np.load(ROOT / "Round2_Train_Channel.npy", mmap_mode="r")
    out = np.empty((len(indices), 256, 4, 192), np.complex64)
    for i in range(len(indices)):
        out[i] = np.tensordot(weights[i].astype(np.float32), np.asarray(channel[indices[i]]), axes=1)
    return out


def project_spectra(indices: np.ndarray, weights: np.ndarray, initial: np.ndarray,
                    iterations: int = 6, blend: float = 1.0):
    channel = np.load(ROOT / "Round2_Train_Channel.npy", mmap_mode="r")
    out = np.empty_like(initial)
    for i in range(len(indices)):
        neighbors = np.asarray(channel[indices[i]])
        target_pas = np.tensordot(
            weights[i], np.abs(bs_fft_numpy(neighbors)) ** 2, axes=1)
        target_pdp = np.tensordot(
            weights[i], np.abs(np.fft.fft(neighbors, axis=-1, norm="ortho")) ** 2, axes=1)
        x = initial[i].astype(np.complex128)
        for _ in range(iterations):
            z = bs_fft_numpy(x)
            desired = np.sqrt(np.maximum(target_pas, 0))
            z *= (1-blend) + blend * desired / np.maximum(np.abs(z), 1e-30)
            x = bs_ifft_numpy(z)
            z = np.fft.fft(x, axis=-1, norm="ortho")
            desired = np.sqrt(np.maximum(target_pdp, 0))
            z *= (1-blend) + blend * desired / np.maximum(np.abs(z), 1e-30)
            x = np.fft.ifft(z, axis=-1, norm="ortho")
        out[i] = x.astype(np.complex64)
        if i % 25 == 0:
            print('projected', i, flush=True)
    return out


def spatial_weights(train_pos: np.ndarray, query_pos: np.ndarray, indices: np.ndarray,
                    dist: np.ndarray, method: str, param: float):
    if method == "gaussian":
        scale = np.maximum(dist[:, [-1]] * param, 0.5)
        w = np.exp(-0.5 * (dist / scale) ** 2)
    elif method == "idw":
        w = 1 / np.maximum(dist, 0.25) ** param
    elif method == "linear":
        # Local affine regression weights, with ridge regularization scaled by radius.
        w = np.empty_like(dist)
        for i in range(len(dist)):
            x = train_pos[indices[i], :2] - query_pos[i, :2]
            a = np.column_stack([np.ones(len(x)), x])
            ridge = param * max(float(np.mean(dist[i]) ** 2), 1.0)
            gram = a.T @ a + np.diag([1e-8, ridge, ridge])
            w[i] = a @ np.linalg.solve(gram, np.array([1.0, 0.0, 0.0]))
    else:
        raise ValueError(method)
    return w / w.sum(axis=1, keepdims=True)


def score_numpy(pred: np.ndarray, target: np.ndarray, batch: int = 4):
    pas_sum = pdp_sum = err = energy = 0.0
    pas_n = pdp_n = 0
    for s in range(0, len(pred), batch):
        p = np.asarray(pred[s : s + batch])
        t = np.asarray(target[s : s + batch])
        pp = np.abs(bs_fft_numpy(p)) ** 2
        tp = np.abs(bs_fft_numpy(t)) ** 2
        num = np.sum(pp * tp, axis=1)
        den = np.linalg.norm(pp, axis=1) * np.linalg.norm(tp, axis=1)
        pas_sum += np.sum(num / np.maximum(den, 1e-30)); pas_n += num.size
        pp = np.abs(np.fft.fft(p, axis=-1, norm="ortho")) ** 2
        tp = np.abs(np.fft.fft(t, axis=-1, norm="ortho")) ** 2
        num = np.sum(pp * tp, axis=-1)
        den = np.linalg.norm(pp, axis=-1) * np.linalg.norm(tp, axis=-1)
        pdp_sum += np.sum(num / np.maximum(den, 1e-30)); pdp_n += num.size
        err += np.sum(np.abs(p - t) ** 2, dtype=np.float64)
        energy += np.sum(np.abs(t) ** 2, dtype=np.float64)
    c1, c2, c3 = map(float, (pas_sum / pas_n, pdp_sum / pdp_n, err / energy))
    return {"c1_pas": c1, "c2_pdp": c2, "c3_nmse": c3,
            "score": float(0.4*c1 + 0.4*c2 + 0.2/(1+c3))}


def score_numpy_weighted(pred, target, sample_weights, batch=4):
    sw=np.asarray(sample_weights,dtype=np.float64);pas=pdp=err=energy=0.;pas_den=pdp_den=0.
    for s in range(0,len(pred),batch):
        e=min(s+batch,len(pred));w=sw[s:e];p=np.asarray(pred[s:e]);t=np.asarray(target[s:e])
        pp=np.abs(bs_fft_numpy(p))**2;tt=np.abs(bs_fft_numpy(t))**2;c=np.sum(pp*tt,axis=1)/(np.linalg.norm(pp,axis=1)*np.linalg.norm(tt,axis=1)).clip(1e-30);pas+=np.sum(c*w[:,None,None]);pas_den+=np.sum(w)*c.shape[1]*c.shape[2]
        pp=np.abs(np.fft.fft(p,axis=-1,norm='ortho'))**2;tt=np.abs(np.fft.fft(t,axis=-1,norm='ortho'))**2;c=np.sum(pp*tt,axis=-1)/(np.linalg.norm(pp,axis=-1)*np.linalg.norm(tt,axis=-1)).clip(1e-30);pdp+=np.sum(c*w[:,None,None]);pdp_den+=np.sum(w)*c.shape[1]*c.shape[2]
        err+=np.sum(np.abs(p-t)**2*w[:,None,None,None]);energy+=np.sum(np.abs(t)**2*w[:,None,None,None])
    c1=pas/pas_den;c2=pdp/pdp_den;c3=err/energy;return dict(c1_pas=c1,c2_pdp=c2,c3_nmse=c3,score=.4*c1+.4*c2+.2/(1+c3))


def validate(fold: int):
    pos, channel, _ = load_data()
    train, val = make_block_split(fold)
    dist, idx = neighbor_plan(train, pos[val], 12)
    print(json.dumps({"fold": fold, "train": len(train), "val": len(val),
                      "distance_quantiles": np.quantile(dist[:, 0], [0,.1,.5,.9,1]).tolist()}))
    target = channel[val]
    configs = [("nearest", 1, 0), ("idw", 2, 1), ("idw", 4, 1),
               ("idw", 4, 2), ("idw", 8, 2), ("idw", 12, 2)]
    results = []
    for method, k, power in configs:
        pred = predict_knn(idx[:, :k], dist[:, :k], method, power)
        result = score_numpy(pred, target)
        result.update(method=method, k=k, power=power)
        results.append(result)
        print(json.dumps(result))
    return results


def sweep(fold: int):
    pos, channel, _ = load_data()
    train, val = make_block_split(fold)
    dist, idx = neighbor_plan(train, pos[val], 32)
    target = channel[val]
    configs = []
    for k in [4, 8, 12, 16, 24, 32]:
        for method, params in [("idw", [0.5, 1, 1.5, 2, 3]),
                               ("gaussian", [0.35, 0.5, 0.75, 1, 1.5])]:
            for param in params:
                configs.append((method, k, param))
    for k in [8, 12, 16, 24, 32]:
        for param in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            configs.append(("linear", k, param))
    for method, k, param in configs:
        w = spatial_weights(pos, pos[val], idx[:, :k], dist[:, :k], method, param)
        pred = predict_weighted(idx[:, :k], w)
        # Optimal real scalar for the NMSE term. Spectral cosine terms are invariant.
        cross = np.vdot(pred, target)
        pred_e = float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
        alpha = cross / max(pred_e, 1e-30)
        result = score_numpy(pred * alpha, target)
        result.update(method=method, k=k, param=param, alpha=alpha)
        print(json.dumps(result), flush=True)


def spectral_validate(fold: int):
    pos, channel, _ = load_data()
    train, val = make_block_split(fold)
    dist, idx = neighbor_plan(train, pos[val], 16)
    target = channel[val]
    for k, power in [(4,1), (8,1), (8,2), (12,1), (12,2)]:
        w = spatial_weights(pos, pos[val], idx[:, :k], dist[:, :k], "idw", power)
        base = predict_weighted(idx[:, :k], w)
        for iters, blend in [(2,.5),(4,.5),(4,1.0),(8,.75)]:
            pred = project_spectra(idx[:, :k], w, base, iters, blend)
            cross = float(np.real(np.vdot(pred, target)))
            pred_e = float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
            alpha = max(0.0, cross / max(pred_e, 1e-30))
            result = score_numpy(pred * alpha, target)
            result.update(method="spectral", k=k, power=power, iterations=iters,
                          blend=blend, alpha=alpha)
            print(json.dumps(result), flush=True)


def project_spectra_gpu(indices: np.ndarray, weights: np.ndarray,
                        iterations: int = 4, blend: float = 1.0,
                        batch_size: int = 6, normalize_shapes: bool = False):
    """Batched CUDA implementation; each neighbor channel is transformed once/batch."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fast spectral path")
    channel = np.load(ROOT / "Round2_Train_Channel.npy", mmap_mode="r")
    out = np.empty((len(indices), 256, 4, 192), np.complex64)
    device = torch.device("cuda")
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        ids = indices[start:stop]
        w = torch.as_tensor(weights[start:stop], device=device, dtype=torch.float32)
        h = torch.as_tensor(np.asarray(channel[ids]).copy(), device=device)
        x = torch.sum(w[:, :, None, None, None] * h, dim=1)
        neighbor_pas = torch.abs(bs_fft_torch(h)) ** 2
        neighbor_pdp = torch.abs(torch.fft.fft(h, dim=4, norm="ortho")) ** 2
        if normalize_shapes:
            neighbor_pas /= torch.linalg.vector_norm(
                neighbor_pas, dim=2, keepdim=True).clamp_min(1e-30)
            neighbor_pdp /= torch.linalg.vector_norm(
                neighbor_pdp, dim=4, keepdim=True).clamp_min(1e-30)
        target_pas = torch.sum(w[:, :, None, None, None] * neighbor_pas, dim=1)
        target_pdp = torch.sum(w[:, :, None, None, None] * neighbor_pdp, dim=1)
        if normalize_shapes:
            # Match global energy so the two marginal constraints are compatible.
            target_pdp *= (target_pas.sum(dim=(1,2,3)) /
                           target_pdp.sum(dim=(1,2,3)).clamp_min(1e-30))[:,None,None,None]
        del h
        for _ in range(iterations):
            z = bs_fft_torch(x)
            ratio = torch.sqrt(target_pas.clamp_min(0)) / torch.abs(z).clamp_min(1e-30)
            z = z * ((1.0 - blend) + blend * ratio)
            x = bs_ifft_torch(z)
            z = torch.fft.fft(x, dim=-1, norm="ortho")
            ratio = torch.sqrt(target_pdp.clamp_min(0)) / torch.abs(z).clamp_min(1e-30)
            z = z * ((1.0 - blend) + blend * ratio)
            x = torch.fft.ifft(z, dim=-1, norm="ortho")
        out[start:stop] = x.cpu().numpy().astype(np.complex64)
        del x, z, target_pas, target_pdp
        print(f"projected {stop}/{len(indices)}", flush=True)
    return out


def project_spectra_gpu_dual(pas_indices, pas_weights, pdp_indices, pdp_weights,
                             iterations=20, batch_size=4, pas_q=1.0, pdp_q=1.0,
                             final_pas_blend=None, group=None, query_xy=None):
    import torch
    channel=np.load(ROOT/'Round2_Train_Channel.npy',mmap_mode='r');device='cuda'
    out=np.empty((len(pas_indices),256,4,192),np.complex64)
    batch_size=min(batch_size,max(1,192//max(pas_indices.shape[1],pdp_indices.shape[1])))
    for s in range(0,len(out),batch_size):
        e=min(s+batch_size,len(out))
        wp=torch.as_tensor(pas_weights[s:e],device=device,dtype=torch.float32)
        hp=torch.as_tensor(np.asarray(channel[pas_indices[s:e]]).copy(),device=device)
        npas=torch.abs(bs_fft_torch(hp))**2
        raw_npas=npas
        npas=npas/torch.linalg.vector_norm(npas,dim=2,keepdim=True).clamp_min(1e-30)
        if group is not None and os.environ.get('R2_PAS_MOMENT_ALIGN'):
            npas=align_pas_neighbors(npas,wp,int(group))
        if group is not None and os.environ.get('R2_PAS_XCORR_ALIGN'):
            npas=align_pas_neighbors_xcorr(npas,wp,int(group))
        target_pas=(wp[:,:,None,None,None]*npas.pow(pas_q)).sum(1).clamp_min(1e-12).pow(1.0/pas_q)
        if group is not None and os.environ.get('R2_PAS_LOW_RANK'):
            target_pas=low_rank_pas_target(target_pas,int(group))
        if group is not None and os.environ.get('R2_PAS_POOL'):
            target_pas=pool_pas_target(target_pas,int(group))
        if group is not None and os.environ.get('R2_PAS_SMOOTH'):
            target_pas=smooth_pas_target(target_pas,int(group))
        if group is not None and query_xy is not None and os.environ.get('R2_PAS_LOCAL_SHIFT'):
            target_pas=local_shift_pas_target(target_pas,raw_npas,pas_indices[s:e],query_xy[s:e],int(group))
        wd=torch.as_tensor(pdp_weights[s:e],device=device,dtype=torch.float32)
        hd=torch.as_tensor(np.asarray(channel[pdp_indices[s:e]]).copy(),device=device)
        npdp=torch.abs(torch.fft.fft(hd,dim=4,norm='ortho'))**2
        npdp/=torch.linalg.vector_norm(npdp,dim=4,keepdim=True).clamp_min(1e-30)
        if group is not None and os.environ.get('R2_PDP_XCORR_ALIGN'):
            npdp=align_pdp_neighbors_xcorr(npdp,wd,int(group))
        target_pdp=(wd[:,:,None,None,None]*npdp.pow(pdp_q)).sum(1).clamp_min(1e-12).pow(1.0/pdp_q)
        if group is not None and os.environ.get('R2_PDP_POOL'):
            target_pdp=pool_pdp_target(target_pdp,int(group))
        if group is not None and os.environ.get('R2_PDP_SMOOTH'):
            target_pdp=smooth_pdp_target(target_pdp,int(group))
        target_pdp*= (target_pas.sum((1,2,3))/target_pdp.sum((1,2,3)).clamp_min(1e-30))[:,None,None,None]
        reverse=os.environ.get('R2_PROJECTION_ORDER','pdp_last')=='pas_last'
        final_pas=float(os.environ.get('R2_FINAL_PAS_BLEND',final_pas_blend if final_pas_blend is not None else '.4'))
        refine_steps=int(os.environ.get('R2_REFINE_STEPS','0'))
        modes=(os.environ['R2_MULTI_START'].split(',') if os.environ.get('R2_MULTI_START')
               else [os.environ.get('R2_INIT','weighted')])
        best_x=None;best_quality=None
        for init_mode in modes:
            if init_mode.startswith('nearest'):
                init_rank=int(init_mode[7:] or '1')-1
                x=hp[:,init_rank].clone()
            else:
                x=(wp[:,:,None,None,None]*hp).sum(1)
            for _ in range(iterations):
                if reverse:
                    z=torch.fft.fft(x,dim=-1,norm='ortho');z*=torch.sqrt(target_pdp)/torch.abs(z).clamp_min(1e-30);x=torch.fft.ifft(z,dim=-1,norm='ortho')
                    z=bs_fft_torch(x);z*=torch.sqrt(target_pas)/torch.abs(z).clamp_min(1e-30);x=bs_ifft_torch(z)
                else:
                    z=bs_fft_torch(x);z*=torch.sqrt(target_pas)/torch.abs(z).clamp_min(1e-30);x=bs_ifft_torch(z)
                    z=torch.fft.fft(x,dim=-1,norm='ortho');z*=torch.sqrt(target_pdp)/torch.abs(z).clamp_min(1e-30);x=torch.fft.ifft(z,dim=-1,norm='ortho')
            if final_pas and not reverse:
                z=bs_fft_torch(x)
                ratio=torch.sqrt(target_pas)/torch.abs(z).clamp_min(1e-30)
                z*=1-final_pas+final_pas*ratio
                x=bs_ifft_torch(z)
            if refine_steps:
                x=x.detach().requires_grad_(True)
                opt=torch.optim.Adam([x],lr=float(os.environ.get('R2_REFINE_LR','.003')))
                for _ in range(refine_steps):
                    opt.zero_grad(set_to_none=True)
                    pp=torch.abs(bs_fft_torch(x))**2
                    pd=torch.abs(torch.fft.fft(x,dim=-1,norm='ortho'))**2
                    cp=(pp*target_pas).sum(1)/(torch.linalg.vector_norm(pp,dim=1)*torch.linalg.vector_norm(target_pas,dim=1)).clamp_min(1e-30)
                    cd=(pd*target_pdp).sum(-1)/(torch.linalg.vector_norm(pd,dim=-1)*torch.linalg.vector_norm(target_pdp,dim=-1)).clamp_min(1e-30)
                    loss=-(cp.mean()+cd.mean())/2
                    loss.backward();opt.step()
                x=x.detach()
            pp=torch.abs(bs_fft_torch(x))**2
            pd=torch.abs(torch.fft.fft(x,dim=-1,norm='ortho'))**2
            cp=(pp*target_pas).sum(1)/(torch.linalg.vector_norm(pp,dim=1)*torch.linalg.vector_norm(target_pas,dim=1)).clamp_min(1e-30)
            cd=(pd*target_pdp).sum(-1)/(torch.linalg.vector_norm(pd,dim=-1)*torch.linalg.vector_norm(target_pdp,dim=-1)).clamp_min(1e-30)
            quality=cp.mean((1,2))+cd.mean((1,2))
            if best_x is None:
                best_x=x;best_quality=quality
            else:
                take=quality>best_quality
                best_x=torch.where(take[:,None,None,None],x,best_x)
                best_quality=torch.where(take,quality,best_quality)
        x=best_x
        out[s:e]=x.cpu().numpy().astype(np.complex64);print('dual',e,len(out),flush=True)
    return out


def bearing_coords(pos,a=50.):
    xy=pos[:,:2];side=(xy[:,1]>0).astype(int);bs=np.array([[-18.413,-65.881],[52.,35.]])[side]
    rel=xy-bs;ang=np.arctan2(rel[:,1],rel[:,0])
    return np.c_[xy,a*np.cos(ang),a*np.sin(ang)]


def official_island_labels(query):
    from sklearn.cluster import DBSCAN
    test=np.load(ROOT/'Round2_Test_Pos.npy');labels=DBSCAN(eps=10,min_samples=3).fit_predict(test[:,:2])
    out=np.full(len(query),-1,np.int64)
    for g in sorted(set(labels)):
        z=test[labels==g,:2];out[np.all(query[:,:2]>=z.min(0),1)&np.all(query[:,:2]<=z.max(0),1)]=g
    return out


ISLAND_PARAMS={
 0:((48,1.5,.25),(24,2.0,.25)), 1:((8,4.0,1.0),(48,1.5,.8)), 3:((24,1.5,.25),(16,1.0,.5)),
 4:((192,2.0,.5),(256,1.0,.25)), 5:((16,3.0,.65),(8,1.0,.5)),
 6:((32,2.0,.8),(96,1.5,.25)), 7:((128,4.0,1.25),(16,2.0,.25)),
 8:((8,3.5,.25),(64,4.0,.25)), 9:((64,3.5,.8),(64,3.0,.65)),
 10:((48,2.0,1.0),(32,1.0,1.25)),
}
ISLAND_PARAMS_DENSITY={**ISLAND_PARAMS,
 5:((16,1.25,.5),(24,1.25,.5)),6:((8,1.0,1.25),(8,1.0,1.25)),
 8:((8,4.0,.25),(8,2.25,.25)),10:((8,1.0,.65),(64,1.5,.65)),
}
ISLAND_PARAMS_SHADOW_ROBUST={**ISLAND_PARAMS,
 1:(ISLAND_PARAMS[1][0],(48,1.0,.25)),
 4:(ISLAND_PARAMS[4][0],(32,2.5,.8)),
 5:((8,2.5,.25),(32,2.0,.5)),
 6:((24,1.5,.65),(96,2.0,.25)),
 8:(ISLAND_PARAMS[8][0],(256,3.0,.8)),
 9:(ISLAND_PARAMS[9][0],(128,2.0,.35)),
 10:(ISLAND_PARAMS[10][0],(256,2.5,1.0)),
}
ISLAND_PARAMS_SHADOW_PAS2D_V5={**ISLAND_PARAMS_SHADOW_ROBUST,
 1:((32,3.0,.25),ISLAND_PARAMS_SHADOW_ROBUST[1][1]),
 3:((24,2.0,.25),ISLAND_PARAMS_SHADOW_ROBUST[3][1]),
 4:((32,2.0,.25),ISLAND_PARAMS_SHADOW_ROBUST[4][1]),
 9:((16,2.5,.25),ISLAND_PARAMS_SHADOW_ROBUST[9][1]),
}
ISLAND_PARAMS_SHADOW_PAS2D_V5_CONSERVATIVE={**ISLAND_PARAMS_SHADOW_ROBUST,
 1:((8,3.5,1.0),ISLAND_PARAMS_SHADOW_ROBUST[1][1]),
 3:((24,2.0,.25),ISLAND_PARAMS_SHADOW_ROBUST[3][1]),
 4:((96,2.5,.25),ISLAND_PARAMS_SHADOW_ROBUST[4][1]),
 9:((48,2.5,.25),ISLAND_PARAMS_SHADOW_ROBUST[9][1]),
}
ISLAND_PARAMS_SHADOW_PAS2D_V5_ULTRA={**ISLAND_PARAMS_SHADOW_ROBUST,
 4:((96,2.5,.25),ISLAND_PARAMS_SHADOW_ROBUST[4][1]),
 9:((48,2.5,.25),ISLAND_PARAMS_SHADOW_ROBUST[9][1]),
}
ISLAND_PARAMS_SHADOW_PAS2D_V5_EXTENDED={**ISLAND_PARAMS_SHADOW_PAS2D_V5,
 2:((128,1.5,1.0),(64,2.5,1.25)),
 7:((256,4.0,1.0),ISLAND_PARAMS_SHADOW_ROBUST[7][1]),
}
ISLAND_PARAMS_MATCHED_CORE={**ISLAND_PARAMS_SHADOW_ROBUST,
 3:((128,3.5,.25),(32,1.25,1.25)),
 4:(ISLAND_PARAMS_SHADOW_ROBUST[4][0],(48,1.5,1.0)),
 7:(ISLAND_PARAMS_SHADOW_ROBUST[7][0],(96,3.0,1.25)),
 8:(ISLAND_PARAMS_SHADOW_ROBUST[8][0],(8,1.5,.8)),
 10:((8,2.0,.8),(24,1.25,.5)),
}
ISLAND_PARAMS_MATCHED_CORE_NOG10={**ISLAND_PARAMS_MATCHED_CORE,
 10:ISLAND_PARAMS_SHADOW_ROBUST[10],
}
ISLAND_PARAMS_MATCHED_JOINT_METRIC={**ISLAND_PARAMS_MATCHED_CORE_NOG10,
 1:(ISLAND_PARAMS_MATCHED_CORE_NOG10[1][0],(24,.75,.65)),
 8:((8,2.5,.25),ISLAND_PARAMS_MATCHED_CORE_NOG10[8][1]),
}
ISLAND_PARAMS_MATCHED_JOINT_EXTENDED={**ISLAND_PARAMS_MATCHED_JOINT_METRIC,
 0:(ISLAND_PARAMS_MATCHED_JOINT_METRIC[0][0],(48,2.5,.25)),
 5:(ISLAND_PARAMS_MATCHED_JOINT_METRIC[5][0],(256,2.0,.25)),
}
ISLAND_PARAMS_MATCHED_JOINT_MAP={**ISLAND_PARAMS_MATCHED_JOINT_EXTENDED,
 7:(ISLAND_PARAMS_MATCHED_JOINT_EXTENDED[7][0],(48,3.0,1.25)),
}
ISLAND_PARAMS_MATCHED_JOINT_G6={**ISLAND_PARAMS_MATCHED_JOINT_MAP,
 6:((24,1.5,.8),(24,1.25,.65)),
}

def active_island_params():
    if os.environ.get('R2_MATCHED_JOINT_METRIC_KERNEL_G6'):
        return ISLAND_PARAMS_MATCHED_JOINT_G6
    if os.environ.get('R2_MATCHED_JOINT_METRIC_KERNEL_MAP'):
        return ISLAND_PARAMS_MATCHED_JOINT_MAP
    if os.environ.get('R2_MATCHED_JOINT_METRIC_KERNEL_EXTENDED'):
        return ISLAND_PARAMS_MATCHED_JOINT_EXTENDED
    if os.environ.get('R2_MATCHED_JOINT_METRIC_KERNEL'):
        return ISLAND_PARAMS_MATCHED_JOINT_METRIC
    if os.environ.get('R2_MATCHED_KERNEL_CORE_NOG10'):
        return ISLAND_PARAMS_MATCHED_CORE_NOG10
    if os.environ.get('R2_MATCHED_KERNEL_CORE'):
        return ISLAND_PARAMS_MATCHED_CORE
    if os.environ.get('R2_SHADOW_PAS2D_PROFILE_V5_ULTRA'):
        return ISLAND_PARAMS_SHADOW_PAS2D_V5_ULTRA
    if os.environ.get('R2_SHADOW_PAS2D_PROFILE_V5_CONSERVATIVE'):
        return ISLAND_PARAMS_SHADOW_PAS2D_V5_CONSERVATIVE
    if os.environ.get('R2_SHADOW_PAS2D_PROFILE_V5_EXTENDED'):
        return ISLAND_PARAMS_SHADOW_PAS2D_V5_EXTENDED
    if os.environ.get('R2_SHADOW_PAS2D_PROFILE_V5'):
        return ISLAND_PARAMS_SHADOW_PAS2D_V5
    if os.environ.get('R2_SHADOW_KERNEL_PROFILE'):
        return ISLAND_PARAMS_SHADOW_ROBUST
    return ISLAND_PARAMS_DENSITY if os.environ.get('R2_DENSITY_PARAMS') else ISLAND_PARAMS


def island_params_for(group, default):
    pas, pdp = active_island_params().get(int(group), default)
    pas_override = os.environ.get('R2_KERNEL_PAS_OVERRIDE')
    pdp_override = os.environ.get('R2_KERNEL_PDP_OVERRIDE')
    if pas_override:
        k, power, q = pas_override.split(':')
        pas = (int(k), float(power), float(q))
    if pdp_override:
        k, power, q = pdp_override.split(':')
        pdp = (int(k), float(power), float(q))
    return pas, pdp


AFFINE_PROFILE={
 0:{'pas':(0.,1.),'pdp':(0.,1.)},1:{'pas':(1.,1.),'pdp':(.5,.01)},
 3:{'pas':(1.,.01),'pdp':(1.,.01)},4:{'pas':(1.,.01),'pdp':(0.,1.)},
 5:{'pas':(1.,1.),'pdp':(0.,1.)},6:{'pas':(.5,.01),'pdp':(1.,.01)},
 7:{'pas':(0.,1.),'pdp':(1.,1.)},8:{'pas':(1.,.1),'pdp':(1.,.01)},
 10:{'pas':(.5,.01),'pdp':(1.,.01)},
}
AFFINE_PROFILE_SHADOW_SAFE={**AFFINE_PROFILE,
 0:{**AFFINE_PROFILE[0],'pas':(1.,.1)},
 1:{**AFFINE_PROFILE[1],'pas':(0.,1.)},
 9:{**AFFINE_PROFILE.get(9,{}),'pdp':(1.,.001)},
}
AFFINE_PROFILE_SHADOW_EXTENDED={**AFFINE_PROFILE_SHADOW_SAFE,
 4:{**AFFINE_PROFILE[4],'pdp':(.25,.001)},
 7:{**AFFINE_PROFILE[7],'pdp':(.25,.1)},
 8:{**AFFINE_PROFILE[8],'pas':(.75,.1),'pdp':(.25,1.)},
 10:{**AFFINE_PROFILE[10],'pas':(1.,1.)},
}
AFFINE_PROFILE_SHADOW_CONSERVATIVE={**AFFINE_PROFILE_SHADOW_SAFE,
 4:{**AFFINE_PROFILE[4],'pdp':(.25,.001)},
 10:{**AFFINE_PROFILE[10],'pas':(1.,1.)},
}
AFFINE_PROFILE_SHADOW_NOG7={**AFFINE_PROFILE_SHADOW_CONSERVATIVE,
 8:{**AFFINE_PROFILE[8],'pas':(.75,.1),'pdp':(.25,1.)},
}
AFFINE_PROFILE_MATCHED_PHASE4={**AFFINE_PROFILE_SHADOW_NOG7,
 1:{**AFFINE_PROFILE_SHADOW_NOG7[1],'pas':(.25,.1)},
}

def active_affine_profile():
    if os.environ.get('R2_MATCHED_PHASE4'):
        return AFFINE_PROFILE_MATCHED_PHASE4
    if os.environ.get('R2_SHADOW_CORRECTION_EXTENDED'):
        return AFFINE_PROFILE_SHADOW_EXTENDED
    if os.environ.get('R2_SHADOW_CORRECTION_NOG7'):
        return AFFINE_PROFILE_SHADOW_NOG7
    if os.environ.get('R2_SHADOW_CORRECTION_CONSERVATIVE'):
        return AFFINE_PROFILE_SHADOW_CONSERVATIVE
    if os.environ.get('R2_SHADOW_CORRECTION_SAFE'):
        return AFFINE_PROFILE_SHADOW_SAFE
    return AFFINE_PROFILE


def affine_params_for(group, axis, default=(.5,.01)):
    override=os.environ.get(f'R2_AFFINE_{axis.upper()}_OVERRIDE')
    if override:
        beta,ridge=override.split(':')
        return float(beta),float(ridge)
    return active_affine_profile().get(int(group),{}).get(axis,default)
PROJECTION_ITERS_PROFILE={0:4,1:20,3:12,4:2,5:20,6:6,7:20,8:20,9:8,10:20}
PROJECTION_ITERS_SHADOW_V2={**PROJECTION_ITERS_PROFILE,0:12,4:12}
PROJECTION_ITERS_SHADOW_V3={**PROJECTION_ITERS_SHADOW_V2,3:30,6:12,10:8}
PROJECTION_ITERS_MATCHED_PHASE={**PROJECTION_ITERS_SHADOW_V3,4:20,5:30}

def projection_iters_for(group):
    if os.environ.get('R2_PROJECTION_ITERS') is not None:
        return int(os.environ['R2_PROJECTION_ITERS'])
    if os.environ.get('R2_MATCHED_PHASE_SAFE'):
        profile=PROJECTION_ITERS_MATCHED_PHASE
    elif os.environ.get('R2_SHADOW_ITERATION_PROFILE_V3'):
        profile=PROJECTION_ITERS_SHADOW_V3
    elif os.environ.get('R2_SHADOW_ITERATION_PROFILE'):
        profile=PROJECTION_ITERS_SHADOW_V2
    else:
        profile=PROJECTION_ITERS_PROFILE
    return profile.get(int(group),8) if os.environ.get('R2_PROJECTION_ITERS_PROFILE') else 20
FINAL_PAS_PROFILE={0:.6,1:.8,3:1.,4:1.,5:.8,6:.2,7:0.,8:1.,9:.6,10:.8}
FINAL_PAS_PROFILE_MATCHED_PHASE={**FINAL_PAS_PROFILE,3:.8}

def final_pas_blend_for(group):
    if os.environ.get('R2_FINAL_PAS_OVERRIDE') is not None:
        return float(os.environ['R2_FINAL_PAS_OVERRIDE'])
    profile=(FINAL_PAS_PROFILE_MATCHED_PHASE
             if os.environ.get('R2_MATCHED_PHASE_SAFE') else FINAL_PAS_PROFILE)
    return profile.get(int(group),.6)
PAS_LOW_RANK_PROFILE={4:(1,.25),6:(1,.25)}
PAS_MOMENT_ALIGN_PROFILE={4:.35,7:.5}
PAS_POOL_PROFILE={0:('both',.55),1:('sc',1.),5:('both',.7),6:('both',.55),7:('both',.7),8:('ue',.85),10:('both',.7)}
PDP_POOL_PROFILE={0:('hv',1.),1:('ue',1.),3:('hv',1.),4:('hv',1.),5:('hv',.85),6:('hv',.85),7:('both',1.),8:('hv',.7),10:('both',.85)}
PAS_SMOOTH_PROFILE={0:('hv3',.2),1:('v3',.3),3:('h3',.1),5:('h3',.2),6:('h3',.55),8:('v3',.3),10:('v3',.2)}
PDP_SMOOTH_PROFILE={3:('d5',.05),4:('d3',.3),5:('d5',.05),6:('d5',.3),7:('d3',.3),8:('d5',.1),10:('d5',.05)}
PDP_XCORR_PROFILE={0:1.,1:.4,3:1.,5:.3}

# Retuned only on the five mutually-disjoint rectangular shadow folds.  These
# profiles deliberately omit candidates whose gains reversed on held-out
# rectangles, even when their tuning-fold gain was much larger.
PAS_POOL_PROFILE_SHADOW_V4={
 0:('sc',1.),1:('both',1.),3:('both',.25),4:('sc',1.),
 6:('sc',1.),8:('sc',1.),9:('both',.25),
}
PAS_SMOOTH_PROFILE_SHADOW_V4={1:('hv3',.5),5:('v3',.2),8:('v3',.1)}
PDP_POOL_PROFILE_SHADOW_V4={
 0:('hv',1.),1:('both',.85),3:('hv',1.),4:('hv',.85),
 5:('hv',.85),6:('hv',.85),8:('hv',.85),9:('hv',1.),10:('both',.85),
}
PDP_POOL_PROFILE_SHADOW_V4_G7={**PDP_POOL_PROFILE_SHADOW_V4,7:('hv',.85)}
PDP_SMOOTH_PROFILE_SHADOW_V4={3:('d5',.05),4:('d5',.05),6:('d5',.05)}
PAS_POOL_PROFILE_MATCHED_SAFE={
 g:value for g,value in PAS_POOL_PROFILE_SHADOW_V4.items()
 if g in {0,4,6,8,9}
}
PAS_POOL_PROFILE_MATCHED_PHASE={
 **PAS_POOL_PROFILE_MATCHED_SAFE,
 5:('sc',1.),7:('both',.5),
}
PAS_SMOOTH_PROFILE_MATCHED_SAFE={
  g:value for g,value in PAS_SMOOTH_PROFILE_SHADOW_V4.items()
  if g in {8}
}
PDP_POOL_PROFILE_MATCHED_SAFE={
 **{g:value for g,value in PDP_POOL_PROFILE.items()
    if g in {0,1,3,4,5,6,8,10}},
 **{g:value for g,value in PDP_POOL_PROFILE_SHADOW_V4_G7.items()
    if g in {0,4,6,7,8,9}},
}
PDP_SMOOTH_PROFILE_MATCHED_SAFE={
  g:value for g,value in PDP_SMOOTH_PROFILE_SHADOW_V4.items()
  if g in {4,6}
}
PAS_POOL_PROFILE_MATCHED_PHASE4={
 **PAS_POOL_PROFILE_MATCHED_PHASE,
 1:('both',.25),
}
PAS_SMOOTH_PROFILE_MATCHED_PHASE4={
 **PAS_SMOOTH_PROFILE_MATCHED_SAFE,
 1:('hv3',.25),5:('hv3',.10),
}
PDP_POOL_PROFILE_MATCHED_PHASE4={
 **PDP_POOL_PROFILE_MATCHED_SAFE,
 1:('both',.70),
}
PDP_SMOOTH_PROFILE_MATCHED_PHASE4={
 **PDP_SMOOTH_PROFILE_MATCHED_SAFE,
 7:('d5',.20),
}


def active_pas_pool_profile():
    if os.environ.get('R2_MATCHED_PHASE4'):
        return PAS_POOL_PROFILE_MATCHED_PHASE4
    if os.environ.get('R2_MATCHED_PHASE_SAFE'):
        return PAS_POOL_PROFILE_MATCHED_PHASE
    if os.environ.get('R2_MATCHED_TARGET_SAFE'):
        return PAS_POOL_PROFILE_MATCHED_SAFE
    if os.environ.get('R2_SHADOW_TARGET_PROFILE_V4'):
        return PAS_POOL_PROFILE_SHADOW_V4
    return PAS_POOL_PROFILE


def active_pas_smooth_profile():
    if os.environ.get('R2_MATCHED_PHASE4'):
        return PAS_SMOOTH_PROFILE_MATCHED_PHASE4
    if os.environ.get('R2_MATCHED_TARGET_SAFE'):
        return PAS_SMOOTH_PROFILE_MATCHED_SAFE
    if os.environ.get('R2_SHADOW_TARGET_PROFILE_V4'):
        return PAS_SMOOTH_PROFILE_SHADOW_V4
    return PAS_SMOOTH_PROFILE


def active_pdp_pool_profile():
    if os.environ.get('R2_MATCHED_PHASE4'):
        return PDP_POOL_PROFILE_MATCHED_PHASE4
    if os.environ.get('R2_MATCHED_TARGET_SAFE'):
        return PDP_POOL_PROFILE_MATCHED_SAFE
    if os.environ.get('R2_SHADOW_TARGET_PROFILE_V4_G7'):
        return PDP_POOL_PROFILE_SHADOW_V4_G7
    if os.environ.get('R2_SHADOW_TARGET_PROFILE_V4'):
        return PDP_POOL_PROFILE_SHADOW_V4
    return PDP_POOL_PROFILE


def active_pdp_smooth_profile():
    if os.environ.get('R2_MATCHED_PHASE4'):
        return PDP_SMOOTH_PROFILE_MATCHED_PHASE4
    if os.environ.get('R2_MATCHED_TARGET_SAFE'):
        return PDP_SMOOTH_PROFILE_MATCHED_SAFE
    if os.environ.get('R2_SHADOW_TARGET_PROFILE_V4'):
        return PDP_SMOOTH_PROFILE_SHADOW_V4
    return PDP_SMOOTH_PROFILE


def aggregate_pas_moments(channel):
    """Global H/V first circular moments for integer angular-spectrum alignment."""
    import torch
    power=torch.abs(bs_fft_torch(channel))**2
    maps=power.reshape(*power.shape[:-3],BS_P,BS_H,BS_V,4,192).sum((-4,-3,-2,-1))
    hp=torch.exp(2j*torch.pi*torch.arange(BS_H,device=channel.device)/BS_H)
    vp=torch.exp(2j*torch.pi*torch.arange(BS_V,device=channel.device)/BS_V)
    mh=(maps.sum(-1)*hp).sum(-1)/maps.sum((-2,-1)).clamp_min(1e-30)
    mv=(maps.sum(-2)*vp).sum(-1)/maps.sum((-2,-1)).clamp_min(1e-30)
    return mh,mv


def align_pas_neighbors(spectra, weights, group):
    """Align neighbor spectra to their weighted circular-moment consensus."""
    if group not in PAS_MOMENT_ALIGN_PROFILE:
        return spectra
    import torch
    batch,k=spectra.shape[:2]
    shaped=spectra.reshape(batch,k,BS_P,BS_H,BS_V,4,192)
    maps=shaped.sum((2,5,6))
    hp=torch.exp(2j*torch.pi*torch.arange(BS_H,device=spectra.device)/BS_H)
    vp=torch.exp(2j*torch.pi*torch.arange(BS_V,device=spectra.device)/BS_V)
    mh=(maps.sum(-1)*hp).sum(-1)/maps.sum((-2,-1)).clamp_min(1e-30)
    mv=(maps.sum(-2)*vp).sum(-1)/maps.sum((-2,-1)).clamp_min(1e-30)
    th=(weights*mh).sum(1);tv=(weights*mv).sum(1)
    dh=torch.round(torch.angle(th[:,None]*torch.conj(mh))*BS_H/(2*torch.pi)).to(torch.int64)
    dv=torch.round(torch.angle(tv[:,None]*torch.conj(mv))*BS_V/(2*torch.pi)).to(torch.int64)
    aligned=torch.empty_like(shaped)
    for i in range(batch):
        for j in range(k):
            aligned[i,j]=torch.roll(shaped[i,j],(int(dh[i,j]),int(dv[i,j])),dims=(1,2))
    blend=PAS_MOMENT_ALIGN_PROFILE[group]
    return ((1-blend)*shaped+blend*aligned).reshape_as(spectra)


PAS_XCORR_PROFILE={4:('weighted_map',.7),5:('medoid',.55),6:('medoid',.1),7:('nearest',1.),10:('nearest',.4)}
PAS_LOCAL_SHIFT_PROFILE={0:(4,10.,-.5),4:(4,10.,-.5),8:(8,1.,-1.),10:(4,10.,.5)}


def align_pas_neighbors_xcorr(spectra, weights, group):
    """Partially align complete 2D angular maps to their weighted consensus."""
    if group not in PAS_XCORR_PROFILE:
        return spectra
    import torch
    batch,k=spectra.shape[:2]
    shaped=spectra.reshape(batch,k,BS_P,BS_H,BS_V,4,192)
    maps=shaped.sum((2,5,6))
    ref_mode,blend=PAS_XCORR_PROFILE[group]
    if ref_mode=='nearest':
        ref=maps[:,0]
    elif ref_mode=='medoid':
        flat=maps.flatten(2);flat=flat/torch.linalg.vector_norm(flat,dim=2,keepdim=True).clamp_min(1e-30)
        central=((flat@flat.transpose(1,2))*weights[:,None,:]).sum(2)
        ref=maps[torch.arange(batch,device=maps.device),central.argmax(1)]
    else:
        ref=(maps*weights[:,:,None,None]).sum(1)
    corr=torch.fft.ifft2(torch.fft.fft2(ref,dim=(-2,-1))[:,None]
                         *torch.conj(torch.fft.fft2(maps,dim=(-2,-1))),dim=(-2,-1)).real
    arg=corr.flatten(2).argmax(2);dh=arg//BS_V;dv=arg%BS_V
    aligned=torch.empty_like(shaped)
    for i in range(batch):
        for j in range(k):
            aligned[i,j]=torch.roll(shaped[i,j],(int(dh[i,j]),int(dv[i,j])),dims=(1,2))
    return ((1-blend)*shaped+blend*aligned).reshape_as(spectra)


def local_shift_pas_target(target, neighbor_spectra, neighbor_ids, query_xy, group):
    """Extrapolate the local angular-spectrum displacement to the query coordinate."""
    if group not in PAS_LOCAL_SHIFT_PROFILE:
        return target
    import torch
    count,ridge,scale=PAS_LOCAL_SHIFT_PROFILE[group]
    count=min(count,neighbor_spectra.shape[1])
    raw=neighbor_spectra.reshape(len(target),-1,BS_P,BS_H,BS_V,4,192)
    maps=raw[:,:count].sum((2,5,6));ref=maps[:,0]
    corr=torch.fft.ifft2(torch.fft.fft2(ref,dim=(-2,-1))[:,None]
                         *torch.conj(torch.fft.fft2(maps,dim=(-2,-1))),dim=(-2,-1)).real
    arg=corr.flatten(2).argmax(2).cpu().numpy();shift=np.stack((arg//BS_V,arg%BS_V),2).astype(float)
    shift[:,:,0]=(shift[:,:,0]+BS_H//2)%BS_H-BS_H//2
    shift[:,:,1]=(shift[:,:,1]+BS_V//2)%BS_V-BS_V//2
    train_pos=np.load(ROOT/'Round2_Train_Pos.npy',mmap_mode='r')
    shaped=target.reshape(len(target),BS_P,BS_H,BS_V,4,192);out=torch.empty_like(shaped)
    for i in range(len(target)):
        xy=np.asarray(train_pos[neighbor_ids[i,:count],:2]);x=xy-xy[0];z=np.c_[np.ones(count),x]
        coef=np.linalg.solve(z.T@z+np.diag([0.,ridge,ridge]),z.T@shift[i])
        pred=np.r_[1.,query_xy[i]-xy[0]]@coef
        out[i]=torch.roll(shaped[i],(int(np.rint(scale*pred[0])),int(np.rint(scale*pred[1]))),dims=(1,2))
    return out.reshape_as(target)


def low_rank_pas_target(target, group):
    """Shrink unpredictable H/V joint detail while preserving strong marginals."""
    if group not in PAS_LOW_RANK_PROFILE:
        return target
    import torch
    rank, blend = PAS_LOW_RANK_PROFILE[group]
    matrix = target.reshape(len(target),BS_P,BS_H,BS_V,4,192).permute(0,1,4,5,2,3)
    flat = matrix.reshape(-1,BS_H,BS_V)
    u,s,vh = torch.linalg.svd(flat,full_matrices=False)
    low = ((u[:,:,:rank]*s[:,None,:rank])@vh[:,:rank]).clamp_min(0)
    low = low.reshape(len(target),BS_P,4,192,BS_H,BS_V).permute(0,1,4,5,2,3).reshape_as(target)
    return (1-blend)*target+blend*low


def pool_pas_target(target, group):
    """Shrink noisy UE/subcarrier detail toward a shared 2D angular template."""
    profile=active_pas_pool_profile()
    override=os.environ.get('R2_PAS_POOL_OVERRIDE')
    if override:
        if override=='none': return target
        mode,blend=override.split(':');profile={int(group):(mode,float(blend))}
    if group not in profile:
        return target
    import torch
    mode,blend=profile[group]
    shaped=target.reshape(len(target),BS_P,BS_H,BS_V,4,192)
    if mode=='ue':
        common=shaped.mean(4,keepdim=True).expand_as(shaped)
    elif mode=='ue_med':
        common=shaped.median(4,keepdim=True).values.expand_as(shaped)
    elif mode=='sc':
        common=shaped.mean(5,keepdim=True).expand_as(shaped)
    elif mode=='sc_med':
        common=shaped.median(5,keepdim=True).values.expand_as(shaped)
    elif mode=='both_med':
        common=shaped.flatten(4,5).median(4,keepdim=True).values[...,None].expand_as(shaped)
    elif mode=='add':
        common=(shaped.mean(4,keepdim=True)+shaped.mean(5,keepdim=True)
                -shaped.mean((4,5),keepdim=True)).clamp_min(1e-12).expand_as(shaped)
    else:
        common=shaped.mean((4,5),keepdim=True).expand_as(shaped)
    common=common/torch.linalg.vector_norm(common,dim=(2,3),keepdim=True).clamp_min(1e-30)
    norms=torch.linalg.vector_norm(shaped,dim=(2,3),keepdim=True)
    pooled=(common*norms).reshape_as(target)
    return (1-blend)*target+blend*pooled


def smooth_pas_target(target, group):
    """Denoise the 2D angular target with group-specific circular smoothing."""
    profile=active_pas_smooth_profile()
    override=os.environ.get('R2_PAS_SMOOTH_OVERRIDE')
    if override:
        if override=='none': return target
        mode,blend=override.split(':');profile={int(group):(mode,float(blend))}
    if group not in profile:
        return target
    import torch
    mode,blend=profile[group]
    shaped=target.reshape(len(target),BS_P,BS_H,BS_V,4,192)
    if mode=='h3':
        smooth=(torch.roll(shaped,1,2)+2*shaped+torch.roll(shaped,-1,2))/4
    elif mode=='v3':
        smooth=(torch.roll(shaped,1,3)+2*shaped+torch.roll(shaped,-1,3))/4
    else:
        smooth=(
            torch.roll(torch.roll(shaped,1,2),1,3)
            +2*torch.roll(shaped,1,2)
            +torch.roll(torch.roll(shaped,1,2),-1,3)
            +2*torch.roll(shaped,1,3)
            +4*shaped
            +2*torch.roll(shaped,-1,3)
            +torch.roll(torch.roll(shaped,-1,2),1,3)
            +2*torch.roll(shaped,-1,2)
            +torch.roll(torch.roll(shaped,-1,2),-1,3)
        )/16
    norms=torch.linalg.vector_norm(shaped,dim=(2,3),keepdim=True)
    smooth*=norms/torch.linalg.vector_norm(smooth,dim=(2,3),keepdim=True).clamp_min(1e-30)
    return ((1-blend)*shaped+blend*smooth).reshape_as(target)


def pool_pdp_target(target, group):
    """Share stable delay profiles across antenna/UE axes while retaining slice norms."""
    enabled_groups = os.environ.get('R2_PDP_POOL_GROUPS')
    if enabled_groups:
        enabled_groups = {int(value) for value in enabled_groups.split(',') if value.strip()}
        if group not in enabled_groups:
            return target
    profile=active_pdp_pool_profile()
    override=os.environ.get('R2_PDP_POOL_OVERRIDE')
    if override:
        if override=='none': return target
        mode,blend=override.split(':');profile={int(group):(mode,float(blend))}
    if group not in profile:
        return target
    import torch
    mode,blend=profile[group]
    shaped=target.reshape(len(target),BS_P,BS_H,BS_V,4,192)
    norms=torch.linalg.vector_norm(shaped,dim=5,keepdim=True)
    if mode=='hv':
        common=shaped.mean((2,3),keepdim=True).expand_as(shaped)
    elif mode=='hv_med':
        common=shaped.flatten(2,3).median(2).values[:,:,None,None,:,:].expand_as(shaped)
    elif mode=='ue':
        common=shaped.mean(4,keepdim=True).expand_as(shaped)
    elif mode=='ue_med':
        common=shaped.median(4,keepdim=True).values.expand_as(shaped)
    elif mode=='both_med':
        common=(shaped.permute(0,1,5,2,3,4).flatten(3,5).median(3).values
                [:,:,None,None,None,:].expand_as(shaped))
    elif mode=='add':
        common=(shaped.mean((2,3),keepdim=True)+shaped.mean(4,keepdim=True)
                -shaped.mean((2,3,4),keepdim=True)).clamp_min(1e-12).expand_as(shaped)
    else:
        common=shaped.mean((1,2,3,4),keepdim=True).expand_as(shaped)
    common=common/torch.linalg.vector_norm(common,dim=5,keepdim=True).clamp_min(1e-30)
    pooled=(common*norms).reshape_as(target)
    return (1-blend)*target+blend*pooled


def smooth_pdp_target(target, group):
    """Denoise delay-bin detail while preserving every antenna/UE slice norm."""
    profile=active_pdp_smooth_profile()
    override=os.environ.get('R2_PDP_SMOOTH_OVERRIDE')
    if override:
        if override=='none': return target
        mode,blend=override.split(':');profile={int(group):(mode,float(blend))}
    if group not in profile:
        return target
    import torch
    kernel,blend=profile[group]
    if kernel=='d3':
        smooth=(torch.roll(target,1,-1)+2*target+torch.roll(target,-1,-1))/4
    else:
        smooth=(torch.roll(target,2,-1)+4*torch.roll(target,1,-1)+6*target
                +4*torch.roll(target,-1,-1)+torch.roll(target,-2,-1))/16
    smooth[...,0]=(2*target[...,0]+target[...,1])/3
    smooth[...,-1]=(2*target[...,-1]+target[...,-2])/3
    norms=torch.linalg.vector_norm(target,dim=-1,keepdim=True)
    smooth*=norms/torch.linalg.vector_norm(smooth,dim=-1,keepdim=True).clamp_min(1e-30)
    return (1-blend)*target+blend*smooth


def align_pdp_neighbors_xcorr(spectra, weights, group):
    """Align common path-delay motion before combining neighbor PDPs."""
    if group not in PDP_XCORR_PROFILE:
        return spectra
    import torch
    profile=spectra.sum((2,3));ref=(profile*weights[:,:,None]).sum(1)
    corr=torch.fft.ifft(torch.fft.fft(ref,dim=-1)[:,None]
                        *torch.conj(torch.fft.fft(profile,dim=-1)),dim=-1).real
    shift=corr.argmax(-1);aligned=torch.empty_like(spectra)
    for i in range(len(spectra)):
        for j in range(spectra.shape[1]):
            aligned[i,j]=torch.roll(spectra[i,j],int(shift[i,j]),dims=-1)
    blend=PDP_XCORR_PROFILE[group]
    return (1-blend)*spectra+blend*aligned

def affine_blend_weights(query_xy, neighbor_xy, dist, power, weights, beta=.5, ridge=.01):
    """Local-linear spatial correction blended with positive kernel weights."""
    if not (os.environ.get('R2_AFFINE_BLEND') or os.environ.get('R2_AFFINE_PROFILE')):
        return weights
    if os.environ.get('R2_AFFINE_BLEND'): beta=float(os.environ['R2_AFFINE_BLEND'])
    if beta==0:return weights
    out=np.empty_like(weights)
    for i in range(len(weights)):
        x=neighbor_xy[i]-query_xy[i];z=np.c_[np.ones(len(x)),x];w=weights[i]
        a=z.T@(w[:,None]*z);a[1:,1:]+=np.eye(2)*ridge*np.sum(w*np.sum(x*x,axis=1))/2
        try: aw=np.array([1.,0.,0])@np.linalg.solve(a,z.T*w)
        except np.linalg.LinAlgError: aw=w
        out[i]=(1-beta)*w+beta*aw
    return out


QUADRATIC_PROFILE={
 1:{'pas':(.75,.1)},8:{'pdp':(1.,.01)},
}
QUADRATIC_PROFILE_SHADOW_SAFE={8:{'pdp':(1.,.01)}}
QUADRATIC_PROFILE_SHADOW_EXTENDED={7:{'pas':(1.,.1)}}
QUADRATIC_PROFILE_SHADOW_CONSERVATIVE={8:{'pdp':(1.,.01)}}
QUADRATIC_PROFILE_SHADOW_NOG7={}

def active_quadratic_profile():
    if os.environ.get('R2_SHADOW_CORRECTION_EXTENDED'):
        return QUADRATIC_PROFILE_SHADOW_EXTENDED
    if os.environ.get('R2_SHADOW_CORRECTION_NOG7'):
        return QUADRATIC_PROFILE_SHADOW_NOG7
    if os.environ.get('R2_SHADOW_CORRECTION_CONSERVATIVE'):
        return QUADRATIC_PROFILE_SHADOW_CONSERVATIVE
    if os.environ.get('R2_SHADOW_CORRECTION_SAFE'):
        return QUADRATIC_PROFILE_SHADOW_SAFE
    return QUADRATIC_PROFILE
def quadratic_blend_weights(query_xy,neighbor_xy,dist,power,weights,beta,ridge):
    if not os.environ.get('R2_QUADRATIC_PROFILE'):return weights
    out=np.empty_like(weights)
    for i in range(len(weights)):
        x=neighbor_xy[i]-query_xy[i];sc=max(np.median(dist[i]),1.);u=x/sc
        z=np.c_[np.ones(len(x)),u[:,0],u[:,1],u[:,0]**2,u[:,0]*u[:,1],u[:,1]**2];w=weights[i];a=z.T@(w[:,None]*z);a[1:,1:]+=np.eye(5)*ridge
        try:aw=np.array([1.,0,0,0,0,0])@np.linalg.solve(a,z.T*w)
        except np.linalg.LinAlgError:aw=w
        out[i]=(1-beta)*w+beta*aw
    return out
NEAREST_INIT_ISLANDS={0,1,2,3,4,5,6,7,8,9,10}
INIT_RANK_PROFILE={}
INIT_RANK_PROFILE_MATCHED_PHASE={4:5,6:2,9:2}


def init_rank_for(group):
    profile=(INIT_RANK_PROFILE_MATCHED_PHASE
             if os.environ.get('R2_MATCHED_PHASE_SAFE') else INIT_RANK_PROFILE)
    return profile.get(int(group),1)


def g4_feature_neighbor_plan(train, query_xy, query_feature_rows, k=32):
    """G4 neighbors in coordinate + map-ray-obstruction feature space."""
    from scipy.spatial import cKDTree
    train_pos=np.load(ROOT/'Round2_Train_Pos.npy')
    all_los=np.load(ROOT/'los_map_features.npy')[:,[0,1,3,4,5,7]]
    train_los=all_los[:len(train_pos)]
    valid=np.flatnonzero(np.load(ROOT/'train_energy.npy',mmap_mode='r')>0)
    xy_med=np.median(train_pos[valid,:2],axis=0);xy_iqr=np.quantile(train_pos[valid,:2],.75,axis=0)-np.quantile(train_pos[valid,:2],.25,axis=0)
    lo_med=np.median(train_los[valid],axis=0);lo_iqr=np.quantile(train_los[valid],.75,axis=0)-np.quantile(train_los[valid],.25,axis=0)
    z_train=np.c_[(train_pos[train,:2]-xy_med)/np.maximum(xy_iqr,1e-6),
                  .25*(train_los[train]-lo_med)/np.maximum(lo_iqr,1e-6)]
    z_query=np.c_[(query_xy-xy_med)/np.maximum(xy_iqr,1e-6),
                  .25*(all_los[query_feature_rows]-lo_med)/np.maximum(lo_iqr,1e-6)]
    dist,loc=cKDTree(z_train).query(z_query,k=k)
    return dist,train[loc]


MATCHED_FEATURE_METRIC_G1={1:{'pdp':(75.,.4)}}
MATCHED_FEATURE_METRIC_G8={8:{'pas':(75.,.85)}}
MATCHED_FEATURE_METRIC_CORE={
 **MATCHED_FEATURE_METRIC_G1,
 **MATCHED_FEATURE_METRIC_G8,
}
MATCHED_FEATURE_METRIC_EXTENDED={**MATCHED_FEATURE_METRIC_CORE,
 0:{'pdp':(90.,.85)},
 5:{'pdp':(150.,1.45)},
}
MATCHED_FEATURE_METRIC_MAP={**MATCHED_FEATURE_METRIC_EXTENDED,
 7:{'pdp':('map',4.)},
}
MATCHED_FEATURE_METRIC_G6={**MATCHED_FEATURE_METRIC_MAP,
 6:{'pas':('map',4.)},
}


def active_feature_metric_profile():
    if os.environ.get('R2_MATCHED_JOINT_METRIC_KERNEL_G6'):
        return MATCHED_FEATURE_METRIC_G6
    if os.environ.get('R2_MATCHED_JOINT_METRIC_KERNEL_MAP'):
        return MATCHED_FEATURE_METRIC_MAP
    if os.environ.get('R2_MATCHED_JOINT_METRIC_KERNEL_EXTENDED'):
        return MATCHED_FEATURE_METRIC_EXTENDED
    if os.environ.get('R2_MATCHED_JOINT_METRIC_KERNEL'):
        return MATCHED_FEATURE_METRIC_CORE
    if os.environ.get('R2_MATCHED_FEATURE_METRIC_CORE'):
        return MATCHED_FEATURE_METRIC_CORE
    if os.environ.get('R2_MATCHED_FEATURE_METRIC_G1'):
        return MATCHED_FEATURE_METRIC_G1
    if os.environ.get('R2_MATCHED_FEATURE_METRIC_G8'):
        return MATCHED_FEATURE_METRIC_G8
    return {}


def cartesian_metric_neighbor_plan(train,query_xy,angle,ratio,k):
    """KNN after a fixed rotated anisotropic coordinate transform."""
    from scipy.spatial import cKDTree
    train_pos=np.load(ROOT/'Round2_Train_Pos.npy')
    theta=np.deg2rad(angle);u=np.array([np.cos(theta),np.sin(theta)])
    v=np.array([-np.sin(theta),np.cos(theta)])
    train_feature=np.c_[train_pos[train,:2]@u/ratio,train_pos[train,:2]@v]
    query_feature=np.c_[query_xy@u/ratio,query_xy@v]
    dist,loc=cKDTree(train_feature).query(query_feature,k=k)
    return np.atleast_2d(dist),train[np.atleast_2d(loc)]


def configured_metric_neighbor_plan(train,query_xy,query_feature_rows,side,spec,k):
    if isinstance(spec[0],str) and spec[0]=='map':
        from scipy.spatial import cKDTree
        train_pos=np.load(ROOT/'Round2_Train_Pos.npy')
        all_features=np.load(ROOT/'los_map_features.npy')
        train_features=all_features[:len(train_pos)]
        valid=np.flatnonzero(np.load(ROOT/'train_energy.npy',mmap_mode='r')>0)
        center=np.median(train_features[valid],axis=0)
        scale=np.quantile(train_features[valid],.75,axis=0)-np.quantile(train_features[valid],.25,axis=0)
        scale=np.maximum(scale,np.array([3.,.03,2.,3.]*2))
        columns=slice(4,8) if side else slice(0,4);factor=float(spec[1])
        z_train=np.c_[train_pos[train,:2],factor*(train_features[train,columns]-center[columns])/scale[columns]]
        z_query=np.c_[query_xy,factor*(all_features[query_feature_rows,columns]-center[columns])/scale[columns]]
        dist,loc=cKDTree(z_train).query(z_query,k=k)
        return np.atleast_2d(dist),train[np.atleast_2d(loc)]
    return cartesian_metric_neighbor_plan(train,query_xy,float(spec[0]),float(spec[1]),k)


def harmonic_neighbor_plan(train,query_xy,query_feature_rows,side,spec,topk):
    """Transductive graph-harmonic weights for a rectangular query-point cluster."""
    from scipy.spatial import cKDTree
    train_pos=np.load(ROOT/'Round2_Train_Pos.npy')
    if spec is not None and isinstance(spec[0],str) and spec[0]=='map':
        all_features=np.load(ROOT/'los_map_features.npy')
        valid=np.flatnonzero(np.load(ROOT/'train_energy.npy',mmap_mode='r')>0)
        center=np.median(all_features[valid],axis=0)
        scale=np.quantile(all_features[valid],.75,axis=0)-np.quantile(all_features[valid],.25,axis=0)
        scale=np.maximum(scale,np.array([3.,.03,2.,3.]*2))
        columns=slice(4,8) if side else slice(0,4);factor=float(spec[1])
        train_feature=np.c_[train_pos[train,:2],factor*(all_features[train,columns]-center[columns])/scale[columns]]
        query_feature=np.c_[query_xy,factor*(all_features[query_feature_rows,columns]-center[columns])/scale[columns]]
    elif spec is not None:
        angle,ratio=float(spec[0]),float(spec[1]);theta=np.deg2rad(angle)
        u=np.array([np.cos(theta),np.sin(theta)]);v=np.array([-np.sin(theta),np.cos(theta)])
        train_feature=np.c_[train_pos[train,:2]@u/ratio,train_pos[train,:2]@v]
        query_feature=np.c_[query_xy@u/ratio,query_xy@v]
    else:
        train_feature=train_pos[train,:2]
        query_feature=query_xy
    boundary_k=min(int(os.environ.get('R2_HARMONIC_BOUNDARY_K','128')),len(train))
    _,nearest=cKDTree(train_feature).query(query_feature,k=boundary_k)
    nearest=np.asarray(nearest)
    if nearest.ndim==1:
        nearest=nearest[:,None]
    boundary_local=np.unique(nearest.reshape(-1))
    boundary_feature=train_feature[boundary_local]
    combined=np.r_[query_feature,boundary_feature]
    graph_k=min(int(os.environ.get('R2_HARMONIC_GRAPH_K','16')),len(combined)-1)
    distance,neighbor=cKDTree(combined).query(query_feature,k=graph_k+1)
    distance=np.asarray(distance)[:,1:];neighbor=np.asarray(neighbor)[:,1:]
    local_scale=np.maximum(distance[:,-1],1e-6)*float(os.environ.get('R2_HARMONIC_BANDWIDTH','1.0'))
    edge=np.exp(-.5*(distance/local_scale[:,None])**2)
    nq=len(query_feature);nb=len(boundary_feature)
    wqq=np.zeros((nq,nq),np.float64);wqb=np.zeros((nq,nb),np.float64)
    query_scale=float(os.environ.get('R2_HARMONIC_QUERY_SCALE','1.0'))
    for row in range(nq):
        for column,value in zip(neighbor[row],edge[row]):
            if column<nq:
                wqq[row,column]+=query_scale*value
            else:
                wqb[row,column-nq]+=value
    # Guarantee a direct (weak) boundary anchor even for points deep inside a hole.
    anchor_k=min(int(os.environ.get('R2_HARMONIC_ANCHOR_K','4')),boundary_k)
    anchor_scale=float(os.environ.get('R2_HARMONIC_ANCHOR_SCALE','.15'))
    for row in range(nq):
        for local in nearest[row,:anchor_k]:
            column=int(np.searchsorted(boundary_local,local))
            delta=(query_feature[row]-train_feature[local])/local_scale[row]
            wqb[row,column]+=anchor_scale*np.exp(-.5*float(delta@delta))
    wqq=(wqq+wqq.T)/2
    degree=wqq.sum(1)+wqb.sum(1)
    laplacian=np.diag(degree+1e-8)-wqq
    absorption=np.linalg.solve(laplacian,wqb)
    absorption=np.maximum(absorption,0)
    absorption/=np.maximum(absorption.sum(1,keepdims=True),1e-30)
    keep=min(int(topk),nb)
    chosen=np.argpartition(absorption,-keep,axis=1)[:,-keep:]
    weight=np.take_along_axis(absorption,chosen,axis=1)
    order=np.argsort(weight,axis=1)[:,::-1]
    chosen=np.take_along_axis(chosen,order,axis=1)
    weight=np.take_along_axis(weight,order,axis=1)
    weight/=np.maximum(weight.sum(1,keepdims=True),1e-30)
    return train[boundary_local[chosen]],weight


def island_validate():
    pos,ch,energy=load_data();tr,val=make_official_region_split();dist,idx=neighbor_plan(tr,pos[val],384)
    group_file=os.environ.get('R2_VAL_GROUP_FILE')
    groups=np.load(ROOT/group_file).astype(np.int64) if group_file else official_island_labels(pos[val])
    if len(groups)!=len(val):
        raise RuntimeError(f'Validation group override length {len(groups)} != validation length {len(val)}')
    only_group=os.environ.get('R2_ONLY_GROUP')
    if only_group is not None:
        keep=groups==int(only_group)
        val=val[keep];dist=dist[keep];idx=idx[keep];groups=groups[keep]
    side=pos[val,1]>0;defaults={False:((64,2.5,.25),(64,2.5,.35)),True:((48,3.5,.65),(64,2.5,1.25))}
    pred=np.empty((len(val),256,4,192),np.complex64);predicted_energy=np.empty(len(val),np.float64)
    for g in sorted(set(groups)):
        m=np.flatnonzero(groups==g);pa,pd=island_params_for(int(g),defaults[bool(np.mean(side[m])>.5)])
        kp,pp,qp=pa;kd,pd_,qd=pd
        pas_dist,pas_idx=dist[m,:kp],idx[m,:kp];pdp_dist,pdp_idx=dist[m,:kd],idx[m,:kd]
        metric_profile=active_feature_metric_profile().get(int(g),{})
        if 'pas' in metric_profile:
            pas_dist,pas_idx=configured_metric_neighbor_plan(tr,pos[val[m],:2],val[m],bool(np.mean(side[m])>.5),metric_profile['pas'],kp)
        if 'pdp' in metric_profile:
            pdp_dist,pdp_idx=configured_metric_neighbor_plan(tr,pos[val[m],:2],val[m],bool(np.mean(side[m])>.5),metric_profile['pdp'],kd)
        wp=1/np.maximum(pas_dist,.25)**pp;wp/=wp.sum(1,keepdims=True);energy_wp=wp.copy();wd=1/np.maximum(pdp_dist,.25)**pd_;wd/=wd.sum(1,keepdims=True);base_wp=wp.copy();base_wd=wd.copy()
        wp=affine_blend_weights(pos[val[m],:2],pos[pas_idx,:2],pas_dist,pp,wp,*affine_params_for(g,'pas'))
        wd=affine_blend_weights(pos[val[m],:2],pos[pdp_idx,:2],pdp_dist,pd_,wd,*affine_params_for(g,'pdp'))
        qp2=active_quadratic_profile().get(int(g),{});wp=quadratic_blend_weights(pos[val[m],:2],pos[pas_idx,:2],pas_dist,pp,base_wp,*qp2['pas']) if 'pas' in qp2 else wp;wd=quadratic_blend_weights(pos[val[m],:2],pos[pdp_idx,:2],pdp_dist,pd_,base_wd,*qp2['pdp']) if 'pdp' in qp2 else wd
        harmonic_groups={int(x) for x in os.environ.get('R2_HARMONIC_GROUPS','').split(',') if x.strip()}
        if os.environ.get('R2_HARMONIC_GRAPH') and (not harmonic_groups or int(g) in harmonic_groups):
            axes=os.environ.get('R2_HARMONIC_AXES','both')
            if axes in ('pas','both'):
                pas_idx,wp=harmonic_neighbor_plan(tr,pos[val[m],:2],val[m],bool(np.mean(side[m])>.5),metric_profile.get('pas'),kp)
                energy_wp=wp.copy()
            if axes in ('pdp','both'):
                pdp_idx,wd=harmonic_neighbor_plan(tr,pos[val[m],:2],val[m],bool(np.mean(side[m])>.5),metric_profile.get('pdp'),kd)
        if int(g) in (4,6) and os.environ.get('R2_LOCAL_RANK'):
            from local_rank_plan import select
            chosen=select(tr,pos[val[m],:2],idx[m],ch,pos)
            beta=float(os.environ.get('R2_LOCAL_RANK_BLEND','.2'))
            pas_idx=np.c_[pas_idx,chosen];wp=np.c_[wp*(1-beta),np.full(len(m),beta)]
        if int(g)==4 and os.environ.get('R2_G4_FEATURE'):
            fd,pas_idx=g4_feature_neighbor_plan(tr,pos[val[m],:2],val[m],32)
            wp=1/np.maximum(fd,.01);wp/=wp.sum(1,keepdims=True);kp=32;qp=.25
        init_rank=init_rank_for(int(g))
        os.environ['R2_INIT']=os.environ.get('R2_FORCE_INIT', 'weighted' if init_rank==5 else f'nearest{init_rank}')
        nit=projection_iters_for(int(g));pred[m]=project_spectra_gpu_dual(pas_idx,wp,pdp_idx,wd,nit,pas_q=qp,pdp_q=qd,final_pas_blend=final_pas_blend_for(int(g)) if os.environ.get('R2_FINAL_PAS_PROFILE') else None,group=int(g),query_xy=pos[val[m],:2])
        predicted_energy[m]=(energy_wp*energy[pas_idx]).sum(1)
    en=np.sum(np.abs(pred)**2,axis=(1,2,3),dtype=np.float64);unit_scale=np.sqrt(predicted_energy/np.maximum(en,1e-30));target=ch[val]
    if os.environ.get('R2_SAVE_UNIT'):
        np.savez(ROOT/os.environ['R2_SAVE_UNIT'],val=val,unit_scale=unit_scale,side=side,groups=groups,weights=np.zeros(len(val)))
        np.save(ROOT/(os.environ['R2_SAVE_UNIT']+'.unit.npy'),(pred*unit_scale[:,None,None,None]).astype(np.complex64))
    # Learn one calibration scalar per BS side; spectral cosines are invariant.
    calibrated=np.empty_like(pred)
    for is_right in [False,True]:
        m=side==is_right;u=pred[m]*unit_scale[m,None,None,None]
        if os.environ.get('R2_ROBUST_AMPLITUDE'):
            alpha=complex(float(os.environ['R2_ROBUST_AMPLITUDE']),0.)
        else:
            cross=np.vdot(u,target[m]);ue=float(np.sum(np.abs(u)**2,dtype=np.float64));te=float(np.sum(np.abs(target[m])**2,dtype=np.float64));alpha=cross/max(ue,1e-30)
        # Honest shrinkage: cap amplitude to the stable range established by sweeps.
        amp=min(abs(alpha),.2);phase=np.angle(alpha);calibrated[m]=u*(amp*np.exp(1j*phase));print('calibration',is_right,amp,phase,alpha)
    save_prediction = os.environ.get('R2_SAVE_PRED')
    if save_prediction:
        np.save(ROOT/save_prediction, calibrated)
    print(json.dumps(dict(method='island',**score_numpy(calibrated,target))))
    if not os.environ.get('R2_SKIP_LATEST_PRED'):
        np.save(ROOT/'latest_val_prediction.npy',calibrated)
    test_groups=official_island_labels(np.load(ROOT/'Round2_Test_Pos.npy'))
    weights=np.zeros(len(val),np.float64)
    for g,n in zip(*np.unique(test_groups,return_counts=True)):
        m=groups==g
        if np.any(m): weights[m]=n/np.sum(m)
    print(json.dumps(dict(method='island_official_weighted',**score_numpy_weighted(calibrated,target,weights))))
    if os.environ.get('R2_FAST_VALIDATE'):
        for g in sorted(set(groups)):
            m=groups==g
            r=score_numpy(calibrated[m],target[m]);r.update(island=int(g),val_n=int(m.sum()),test_n=int(np.sum(test_groups==g)))
            print(json.dumps(r))
        return
    # Per-UE complex calibration is exactly invariant to both PAS and PDP cosine
    # metrics, while offering four independent degrees of freedom for NMSE.
    ue_calibrated=np.empty_like(pred)
    for is_right in [False,True]:
        m=side==is_right
        for ue in range(4):
            u=pred[m,:,ue,:]*unit_scale[m,None,None];t=np.asarray(target[m,:,ue,:]);alpha=np.vdot(u,t)/max(float(np.sum(np.abs(u)**2,dtype=np.float64)),1e-30)
            alpha=min(abs(alpha),.2)*np.exp(1j*np.angle(alpha))
            ue_calibrated[m,:,ue,:]=u*alpha;print('ue_calibration',is_right,ue,alpha)
    print(json.dumps(dict(method='island_ue_calibration',**score_numpy(ue_calibrated,target))))
    print(json.dumps(dict(method='island_ue_calibration_weighted',**score_numpy_weighted(ue_calibrated,target,weights))))
    # Diagnostic upper bound: per-point/per-UE scalar (not used for test output).
    oracle=np.empty_like(pred)
    for i in range(len(pred)):
        for ue in range(4):
            u=pred[i,:,ue,:]*unit_scale[i];t=np.asarray(target[i,:,ue,:]);a=np.vdot(u,t)/max(float(np.sum(np.abs(u)**2,dtype=np.float64)),1e-30);oracle[i,:,ue,:]=u*a
    print(json.dumps(dict(method='point_ue_scalar_oracle_weighted',**score_numpy_weighted(oracle,target,weights))))
    # Alternating oracle phase gauge: column phases preserve PAS exactly;
    # row phases preserve PDP exactly. This quantifies phase-recovery headroom.
    gauge=pred.copy()
    for _ in range(6):
        # Per (point, UE, subcarrier) phase: invariant to PAS spectrum.
        cross=np.sum(np.conj(gauge)*np.asarray(target),axis=1)
        phase=np.exp(1j*np.angle(cross));gauge*=phase[:,None,:,:]
        # Per (point, BS antenna, UE) phase: invariant to PDP spectrum.
        cross=np.sum(np.conj(gauge)*np.asarray(target),axis=-1)
        phase=np.exp(1j*np.angle(cross));gauge*=phase[:,:,:,None]
    # One per-point/UE magnitude after phase gauge.
    for i in range(len(gauge)):
        for ue in range(4):
            u=gauge[i,:,ue,:];t=np.asarray(target[i,:,ue,:]);a=max(0,float(np.real(np.vdot(u,t))))/max(float(np.sum(np.abs(u)**2,dtype=np.float64)),1e-30);gauge[i,:,ue,:]=u*a
    print(json.dumps(dict(method='alternating_phase_gauge_oracle_weighted',**score_numpy_weighted(gauge,target,weights))))
    for g in sorted(set(groups)):
        m=groups==g
        r=score_numpy(calibrated[m],target[m]);r.update(island=int(g),val_n=int(m.sum()),test_n=int(np.sum(test_groups==g)))
        print(json.dumps(r))
    # Cross-fitted per-island calibration: one spatial half calibrates the other.
    crossfit=np.empty_like(pred)
    for g in sorted(set(groups)):
        ids=np.flatnonzero(groups==g);u=pred[ids]*unit_scale[ids,None,None,None]
        if len(ids)<8:
            is_right=bool(np.mean(side[ids])>.5);m=side==is_right;uu=pred[m]*unit_scale[m,None,None,None];alpha=np.vdot(uu,target[m])/max(float(np.sum(np.abs(uu)**2,dtype=np.float64)),1e-30);crossfit[ids]=u*alpha;continue
        order=np.argsort(pos[val[ids],0]+.618*pos[val[ids],1]);fold=np.arange(len(ids))%2
        for f in [0,1]:
            tune=order[fold==f];score=order[fold!=f];alpha=np.vdot(u[tune],target[ids[tune]])/max(float(np.sum(np.abs(u[tune])**2,dtype=np.float64)),1e-30);alpha=min(abs(alpha),.2)*np.exp(1j*np.angle(alpha));crossfit[ids[score]]=u[score]*alpha
    print(json.dumps(dict(method='island_crossfit_calibration',**score_numpy(crossfit,target))))


def dual_validate():
    pos,ch,energy=load_data();tr,val=make_official_region_split()
    dp,ip=neighbor_plan(tr,pos[val],64)
    dd,id_=dp,ip
    side=pos[val,1]>0
    pas_p=np.where(side,3.5,2.5);pdp_p=np.full(len(val),2.5)
    pas_q=np.where(side,.65,.25);pdp_q=np.where(side,1.25,.35)
    wp=1/np.maximum(dp,.25)**pas_p[:,None];wp/=wp.sum(1,keepdims=True)
    wd=1/np.maximum(dd,.25)**pdp_p[:,None];wd/=wd.sum(1,keepdims=True)
    cache=ROOT/'dual_val_projection_side_powermean.npy'
    if cache.exists(): pred=np.load(cache)
    else:
        # q varies by point; process the two sides independently.
        pred=np.empty((len(val),256,4,192),np.complex64)
        for is_right,pq,dq in [(False,.25,.35),(True,.65,1.25)]:
            m=np.flatnonzero(side==is_right)
            pred[m]=project_spectra_gpu_dual(ip[m],wp[m],id_[m],wd[m],20,pas_q=pq,pdp_q=dq)
        np.save(cache,pred)
    # Neighbor energy and fixed cross-validated phase/amplitude calibration.
    ne=energy[ip];pe=(wp*ne).sum(1)
    e=np.sum(np.abs(pred)**2,axis=(1,2,3),dtype=np.float64);scale=np.sqrt(pe/np.maximum(e,1e-30))
    for amp in [.05,.075,.1,.125,.15]:
        x=pred*(scale*amp)[:,None,None,None]*1j
        r=score_numpy(x,ch[val]);r.update(method='dual',amplitude=amp);print(json.dumps(r))


def spectral_fast_validate(fold: int):
    pos, channel, _ = load_data()
    train, val = make_block_split(fold)
    dist, idx = neighbor_plan(train, pos[val], 16)
    target = channel[val]
    configs = [(4, 1, 2, .5), (8, 1, 4, .5), (8, 2, 4, .5),
               (12, 1, 4, .5), (12, 2, 4, .5)]
    for k, power, iters, blend in configs:
        w = spatial_weights(pos, pos[val], idx[:, :k], dist[:, :k], "idw", power)
        pred = project_spectra_gpu(idx[:, :k], w, iters, blend)
        cross = np.vdot(pred, target)
        pred_e = float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
        alpha = cross / max(pred_e, 1e-30)
        result = score_numpy(pred * alpha, target)
        result.update(method="spectral_gpu", k=k, power=power, iterations=iters,
                      blend=blend, alpha_real=float(alpha.real), alpha_imag=float(alpha.imag))
        print(json.dumps(result), flush=True)


def spectral_norm_validate(fold: int):
    pos, channel, _ = load_data()
    train, val = make_official_region_split() if fold < 0 else make_block_split(fold)
    dist, idx = neighbor_plan(train, pos[val], 32)
    target = channel[val]
    configs = [(32, 3., 20, 1.)]
    for k, power, iters, blend in configs:
        w = spatial_weights(pos, pos[val], idx[:, :k], dist[:, :k], "idw", power)
        cache_path = ROOT / f"val_projection_k{k}_p{power}_i{iters}.npy"
        if fold < 0 and cache_path.exists():
            pred = np.load(cache_path)
        else:
            pred = project_spectra_gpu(idx[:, :k], w, iters, blend,
                                       normalize_shapes=True)
            if fold < 0:
                np.save(cache_path, pred)
        # Estimate per-query energy from neighbors; no validation labels are used.
        neighbor_energy = np.sum(np.abs(np.asarray(channel[idx[:, :k]])) ** 2,
                                 axis=(2,3,4), dtype=np.float64)
        predicted_energy = np.sum(w * neighbor_energy, axis=1)
        pred_energy = np.sum(np.abs(pred) ** 2, axis=(1,2,3), dtype=np.float64)
        unit_scale = np.sqrt(predicted_energy / np.maximum(pred_energy, 1e-30))
        unit = pred * unit_scale[:,None,None,None]
        base = predict_weighted(idx[:, :k], w)
        base_energy = np.sum(np.abs(base) ** 2, axis=(1,2,3), dtype=np.float64)
        base *= np.sqrt(predicted_energy / np.maximum(base_energy, 1e-30))[:,None,None,None]
        cross = np.vdot(unit, target)
        best_phase = np.angle(cross)
        print(json.dumps(dict(cross_real=float(cross.real), cross_imag=float(cross.imag),
                              best_phase=float(best_phase))), flush=True)
        for phase in [0., best_phase, -np.pi, -np.pi/2, np.pi/2]:
            for amplitude in [.01, .025, .05, .075, .1, .15, .2, .3]:
                scaled = unit * (amplitude * np.exp(1j*phase))
                result = score_numpy(scaled, target)
                result.update(method="spectral_norm_gpu", k=k, power=power, iterations=iters,
                              blend=blend, amplitude=amplitude, phase=float(phase))
                print(json.dumps(result), flush=True)
        # Upper bound if a pointwise complex scalar were predictable.
        cross_sample = np.sum(np.conj(unit) * np.asarray(target), axis=(1,2,3), dtype=np.complex128)
        energy_sample = np.sum(np.abs(unit) ** 2, axis=(1,2,3), dtype=np.float64)
        oracle_alpha = cross_sample / np.maximum(energy_sample, 1e-30)
        np.savez(ROOT / "val_scalar_targets.npz", val=val, alpha=oracle_alpha,
                 unit_scale=unit_scale, predicted_energy=predicted_energy)
        oracle = unit * oracle_alpha[:,None,None,None]
        result = score_numpy(oracle, target)
        result.update(method="pointwise_scalar_oracle")
        print(json.dumps(result), flush=True)
        # Honest candidates: fixed phase/amplitude plus normalized raw-channel residual.
        for residual in [.01,.025,.05,.075,.1,.15,.2]:
            mixed = unit * (.1j) + base * residual
            result = score_numpy(mixed, target)
            result.update(method="spectral_plus_base", residual=residual)
            print(json.dumps(result), flush=True)


def generate_submission(output: str = "Round2_Test_Channel.npy"):
    pos, channel, energy = load_data()
    query = np.load(ROOT / "Round2_Test_Pos.npy")
    train = np.flatnonzero(energy > 0)
    dist, idx = neighbor_plan(train, query, 384)
    groups=official_island_labels(query);side=query[:,1]>0;defaults={False:((64,2.5,.25),(64,2.5,.35)),True:((48,3.5,.65),(64,2.5,1.25))}
    base_prediction=os.environ.get('R2_BASE_PRED')
    if base_prediction:
        pred=np.asarray(np.load(ROOT/base_prediction,mmap_mode='r')).copy()
        base_amp=float(os.environ.get('R2_ROBUST_AMPLITUDE','.01'))
        predicted_energy=np.sum(np.abs(pred)**2,axis=(1,2,3),dtype=np.float64)/max(base_amp*base_amp,1e-30)
    else:
        pred=np.empty((len(query),256,4,192),np.complex64);predicted_energy=np.empty(len(query),np.float64)
    only_group=os.environ.get('R2_GENERATE_ONLY_GROUP')
    selected_groups=sorted(set(groups)) if only_group is None else [int(only_group)]
    for g in selected_groups:
        m=np.flatnonzero(groups==g);pa,pd=island_params_for(int(g),defaults[bool(np.mean(side[m])>.5)]);kp,pp,qp=pa;kd,pd_,qd=pd
        pas_dist,pas_idx=dist[m,:kp],idx[m,:kp];pdp_dist,pdp_idx=dist[m,:kd],idx[m,:kd]
        metric_profile=active_feature_metric_profile().get(int(g),{})
        if 'pas' in metric_profile:
            pas_dist,pas_idx=configured_metric_neighbor_plan(train,query[m,:2],len(pos)+m,bool(np.mean(side[m])>.5),metric_profile['pas'],kp)
        if 'pdp' in metric_profile:
            pdp_dist,pdp_idx=configured_metric_neighbor_plan(train,query[m,:2],len(pos)+m,bool(np.mean(side[m])>.5),metric_profile['pdp'],kd)
        wp=1/np.maximum(pas_dist,.25)**pp;wp/=wp.sum(1,keepdims=True);energy_wp=wp.copy();wd=1/np.maximum(pdp_dist,.25)**pd_;wd/=wd.sum(1,keepdims=True);base_wp=wp.copy();base_wd=wd.copy()
        wp=affine_blend_weights(query[m,:2],pos[pas_idx,:2],pas_dist,pp,wp,*affine_params_for(g,'pas'))
        wd=affine_blend_weights(query[m,:2],pos[pdp_idx,:2],pdp_dist,pd_,wd,*affine_params_for(g,'pdp'))
        qp2=active_quadratic_profile().get(int(g),{});wp=quadratic_blend_weights(query[m,:2],pos[pas_idx,:2],pas_dist,pp,base_wp,*qp2['pas']) if 'pas' in qp2 else wp;wd=quadratic_blend_weights(query[m,:2],pos[pdp_idx,:2],pdp_dist,pd_,base_wd,*qp2['pdp']) if 'pdp' in qp2 else wd
        harmonic_groups={int(x) for x in os.environ.get('R2_HARMONIC_GROUPS','').split(',') if x.strip()}
        if os.environ.get('R2_HARMONIC_GRAPH') and (not harmonic_groups or int(g) in harmonic_groups):
            axes=os.environ.get('R2_HARMONIC_AXES','both')
            if axes in ('pas','both'):
                pas_idx,wp=harmonic_neighbor_plan(train,query[m,:2],len(pos)+m,bool(np.mean(side[m])>.5),metric_profile.get('pas'),kp)
                energy_wp=wp.copy()
            if axes in ('pdp','both'):
                pdp_idx,wd=harmonic_neighbor_plan(train,query[m,:2],len(pos)+m,bool(np.mean(side[m])>.5),metric_profile.get('pdp'),kd)
        if int(g) in (4,6) and os.environ.get('R2_LOCAL_RANK'):
            from local_rank_plan import select
            chosen=select(train,query[m,:2],idx[m],channel,pos)
            beta=float(os.environ.get('R2_LOCAL_RANK_BLEND','.2'))
            pas_idx=np.c_[pas_idx,chosen];wp=np.c_[wp*(1-beta),np.full(len(m),beta)]
        if int(g)==4 and os.environ.get('R2_G4_FEATURE'):
            fd,pas_idx=g4_feature_neighbor_plan(train,query[m,:2],len(pos)+m,32)
            wp=1/np.maximum(fd,.01);wp/=wp.sum(1,keepdims=True);kp=32;qp=.25
        init_rank=init_rank_for(int(g))
        os.environ['R2_INIT']=os.environ.get('R2_FORCE_INIT', 'weighted' if init_rank==5 else f'nearest{init_rank}')
        nit=projection_iters_for(int(g));pred[m]=project_spectra_gpu_dual(pas_idx,wp,pdp_idx,wd,nit,pas_q=qp,pdp_q=qd,final_pas_blend=final_pas_blend_for(int(g)) if os.environ.get('R2_FINAL_PAS_PROFILE') else None,group=int(g),query_xy=query[m,:2]);predicted_energy[m]=(energy_wp*energy[pas_idx]).sum(1)
    pred_energy=np.sum(np.abs(pred)**2,axis=(1,2,3),dtype=np.float64)
    unit_scale=np.sqrt(predicted_energy/np.maximum(pred_energy,1e-30))
    if os.environ.get('R2_ROBUST_AMPLITUDE'):
        # The dataset has a point-wise phase gauge that is not predictable from
        # coordinates. A small nonzero magnitude preserves PAS/PDP exactly and
        # keeps NMSE close to 1 without fitting validation phase.
        amp=float(os.environ['R2_ROBUST_AMPLITUDE']);alpha_by_side={False:amp+0j,True:amp+0j}
    else:
        alpha_by_side={False:0.08382487862797007+0.08543028686904118j,True:-0.06274396424114065+0.10461081832878642j}
    for is_right,alpha in alpha_by_side.items():
        m=side==is_right;pred[m]*=(unit_scale[m]*alpha)[:,None,None,None]
    path = ROOT / output
    np.save(path, pred.astype(np.complex64))
    check = np.load(path, mmap_mode="r")
    if check.shape != (500, 256, 4, 192) or check.dtype != np.complex64:
        raise RuntimeError(f"Invalid submission: {check.shape} {check.dtype}")
    print(json.dumps({"output": str(path), "shape": check.shape,
                      "dtype": str(check.dtype), "bytes": path.stat().st_size,
                      "finite": bool(np.isfinite(check).all())}), flush=True)


def spectral_oracle_sweep(fold: int, normalized: bool = False, wide: bool = False):
    """Score the interpolated target spectra before phase-retrieval compatibility loss."""
    import torch
    pos, channel, _ = load_data()
    train, val = make_official_region_split() if fold < 0 else make_block_split(fold)
    max_k = 128 if wide else 32
    dist, idx = neighbor_plan(train, pos[val], max_k)
    device = torch.device("cuda")
    configs = [("nearest", 1, 0.0)]
    configs += [("idw", k, p) for k in [2, 4, 6, 8, 12, 16, 24, 32]
                for p in [.5, 1., 1.5, 2., 3., 4.]]
    configs += [("gaussian", k, p) for k in [8, 12, 16, 24, 32]
                for p in [.25, .35, .5, .75, 1., 1.5]]
    configs += [("linear", k, p) for k in [8, 12, 16, 24, 32]
                for p in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]]
    if wide:
        configs = [("idw", k, p) for k in [32, 48, 64, 96, 128]
                   for p in [.5, 1., 1.5, 2., 3.]]
        configs += [("gaussian", k, p) for k in [32, 48, 64, 96, 128]
                    for p in [.15, .25, .35, .5, .75, 1.]]
    all_weights = {}
    for method, k, power in configs:
        if method != "nearest":
            all_weights[(method, k, power)] = spatial_weights(
                pos, pos[val], idx[:, :k], dist[:, :k], method, power)
    totals = {c: [0., 0., 0] for c in configs}
    for start in range(0, len(val), 4):
        stop = min(start + 4, len(val))
        t = torch.as_tensor(np.asarray(channel[val[start:stop]]).copy(), device=device)
        target_pas = torch.abs(bs_fft_torch(t)) ** 2
        target_pdp = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
        ids = idx[start:stop]
        h = torch.as_tensor(np.asarray(channel[ids]).copy(), device=device)
        npas = torch.abs(bs_fft_torch(h)) ** 2
        npdp = torch.abs(torch.fft.fft(h, dim=4, norm="ortho")) ** 2
        if normalized:
            npas = npas / torch.linalg.vector_norm(npas, dim=2, keepdim=True).clamp_min(1e-30)
            npdp = npdp / torch.linalg.vector_norm(npdp, dim=4, keepdim=True).clamp_min(1e-30)
        for config in configs:
            method, k, power = config
            if method == "nearest":
                pp, pd = npas[:, 0], npdp[:, 0]
            else:
                w = all_weights[config][start:stop]
                wt = torch.as_tensor(w, device=device, dtype=torch.float32)
                pp = torch.sum(wt[:, :, None, None, None] * npas[:, :k], dim=1).clamp_min(0)
                pd = torch.sum(wt[:, :, None, None, None] * npdp[:, :k], dim=1).clamp_min(0)
            c1 = ((pp * target_pas).sum(dim=1) /
                  (torch.linalg.vector_norm(pp, dim=1) *
                   torch.linalg.vector_norm(target_pas, dim=1)).clamp_min(1e-30)).sum()
            c2 = ((pd * target_pdp).sum(dim=-1) /
                  (torch.linalg.vector_norm(pd, dim=-1) *
                   torch.linalg.vector_norm(target_pdp, dim=-1)).clamp_min(1e-30)).sum()
            totals[config][0] += float(c1)
            totals[config][1] += int(target_pas.shape[0]*target_pas.shape[2]*target_pas.shape[3])
            totals[config][2] += float(c2)
        print(f"oracle {stop}/{len(val)}", flush=True)
    results=[]
    pdp_n = len(val)*256*4
    for c,(s1,n1,s2) in totals.items():
        c1=s1/n1; c2=s2/pdp_n; score=.4*c1+.4*c2+.1
        results.append((score,c,c1,c2))
    for score,c,c1,c2 in sorted(results, reverse=True)[:15]:
        print(json.dumps(dict(score=score, normalized=normalized, method=c[0], k=c[1], power=c[2],
                              c1_pas=c1, c2_pdp=c2)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "describe-split", "sweep", "spectral", "spectral-fast", "spectral-norm", "dual-validate", "island-validate", "spectral-oracle", "spectral-oracle-norm", "spectral-oracle-wide", "delaunay-oracle", "generate"])
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    if args.command == "validate":
        validate(args.fold)
    elif args.command == "describe-split":
        describe_split(args.fold)
    elif args.command == "sweep":
        sweep(args.fold)
    elif args.command == "spectral":
        spectral_validate(args.fold)
    elif args.command == "spectral-fast":
        spectral_fast_validate(args.fold)
    elif args.command == "spectral-norm":
        spectral_norm_validate(args.fold)
    elif args.command == "dual-validate":
        dual_validate()
    elif args.command == "island-validate":
        island_validate()
    elif args.command == "spectral-oracle":
        spectral_oracle_sweep(args.fold)
    elif args.command == "spectral-oracle-norm":
        spectral_oracle_sweep(args.fold, normalized=True)
    elif args.command == "spectral-oracle-wide":
        spectral_oracle_sweep(args.fold, normalized=True, wide=True)
    elif args.command == "generate":
        generate_submission(os.environ.get("R2_OUTPUT", "Round2_Test_Channel.npy"))
    elif args.command == "delaunay-oracle":
        delaunay_oracle(args.fold)
