from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp


K = 64


def _maps(channel):
    out = []
    for start in range(0, len(channel), 16):
        h = torch.as_tensor(np.asarray(channel[start:start + 16]).copy(), device='cuda')
        power = torch.abs(rp.bs_fft_torch(h)) ** 2
        out.append(power.reshape(len(power), 2, 16, 8, 4, 192).sum((1, 4, 5)).cpu().numpy())
    return np.concatenate(out)


def _features(pos, query, candidates):
    n = pos[candidates, :2]
    delta = n - query[:, None]
    distance = np.linalg.norm(delta, axis=2)
    side = (query[:, 1] > 0).astype(int)
    bs = np.array([[-18.413, -65.881], [52., 35.]])[side]
    qb = query - bs
    nb = n - bs[:, None]
    qr = np.linalg.norm(qb, axis=1)
    nr = np.linalg.norm(nb, axis=2)
    qa = np.arctan2(qb[:, 1], qb[:, 0])
    na = np.arctan2(nb[:, :, 1], nb[:, :, 0])
    da = np.arctan2(np.sin(na - qa[:, None]), np.cos(na - qa[:, None]))
    rank = np.broadcast_to(np.arange(candidates.shape[1]), candidates.shape)
    return np.stack([
        np.broadcast_to(query[:, 0, None], distance.shape),
        np.broadcast_to(query[:, 1, None], distance.shape), n[:, :, 0], n[:, :, 1],
        delta[:, :, 0], delta[:, :, 1], distance,
        np.broadcast_to(qr[:, None], distance.shape), nr,
        np.broadcast_to(qa[:, None], distance.shape), na, da, rank,
    ], axis=2)


def select(train, query, candidate_ids, channel, pos, radius=180., leaf=2):
    """Honest local model: labels come exclusively from pairs inside train."""
    spectra = np.load(rp.ROOT/'normalized_spectra_2d_f16.npy', mmap_mode='r')
    rng = np.random.default_rng(931)
    dims = np.sort(rng.choice(256*4*192, 32768, replace=False))
    tree = cKDTree(pos[train, :2])
    _, local = tree.query(pos[train, :2], k=K + 1)
    train_candidates = train[local[:, 1:]]
    center = query.mean(0)
    local_mask = np.linalg.norm(pos[train, :2] - center, axis=1) < radius
    query_train = train[local_mask]
    candidates = train_candidates[local_mask]
    x = _features(pos, pos[query_train, :2], candidates).reshape(-1, 13)
    y=np.empty((len(query_train),K),np.float32)
    for start in range(0,len(query_train),32):
        end=min(start+32,len(query_train))
        a=np.asarray(spectra[query_train[start:end],0][:,dims],dtype=np.float32)
        b=np.asarray(spectra[candidates[start:end],0][:,:,dims],dtype=np.float32)
        y[start:end]=np.einsum('bd,bkd->bk',a,b)/len(dims)
    model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=leaf,
                                max_features=1.0, n_jobs=-1,
                                random_state=42).fit(x, y.reshape(-1))
    candidates = candidate_ids[:, :K]
    score = model.predict(_features(pos, query, candidates).reshape(-1, 13)).reshape(len(query), K)
    return candidates[np.arange(len(query)), score.argmax(1)]
