"""Landmark graph topology learned from training coordinates (numpy only)."""

from __future__ import annotations

import numpy as np

__all__ = [
    "group_graphical_lasso",
    "block_norms",
    "standardised_correlation",
    "lambda_path",
    "heldout_loglik",
    "select_lambda",
    "spanning_edges",
    "build_topology",
    "SELECTIONS",
]

SELECTIONS = ("mst", "connected")


def group_graphical_lasso(S, lam, d=3, rho=1.0, max_iter=500, tol=1e-5, return_info=False):
    """ADMM for max log det(T) - tr(S T) - sum_{i != j} lam_ij ||T_ij||_F over d x d blocks."""
    S = np.asarray(S, float)
    p = S.shape[0]
    if p % d:
        raise ValueError(f"p={p} is not a multiple of d={d}")
    K = p // d
    lam = np.broadcast_to(np.asarray(lam, float), (K, K)).copy()
    np.fill_diagonal(lam, 0.0)
    Z, U = np.eye(p), np.zeros((p, p))
    r = s = np.inf
    for it in range(max_iter):
        M = Z - U - S / rho
        w, Q = np.linalg.eigh(0.5 * (M + M.T))
        Theta = (Q * ((w + np.sqrt(w**2 + 4.0 / rho)) / 2.0)) @ Q.T
        Zold = Z
        Ab = (Theta + U).reshape(K, d, K, d)
        nrm = np.linalg.norm(Ab, axis=(1, 3))
        with np.errstate(invalid="ignore"):
            scale = np.maximum(0.0, 1.0 - lam / (rho * np.maximum(nrm, 1e-30)))
        scale[~np.isfinite(lam)] = 0.0
        np.fill_diagonal(scale, 1.0)
        Z = (Ab * scale[:, None, :, None]).reshape(p, p)
        Z = 0.5 * (Z + Z.T)
        U = U + Theta - Z
        r, s = np.linalg.norm(Theta - Z), rho * np.linalg.norm(Z - Zold)
        if r < tol * max(1.0, np.linalg.norm(Theta)) and s < tol * max(
            1.0, rho * np.linalg.norm(U)
        ):
            break
    info = {
        "n_iter": it + 1,
        "primal": float(r),
        "dual": float(s),
        "converged": bool(it + 1 < max_iter),
    }
    return (Z, info) if return_info else Z


def block_norms(Theta, d=3):
    """(K, K) Frobenius norm of every d x d off-diagonal block, zero diagonal."""
    K = Theta.shape[0] // d
    B = np.linalg.norm(np.asarray(Theta).reshape(K, d, K, d), axis=(1, 3))
    np.fill_diagonal(B, 0.0)
    return B


def _standardise(F, ref=None):
    ref = F if ref is None else ref
    sd = ref.std(0, ddof=1)
    return (F - ref.mean(0)) / np.where(sd < 1e-9, 1.0, sd)


def standardised_correlation(X, ref=None):
    """X (n, K, d) -> correlation of the standardised coordinates (statistics of `ref`
    when given, so a held-out fold is scaled like its training fold)."""
    F = np.asarray(X, float).reshape(len(X), -1)
    R = None if ref is None else np.asarray(ref, float).reshape(len(ref), -1)
    Z = _standardise(F, R)
    return Z.T @ Z / len(Z)


def _fit(S, lam, d, group):
    return group_graphical_lasso(S, lam, d=d if group else 1, max_iter=2000, tol=1e-8)


def lambda_path(S, d=3, n_lambda=30, ratio=0.02, group=False):
    """Log-spaced path from the smallest lambda that zeroes every off-diagonal penalised
    unit (block or entry) down to ratio * it."""
    if group:
        lam_max = float(block_norms(S, d).max())
    else:
        A = np.abs(S - np.diag(np.diag(S)))
        lam_max = float(A.max())
    return lam_max * np.logspace(0.0, np.log10(ratio), n_lambda)


def heldout_loglik(Theta, S_test):
    sign, logdet = np.linalg.slogdet(Theta)
    return -np.inf if sign <= 0 else float(logdet - np.trace(S_test @ Theta))


def _connected(adj):
    K = len(adj)
    seen, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(adj[i]):
            if j not in seen:
                seen.add(int(j))
                stack.append(int(j))
    return len(seen) == K


class TooFewSubjects(ValueError):
    """Not enough training subjects to fit the landmark graph's penalty by cross-validation."""


def select_lambda(X, d=3, group=False, n_folds=5, n_lambda=30, ratio=0.02, seed=0, zero_tol=1e-8):
    """Cross-validated lambda (max mean held-out log-likelihood), lowered along the path
    until the support connects every landmark. Returns (lambda, Theta fitted on all X, info)."""
    X = np.asarray(X, float)
    n = len(X)
    if n < 2 * n_folds:
        raise TooFewSubjects(
            f"{n} training subjects are too few to choose the graph's penalty by "
            f"{n_folds}-fold cross-validation (at least {2 * n_folds} are needed). Fix the "
            "penalty instead: --set model.lam.graph_lambda=0.1, or use every pair of "
            "landmarks with --set model.lam.graph_topology=fully_connected"
        )
    S = standardised_correlation(X)
    path = lambda_path(S, d, n_lambda, ratio, group)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), n_folds)
    cv = np.zeros((n_folds, len(path)))
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test)
        S_tr = standardised_correlation(X[train])
        S_te = standardised_correlation(X[test], ref=X[train])
        for k, lam in enumerate(path):
            cv[f, k] = heldout_loglik(_fit(S_tr, lam, d, group), S_te)
    mean = cv.mean(0)
    best = int(np.argmax(mean))
    fits = {}

    def fit_all(k):
        if k not in fits:
            Th = _fit(S, path[k], d, group)
            fits[k] = (Th, block_norms(Th, d) > zero_tol)
        return fits[k]

    chosen = best
    while chosen < len(path) - 1 and not _connected(fit_all(chosen)[1]):
        chosen += 1
    Theta, support = fit_all(chosen)
    if not _connected(support):
        raise ValueError("no lambda on the path connects every landmark; lower `ratio`")
    curve = [
        {"lambda": float(l), "cv_loglik": float(m), "cv_sd": float(s)}
        for l, m, s in zip(path, mean, cv.std(0))
    ]
    return (
        float(path[chosen]),
        Theta,
        {
            "criterion": f"{n_folds}-fold CV held-out log-likelihood",
            "n_subjects": n,
            "p": int(S.shape[0]),
            "group": bool(group),
            "cv_best_lambda": float(path[best]),
            "selected_lambda": float(path[chosen]),
            "lowered_for_connectivity": chosen != best,
            "n_pairs_in_support": int(np.triu(support, 1).sum()),
            "path": curve,
        },
    )


def spanning_edges(score, keep_cycles=False):
    """Kruskal on decreasing score: a pair is kept if it joins two separate groups
    (or always, with keep_cycles), until one group remains. Returns a symmetric bool (K, K)."""
    K = len(score)
    iu = np.triu_indices(K, 1)
    order = np.lexsort((iu[1], iu[0], -score[iu]))
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    A = np.zeros((K, K), bool)
    groups = K
    for o in order:
        i, j = int(iu[0][o]), int(iu[1][o])
        ri, rj = find(i), find(j)
        joins = ri != rj
        if joins:
            parent[ri] = rj
            groups -= 1
        if joins or keep_cycles:
            A[i, j] = A[j, i] = True
        if groups == 1:
            return A
    raise ValueError(f"the score leaves {groups} separate groups: no spanning graph exists")


def build_topology(X, select="mst", lam="auto", d=3, group=False):
    """X (n, K, d) training coordinates -> {'edge_index': (2, 2E) int64, 'provenance': dict}.

    lam="auto" selects the penalty by cross-validation (see select_lambda); a float fixes it.
    """
    if select not in SELECTIONS:
        raise ValueError(f"select must be one of {SELECTIONS}, got {select!r}")
    X = np.asarray(X, float)
    if X.ndim != 3 or X.shape[2] != d:
        raise ValueError(f"coords must be (n, K, {d}), got {X.shape}")
    if lam == "auto":
        lam_value, Theta, sel = select_lambda(X, d=d, group=group)
    else:
        lam_value = float(lam)
        Theta = _fit(standardised_correlation(X), lam_value, d, group)
        sel = {"criterion": "fixed", "selected_lambda": lam_value, "group": bool(group)}
    score = block_norms(Theta, d)
    A = spanning_edges(score, keep_cycles=(select == "connected"))
    i, j = np.nonzero(A)
    deg = A.sum(1)
    iu = np.triu_indices(len(A), 1)
    kept = [(int(a), int(b), float(score[a, b])) for a, b in zip(*iu) if A[a, b]]
    return {
        "edge_index": np.stack([i, j]).astype(np.int64),
        "adjacency": A,
        "score": score,
        "provenance": {
            "select": select,
            "method": "graphical lasso" + (" (grouped 3x3)" if group else "") + " + Kruskal",
            "lambda": lam_value,
            "lambda_selection": sel,
            "n_subjects": int(len(X)),
            "K": int(X.shape[1]),
            "n_edges": int(A.sum() // 2),
            "degree_min": int(deg.min()),
            "degree_max": int(deg.max()),
            "edges_by_strength": sorted(kept, key=lambda e: -e[2]),
        },
    }
