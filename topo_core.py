"""Exact H0/H1 persistence pairing for 2-D Delaunay-type complexes.

Complex: vertices (cells), Delaunay edges, Delaunay triangles.
Filtration: vertex values fv, edge values fe >= max(fv[u], fv[v]),
triangle value = max of its three edge values (lower-star / flag style).

H0: Kruskal union-find with elder rule (births = vertex values).
H1: Alexander-type duality on the planar complex: loops created by edges
    are killed by triangles; computed as superlevel H0 on the dual graph
    (triangles + one outer face of value +inf), processed in reverse order.

Only *indices* are returned (which vertex/edge gives each birth/death), so the
caller gathers values from autograd tensors -> exact a.e. gradients.
"""
import numpy as np
from scipy.spatial import Delaunay

try:
    from numba import njit
except Exception:  # numba optional (pure python fallback)
    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f


def build_complex(coords):
    """Delaunay edges (E,2), triangles (T,3) as edge ids, edge->2 faces (outer = T)."""
    coords = np.asarray(coords, dtype=np.float64)
    tri = Delaunay(coords, qhull_options="QJ").simplices.astype(np.int64)
    T = tri.shape[0]
    e_all = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]], 0)
    e_all.sort(1)
    edges, inv = np.unique(e_all, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    tri_edges = np.stack([inv[:T], inv[T:2 * T], inv[2 * T:]], 1).astype(np.int64)
    E = edges.shape[0]
    faces = np.full((E, 2), T, dtype=np.int64)  # default outer face
    cnt = np.zeros(E, dtype=np.int64)
    for t in range(T):
        for k in range(3):
            e = tri_edges[t, k]
            faces[e, cnt[e]] = t
            cnt[e] += 1
    return edges.astype(np.int64), tri_edges, faces


@njit(cache=True)
def _find(par, x):
    r = x
    while par[r] != r:
        r = par[r]
    while par[x] != r:
        nx = par[x]
        par[x] = r
        x = nx
    return r


@njit(cache=True)
def _pairs(n_v, edges, faces, n_t, fv, fe, tri_maxedge):
    E = edges.shape[0]
    order = np.argsort(fe, kind="mergesort")
    rank = np.empty(E, np.int64)
    for i in range(E):
        rank[order[i]] = i
    # ---------- H0 ----------
    vorder = np.argsort(fv, kind="mergesort")
    vrank = np.empty(n_v, np.int64)
    for i in range(n_v):
        vrank[vorder[i]] = i
    par = np.arange(n_v)
    oldest = np.arange(n_v)  # vertex with smallest (fv, rank) in component
    h0_b = np.empty(n_v, np.int64)
    h0_d = np.empty(n_v, np.int64)
    n0 = 0
    for i in range(E):
        e = order[i]
        a = _find(par, edges[e, 0])
        b = _find(par, edges[e, 1])
        if a == b:
            continue
        oa = oldest[a]
        ob = oldest[b]
        if vrank[oa] < vrank[ob]:
            young, keep = b, a
        else:
            young, keep = a, b
        h0_b[n0] = oldest[young]
        h0_d[n0] = e
        n0 += 1
        par[young] = keep
    # ---------- H1 via dual ----------
    # triangle key: rank of its max edge (ties by triangle id); outer = +inf
    tkey = np.empty(n_t + 1, np.int64)
    for t in range(n_t):
        tkey[t] = rank[tri_maxedge[t]] * (n_t + 1) + t
    tkey[n_t] = 1 << 62
    dpar = np.arange(n_t + 1)
    dold = np.arange(n_t + 1)  # triangle with LARGEST key in dual comp
    h1_b = np.empty(n_t, np.int64)
    h1_d = np.empty(n_t, np.int64)
    n1 = 0
    for i in range(E - 1, -1, -1):
        e = order[i]
        a = _find(dpar, faces[e, 0])
        b = _find(dpar, faces[e, 1])
        if a == b:
            continue
        oa = dold[a]
        ob = dold[b]
        if tkey[oa] > tkey[ob]:
            young, keep = b, a
        else:
            young, keep = a, b
        h1_b[n1] = e
        h1_d[n1] = dold[young]
        n1 += 1
        dpar[young] = keep
    return h0_b[:n0], h0_d[:n0], h1_b[:n1], h1_d[:n1]


def persistence_pairs(fv, fe, edges, tri_edges, faces):
    """Returns index arrays:
       h0_birth_vertex, h0_death_edge, h1_birth_edge, h1_death_edge(=max edge of killing triangle).
    """
    fv = np.ascontiguousarray(fv, dtype=np.float64)
    fe = np.ascontiguousarray(fe, dtype=np.float64)
    tm = tri_edges[np.arange(tri_edges.shape[0])[:, None],
                   np.argmax(fe[tri_edges], 1)[:, None]][:, 0]
    tm = np.ascontiguousarray(tm, dtype=np.int64)
    h0b, h0d, h1b, h1t = _pairs(fv.shape[0], edges, faces, tri_edges.shape[0], fv, fe, tm)
    return h0b, h0d, h1b, tm[h1t]


# ---------------------------------------------------------------- brute force
def _brute_diagrams(fv, fe, edges, tri_edges):
    """Standard Z2 column reduction (for testing)."""
    N, E, T = len(fv), len(edges), len(tri_edges)
    ft = fe[tri_edges].max(1)
    simp = [(fv[i], 0, i) for i in range(N)] + [(fe[i], 1, i) for i in range(E)] + \
           [(ft[i], 2, i) for i in range(T)]
    simp.sort(key=lambda s: (s[0], s[1]))
    pos = {(d, i): k for k, (_, d, i) in enumerate(simp)}
    cols = []
    for v, d, i in simp:
        if d == 0:
            cols.append(set())
        elif d == 1:
            cols.append({pos[(0, edges[i, 0])], pos[(0, edges[i, 1])]})
        else:
            cols.append({pos[(1, e)] for e in tri_edges[i]})
    low_of = {}
    D0, D1 = [], []
    for j in range(len(cols)):
        c = cols[j]
        while c and max(c) in low_of:
            c = c ^ cols[low_of[max(c)]]
        cols[j] = c
        if c:
            l = max(c)
            low_of[l] = j
            b = simp[l][0]
            dval = simp[j][0]
            (D0 if simp[j][1] == 1 else D1).append((b, dval))
    return sorted(D0), sorted(D1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for trial in range(30):
        n = int(rng.integers(5, 80))
        X = rng.random((n, 2))
        edges, tri_edges, faces = build_complex(X)
        fv = rng.random(n) * 0.3
        w = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1) * np.exp(rng.normal(size=len(edges)))
        fe = np.maximum(np.maximum(fv[edges[:, 0]], fv[edges[:, 1]]), w)
        h0b, h0d, h1b, h1d = persistence_pairs(fv, fe, edges, tri_edges, faces)
        D0 = sorted(zip(fv[h0b], fe[h0d]))
        D1 = sorted(zip(fe[h1b], fe[h1d]))
        B0, B1 = _brute_diagrams(fv, fe, edges, tri_edges)
        strip = lambda D: sorted((round(a, 10), round(b, 10)) for a, b in D if b - a > 1e-12)
        assert strip(D0) == strip(B0), ("H0 mismatch", trial)
        assert strip(D1) == strip(B1), ("H1 mismatch", trial)
        assert len(h1b) == len(tri_edges)
    print("persistence tests passed")
