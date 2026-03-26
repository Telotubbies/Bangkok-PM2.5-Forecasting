"""
graph_builder.py
----------------
Build all graph structures for STC-HGAT PM2.5 forecasting.

Implements ideas from STC-HGAT paper (Yang & Peng, 2024) adapted for air quality:

  Paper concept              → PM2.5 adaptation
  ─────────────────────────────────────────────
  Spatial hyperedges         → Station proximity hyperedges (multi-scale)
  LDA category nodes         → Geographic region nodes (North/Central/South/East/NE)
  Semantic hyperedges        → PM2.5 correlation-based hyperedges
  Temporal graph             → Sequential day-edges + seasonal pattern edges
  Wind edges                 → Dynamic wind-direction edges (u10/v10)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Station region mapping  (paper: LDA category nodes → geographic clusters)
# ---------------------------------------------------------------------------
THAILAND_REGIONS: Dict[str, List[str]] = {
    "North":     ["CM", "LM", "LP", "PY", "NAN", "PR", "PL", "MSH", "TAK", "SUK"],
    "Northeast": ["KKN", "UDT", "NKP", "SK", "BRM", "ROI", "MKM", "YST", "SR"],
    "Central":   ["BKK", "NPT", "AYA", "SBR", "NBR", "CNB", "KRI", "PKN", "SMT"],
    "East":      ["RY", "CH", "TRT", "PKO", "SA"],
    "South":     ["HYI", "PKT", "SNG", "NRT", "PT", "KBI", "TG", "PAN"],
}
REGION_NAMES = list(THAILAND_REGIONS.keys())   # 5 regions


# ---------------------------------------------------------------------------
# 1. Haversine distance
# ---------------------------------------------------------------------------

def haversine_km(
    lat1: float | Tensor, lon1: float | Tensor,
    lat2: float | Tensor, lon2: float | Tensor,
) -> Tensor:
    """Vectorised haversine distance in kilometres."""
    R = 6371.0

    def to_t(x: float | Tensor) -> Tensor:
        return x if isinstance(x, Tensor) else torch.tensor(x, dtype=torch.float32)

    lat1, lon1, lat2, lon2 = map(to_t, [lat1, lon1, lat2, lon2])
    dlat = torch.deg2rad(lat2 - lat1)
    dlon = torch.deg2rad(lon2 - lon1)
    a = (torch.sin(dlat / 2) ** 2
         + torch.cos(torch.deg2rad(lat1))
         * torch.cos(torch.deg2rad(lat2))
         * torch.sin(dlon / 2) ** 2)
    return R * 2 * torch.asin(torch.clamp(torch.sqrt(a), 0.0, 1.0))


def pairwise_distance_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return (N, N) symmetric haversine distance matrix in km."""
    N = len(lats)
    D = np.zeros((N, N), dtype=np.float32)
    lat_t = torch.tensor(lats, dtype=torch.float32)
    lon_t = torch.tensor(lons, dtype=torch.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = float(haversine_km(lat_t[i], lon_t[i], lat_t[j], lon_t[j]))
            D[i, j] = D[j, i] = d
    return D


# ---------------------------------------------------------------------------
# 2. Spatial edges (simple pairwise, used as base graph)
# ---------------------------------------------------------------------------

def build_spatial_edges(
    lats: np.ndarray,
    lons: np.ndarray,
    threshold_km: float = 150.0,
) -> Tensor:
    """
    Connect stations within *threshold_km*.  Falls back to k-NN if no
    edges are found (avoids isolated nodes).

    Returns
    -------
    edge_index : LongTensor (2, E)  — bidirectional
    """
    D = pairwise_distance_matrix(lats, lons)
    src, dst = np.where((D < threshold_km) & (D > 0))

    if len(src) == 0:
        # Fallback: connect every node to its 3 nearest neighbours
        src_list, dst_list = [], []
        for i in range(len(lats)):
            row = D[i].copy(); row[i] = np.inf
            for j in np.argsort(row)[:3]:
                src_list += [i, j]; dst_list += [j, i]
        src, dst = np.array(src_list), np.array(dst_list)

    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


# ---------------------------------------------------------------------------
# 3. Semantic edges  (correlation-based, paper: semantic hyperedges)
# ---------------------------------------------------------------------------

def build_semantic_edges(
    pm25_history: np.ndarray,
    corr_threshold: float = 0.70,
) -> Tensor:
    """
    Connect stations whose Pearson PM2.5 correlation >= threshold.

    Parameters
    ----------
    pm25_history : ndarray (N, T_hist)
    """
    corr = np.corrcoef(pm25_history)                       # (N, N)
    eye  = ~np.eye(len(corr), dtype=bool)
    idx  = np.argwhere((corr >= corr_threshold) & eye)

    if len(idx) == 0:                                       # fallback: top-3
        rows, cols = [], []
        for i in range(len(corr)):
            row = corr[i].copy(); row[i] = -1
            for j in np.argsort(row)[-3:]:
                rows += [i, j]; cols += [j, i]
        idx = np.array(list(zip(rows, cols)))

    return torch.tensor(idx.T, dtype=torch.long)


# ---------------------------------------------------------------------------
# 4. Wind-direction edges  (dynamic, recomputed each day)
# ---------------------------------------------------------------------------

def build_wind_edges(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    top_k: int = 3,
) -> Tensor:
    """
    Connect station i to the top-k downwind stations (positive dot-product
    of wind vector with i→j direction vector).

    Parameters
    ----------
    wind_u, wind_v : ndarray (N,)  daily mean u10/v10
    """
    N = len(lats)
    src, dst = [], []
    for i in range(N):
        scores = np.full(N, -np.inf)
        for j in range(N):
            if i == j:
                continue
            # flat-earth direction vector i→j
            dx = lons[j] - lons[i]
            dy = lats[j] - lats[i]
            norm = math.sqrt(dx ** 2 + dy ** 2) + 1e-8
            scores[j] = (wind_u[i] * dx + wind_v[i] * dy) / norm

        for j in np.argsort(scores)[-top_k:]:
            if scores[j] > 0:                               # only downwind
                src.append(i); dst.append(j)

    if not src:                                             # fallback
        return build_spatial_edges(lats, lons, threshold_km=200.0)

    return torch.tensor([src, dst], dtype=torch.long)


# ---------------------------------------------------------------------------
# 5. Hyperedge construction  (paper: sliding-window hyperedges, multi-scale)
# ---------------------------------------------------------------------------

def build_hyperedges(
    dist_matrix: np.ndarray,
    thresholds_km: Tuple[float, ...] = (50.0, 100.0, 200.0),
) -> List[List[int]]:
    """
    Multi-scale spatial hyperedges (paper: E_s = e_s1 ∪ e_s2 ∪ ... ∪ e_sW).

    Each threshold defines one "scale":
      - Small threshold → tight local clusters
      - Large threshold → regional clusters

    Returns
    -------
    hyperedges : list of lists, each inner list is a set of node indices
    """
    N = len(dist_matrix)
    hyperedges: List[List[int]] = []
    seen: set = set()

    for thr in thresholds_km:
        adj = (dist_matrix < thr) & (dist_matrix > 0)      # (N, N) bool
        for i in range(N):
            members = tuple(sorted([i] + np.where(adj[i])[0].tolist()))
            if len(members) >= 2 and members not in seen:
                hyperedges.append(list(members))
                seen.add(members)

    # Guarantee at least one hyperedge per node
    covered = set(n for he in hyperedges for n in he)
    for i in range(N):
        if i not in covered:
            hyperedges.append([i])

    return hyperedges


def hyperedges_to_incidence(
    hyperedges: List[List[int]],
    n_nodes: int,
) -> Tensor:
    """
    Build sparse incidence matrix H: (n_nodes, n_hyperedges).
    H[i, e] = 1 if node i belongs to hyperedge e.

    Returns
    -------
    H : FloatTensor (n_nodes, n_hyperedges)
    """
    E = len(hyperedges)
    H = torch.zeros(n_nodes, E, dtype=torch.float32)
    for e_idx, nodes in enumerate(hyperedges):
        for n in nodes:
            H[n, e_idx] = 1.0
    return H


# ---------------------------------------------------------------------------
# 6. Region (category) nodes  (paper: LDA category nodes via sumpooling)
# ---------------------------------------------------------------------------

def build_region_membership(
    station_ids: List[str],
    region_map: Optional[Dict[str, List[str]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Assign each station to a geographic region (paper: LDA category).

    Returns
    -------
    membership : int ndarray (N,)  region index per station (-1 = unknown)
    region_names : list of str
    """
    if region_map is None:
        region_map = THAILAND_REGIONS

    region_names = list(region_map.keys())
    station_to_region: Dict[str, int] = {}
    for r_idx, (region, stations) in enumerate(region_map.items()):
        for s in stations:
            station_to_region[s] = r_idx

    membership = np.array(
        [station_to_region.get(sid, -1) for sid in station_ids],
        dtype=np.int64,
    )
    return membership, region_names


def compute_region_embeddings(
    node_embeddings: Tensor,      # (N, H)
    membership: np.ndarray,       # (N,)  region index, -1 = unknown
    n_regions: int,
) -> Tensor:
    """
    Compute region embeddings by averaging member station embeddings.
    (paper: h_ci = (1/k) Σ h_θj  for θj ∈ ci)

    Returns
    -------
    region_emb : FloatTensor (n_regions, H)  — zero for empty regions
    """
    H = node_embeddings.shape[-1]
    region_emb = torch.zeros(n_regions, H, device=node_embeddings.device)
    counts      = torch.zeros(n_regions, device=node_embeddings.device)

    for i, r in enumerate(membership):
        if 0 <= r < n_regions:
            region_emb[r] += node_embeddings[i]
            counts[r]      += 1

    nonzero = counts > 0
    region_emb[nonzero] /= counts[nonzero].unsqueeze(-1)
    return region_emb


# ---------------------------------------------------------------------------
# 7. Temporal graph  (paper: temporal heterogeneous graph)
# ---------------------------------------------------------------------------

def build_temporal_edges(
    n_nodes: int,
    seq_len: int,
) -> Dict[str, Tensor]:
    """
    Build two types of temporal edges for the temporal heterogeneous graph
    (paper: G_t = {V_t, E_t} — sequential + seasonal).

    Sequential edges : node i at time t → node i at time t+1  (self-edges across time)
    Seasonal edges   : same day-of-week positions across the lookback window

    Returns
    -------
    dict with keys 'sequential' and 'seasonal', each (2, E) LongTensor
    """
    # Sequential: (node, t) → (node, t+1)
    seq_src, seq_dst = [], []
    for n in range(n_nodes):
        for t in range(seq_len - 1):
            seq_src.append(n * seq_len + t)
            seq_dst.append(n * seq_len + t + 1)

    # Seasonal: same position mod 7 (weekly seasonality)
    sea_src, sea_dst = [], []
    for n in range(n_nodes):
        for t in range(seq_len):
            for t2 in range(t + 1, seq_len):
                if (t2 - t) % 7 == 0:
                    idx1 = n * seq_len + t
                    idx2 = n * seq_len + t2
                    sea_src += [idx1, idx2]
                    sea_dst += [idx2, idx1]

    return {
        "sequential": torch.tensor([seq_src, seq_dst], dtype=torch.long),
        "seasonal":   torch.tensor([sea_src, sea_dst], dtype=torch.long)
        if sea_src else torch.zeros(2, 0, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# 8.  Master graph builder  (single entry point)
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Builds and caches all graph structures needed by STC-HGAT.

    Usage
    -----
    gb = GraphBuilder(lats, lons, station_ids, pm25_history)
    graphs = gb.build(wind_u=wu, wind_v=wv)
    # graphs['spatial_edges'], graphs['hyperedges_incidence'], etc.
    """

    def __init__(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        station_ids: List[str],
        pm25_history: np.ndarray,
        spatial_thresholds_km: Tuple[float, ...] = (50.0, 100.0, 200.0),
        spatial_edge_km: float = 150.0,
        corr_threshold: float = 0.70,
        wind_top_k: int = 3,
        region_map: Optional[Dict[str, List[str]]] = None,
    ):
        self.lats = lats
        self.lons = lons
        self.station_ids  = station_ids
        self.pm25_history = pm25_history
        self.spatial_thresholds_km = spatial_thresholds_km
        self.spatial_edge_km  = spatial_edge_km
        self.corr_threshold   = corr_threshold
        self.wind_top_k       = wind_top_k
        self.region_map       = region_map or THAILAND_REGIONS

        self.N = len(lats)
        self._dist_matrix = pairwise_distance_matrix(lats, lons)
        self._membership, self._region_names = build_region_membership(
            station_ids, self.region_map
        )

    # ------------------------------------------------------------------
    def build(
        self,
        wind_u: Optional[np.ndarray] = None,
        wind_v: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Build all graph structures.

        Returns
        -------
        dict with keys:
          spatial_edges        (2, E) LongTensor
          semantic_edges       (2, E) LongTensor
          wind_edges           (2, E) LongTensor
          hyperedges           list[list[int]]
          hyperedges_incidence (N, E_h) FloatTensor
          membership           (N,) int ndarray
          region_names         list[str]
          n_regions            int
          dist_matrix          (N, N) float ndarray
        """
        spatial_ei  = build_spatial_edges(self.lats, self.lons, self.spatial_edge_km)
        semantic_ei = build_semantic_edges(self.pm25_history, self.corr_threshold)

        if wind_u is not None and wind_v is not None:
            wind_ei = build_wind_edges(wind_u, wind_v, self.lats, self.lons, self.wind_top_k)
        else:
            wind_ei = spatial_ei.clone()

        hyperedges = build_hyperedges(self._dist_matrix, self.spatial_thresholds_km)
        H_inc      = hyperedges_to_incidence(hyperedges, self.N)

        return {
            "spatial_edges":         spatial_ei,
            "semantic_edges":        semantic_ei,
            "wind_edges":            wind_ei,
            "hyperedges":            hyperedges,
            "hyperedges_incidence":  H_inc,          # (N, E_h)
            "membership":            self._membership,
            "region_names":          self._region_names,
            "n_regions":             len(self._region_names),
            "dist_matrix":           self._dist_matrix,
        }

    def n_hyperedges(self, graphs: Dict) -> int:
        return graphs["hyperedges_incidence"].shape[1]

    def summary(self, graphs: Dict) -> str:
        lines = [
            f"Nodes          : {self.N}",
            f"Spatial edges  : {graphs['spatial_edges'].shape[1]}",
            f"Semantic edges : {graphs['semantic_edges'].shape[1]}",
            f"Wind edges     : {graphs['wind_edges'].shape[1]}",
            f"Hyperedges     : {len(graphs['hyperedges'])}",
            f"Regions        : {graphs['n_regions']} ({', '.join(graphs['region_names'])})",
        ]
        return "\n".join(lines)
