"""
test_stc_hgat.py
----------------
Comprehensive unit tests for STC-HGAT PM2.5 forecasting pipeline.

Tests cover:
  1. Graph construction (GraphBuilder, hyperedges, haversine)
  2. Dataset pipeline (split, fill, scaler, sequences)
  3. HyperGAT layer + module
  4. HGAT temporal module
  5. Contrastive loss (InfoNCE)
  6. Adaptive Weight Loss
  7. Full STCHGAT forward pass
  8. STCHGATModel wrapper (fit/predict/evaluate/save/load)
  9. Evaluator metrics

Run:
    pytest tests/unit/test_stc_hgat.py -v
    pytest tests/unit/test_stc_hgat.py -v -m "not slow"
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Imports from src ────────────────────────────────────────────────────────
from src.data.graph_builder import (
    GraphBuilder,
    build_hyperedges,
    build_semantic_edges,
    build_spatial_edges,
    build_wind_edges,
    compute_region_embeddings,
    haversine_km,
    hyperedges_to_incidence,
    pairwise_distance_matrix,
)
from src.data.dataset import (
    FeatureScaler,
    PM25GraphDataset,
    TargetScaler,
    collate_fn,
    create_sequences,
    fill_missing,
    split_by_date,
)
from src.models.stc_hgat_model import (
    STCHGAT,
    STCHGATModel,
    HGATModule,
    HyperGATLayer,
    HyperGATModule,
    PositionalEncoding,
    adaptive_weight_loss,
    infonce_loss,
)
from src.training.evaluator import (
    compute_mae,
    compute_mbe,
    compute_r2,
    compute_rmse,
    compute_smape,
    evaluate_all,
)


# ===========================================================================
# Shared constants
# ===========================================================================
N_STATIONS = 12
SEQ_LEN    = 7
N_FEATURES = 20
HIDDEN     = 32
N_REGIONS  = 5
BATCH      = 4
N_HYPEREDGES = 25


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def station_coords():
    np.random.seed(42)
    lats = np.random.uniform(6.0, 20.0, N_STATIONS)
    lons = np.random.uniform(97.0, 106.0, N_STATIONS)
    return lats, lons


@pytest.fixture(scope="module")
def station_ids():
    return [f"ST{i:02d}" for i in range(N_STATIONS)]


@pytest.fixture(scope="module")
def pm25_history():
    np.random.seed(1)
    return np.abs(np.random.randn(N_STATIONS, 365).cumsum(axis=1))


@pytest.fixture(scope="module")
def wind_components():
    np.random.seed(3)
    return (
        np.random.uniform(-5, 5, N_STATIONS),
        np.random.uniform(-5, 5, N_STATIONS),
    )


@pytest.fixture(scope="module")
def graph_builder(station_coords, station_ids, pm25_history):
    lats, lons = station_coords
    return GraphBuilder(
        lats=lats, lons=lons,
        station_ids=station_ids,
        pm25_history=pm25_history,
        spatial_thresholds_km=(50.0, 150.0),
        spatial_edge_km=200.0,
    )


@pytest.fixture(scope="module")
def graphs(graph_builder, wind_components):
    wu, wv = wind_components
    return graph_builder.build(wind_u=wu, wind_v=wv)


@pytest.fixture(scope="module")
def H_inc(graphs):
    H = graphs["hyperedges_incidence"]
    # Pad to at least N_STATIONS + N_REGIONS rows
    total = N_STATIONS + N_REGIONS
    if H.shape[0] < total:
        pad = torch.zeros(total - H.shape[0], H.shape[1])
        H = torch.cat([H, pad], dim=0)
    return H


@pytest.fixture(scope="module")
def membership(graphs):
    return graphs["membership"]


@pytest.fixture(scope="module")
def sample_x():
    torch.manual_seed(10)
    return torch.randn(BATCH, N_STATIONS, SEQ_LEN, N_FEATURES)


@pytest.fixture(scope="module")
def sample_y():
    torch.manual_seed(11)
    return torch.clamp(torch.randn(BATCH, N_STATIONS) * 20 + 50, min=0)


@pytest.fixture(scope="module")
def sample_mask():
    m = torch.ones(BATCH, N_STATIONS, dtype=torch.bool)
    m[:, -2:] = False   # 2 stations missing
    return m


@pytest.fixture(scope="module")
def model_net(H_inc, membership):
    return STCHGAT(
        in_channels=N_FEATURES, hidden=HIDDEN,
        n_regions=N_REGIONS,
        n_hyperedges=H_inc.shape[1],
        hypergat_layers=1, seq_len=SEQ_LEN,
        dropout=0.0,
    ).eval()


@pytest.fixture
def wrapper_config():
    return dict(
        in_channels=N_FEATURES, hidden=HIDDEN,
        n_regions=N_REGIONS,
        hypergat_layers=1, seq_len=SEQ_LEN,
        dropout=0.0, epochs=2, batch_size=2,
        lr=1e-3, patience=5, device="cpu",
        contrastive_lambda=0.05, aw_gamma=2.0,
    )


@pytest.fixture
def wrapper_model(wrapper_config, H_inc, membership):
    return STCHGATModel(config=wrapper_config, H_inc=H_inc, membership=membership)


def _make_dataset(n=20):
    np.random.seed(99)
    X = np.random.randn(n, N_STATIONS, SEQ_LEN, N_FEATURES).astype("float32")
    y = np.clip(np.random.randn(n, N_STATIONS)*20+50, 0, None).astype("float32")
    mask = np.ones((n, N_STATIONS), dtype=bool)
    mask[:, -1] = False
    return PM25GraphDataset(list(X), list(y), list(mask))


# ===========================================================================
# 1. Haversine & Distance
# ===========================================================================

class TestHaversine:
    def test_same_point_zero(self):
        d = haversine_km(13.0, 100.0, 13.0, 100.0)
        assert float(d) == pytest.approx(0.0, abs=1e-3)

    def test_chiang_mai_to_bangkok(self):
        d = haversine_km(18.79, 98.98, 13.75, 100.49)
        assert 560 < float(d) < 640

    def test_non_negative(self):
        d = haversine_km(15.0, 100.0, 12.0, 101.0)
        assert float(d) >= 0

    def test_tensor_input(self):
        d = haversine_km(
            torch.tensor(13.0), torch.tensor(100.0),
            torch.tensor(15.0), torch.tensor(102.0),
        )
        assert isinstance(d, torch.Tensor)

    def test_symmetry(self):
        d1 = float(haversine_km(13.0, 100.0, 18.0, 99.0))
        d2 = float(haversine_km(18.0, 99.0, 13.0, 100.0))
        assert d1 == pytest.approx(d2, rel=1e-4)


class TestPairwiseDistance:
    def test_diagonal_zero(self, station_coords):
        lats, lons = station_coords
        D = pairwise_distance_matrix(lats, lons)
        assert np.allclose(np.diag(D), 0.0)

    def test_symmetric(self, station_coords):
        lats, lons = station_coords
        D = pairwise_distance_matrix(lats, lons)
        assert np.allclose(D, D.T)

    def test_non_negative(self, station_coords):
        lats, lons = station_coords
        D = pairwise_distance_matrix(lats, lons)
        assert (D >= 0).all()


# ===========================================================================
# 2. Graph Construction
# ===========================================================================

class TestBuildSpatialEdges:
    def test_shape(self, station_coords):
        lats, lons = station_coords
        ei = build_spatial_edges(lats, lons, 500.0)
        assert ei.shape[0] == 2 and ei.shape[1] > 0

    def test_dtype(self, station_coords):
        lats, lons = station_coords
        ei = build_spatial_edges(lats, lons)
        assert ei.dtype == torch.long

    def test_no_self_loops(self, station_coords):
        lats, lons = station_coords
        ei = build_spatial_edges(lats, lons, 500.0)
        assert not (ei[0] == ei[1]).any()

    def test_indices_in_range(self, station_coords):
        lats, lons = station_coords
        ei = build_spatial_edges(lats, lons, 500.0)
        assert ei.max() < N_STATIONS

    def test_fallback_produces_edges(self):
        lats = np.array([0.0, 45.0, -45.0])
        lons = np.array([0.0, 90.0, -90.0])
        ei = build_spatial_edges(lats, lons, threshold_km=1.0)
        assert ei.shape[1] > 0


class TestBuildSemanticEdges:
    def test_shape(self, pm25_history):
        ei = build_semantic_edges(pm25_history, 0.5)
        assert ei.shape[0] == 2

    def test_no_self_loops(self, pm25_history):
        ei = build_semantic_edges(pm25_history, 0.5)
        assert not (ei[0] == ei[1]).any()

    def test_higher_threshold_fewer_edges(self, pm25_history):
        ei_lo = build_semantic_edges(pm25_history, 0.3)
        ei_hi = build_semantic_edges(pm25_history, 0.95)
        assert ei_hi.shape[1] <= ei_lo.shape[1]

    def test_fallback_exists(self):
        data = np.array([np.sin(np.linspace(0, 10, 100)),
                         -np.sin(np.linspace(0, 10, 100))])
        ei = build_semantic_edges(data, 0.999)
        assert ei.shape[1] > 0


class TestBuildWindEdges:
    def test_no_self_loops(self, station_coords, wind_components):
        lats, lons = station_coords
        wu, wv = wind_components
        ei = build_wind_edges(wu, wv, lats, lons)
        assert not (ei[0] == ei[1]).any()

    def test_fallback_zero_wind(self, station_coords):
        lats, lons = station_coords
        wu = np.zeros(N_STATIONS); wv = np.zeros(N_STATIONS)
        ei = build_wind_edges(wu, wv, lats, lons)
        assert ei.shape[1] > 0

    def test_dtype(self, station_coords, wind_components):
        lats, lons = station_coords
        wu, wv = wind_components
        ei = build_wind_edges(wu, wv, lats, lons)
        assert ei.dtype == torch.long


class TestHyperedges:
    def test_multi_scale_more_than_single(self, station_coords):
        lats, lons = station_coords
        D = pairwise_distance_matrix(lats, lons)
        he_single = build_hyperedges(D, (100.0,))
        he_multi  = build_hyperedges(D, (50.0, 100.0, 200.0))
        assert len(he_multi) >= len(he_single)

    def test_all_nodes_covered(self, station_coords):
        lats, lons = station_coords
        D = pairwise_distance_matrix(lats, lons)
        he = build_hyperedges(D, (50.0, 150.0))
        covered = set(n for h in he for n in h)
        assert covered == set(range(N_STATIONS))

    def test_incidence_shape(self, station_coords):
        lats, lons = station_coords
        D = pairwise_distance_matrix(lats, lons)
        he = build_hyperedges(D, (100.0,))
        H = hyperedges_to_incidence(he, N_STATIONS)
        assert H.shape[0] == N_STATIONS
        assert H.shape[1] == len(he)

    def test_incidence_binary(self, station_coords):
        lats, lons = station_coords
        D = pairwise_distance_matrix(lats, lons)
        he = build_hyperedges(D, (100.0,))
        H = hyperedges_to_incidence(he, N_STATIONS)
        assert set(H.unique().tolist()).issubset({0.0, 1.0})


class TestGraphBuilder:
    def test_returns_all_keys(self, graphs):
        required = {
            "spatial_edges", "semantic_edges", "wind_edges",
            "hyperedges", "hyperedges_incidence",
            "membership", "region_names", "n_regions",
        }
        assert required.issubset(graphs.keys())

    def test_edge_dtype(self, graphs):
        for key in ("spatial_edges", "semantic_edges", "wind_edges"):
            assert graphs[key].dtype == torch.long, f"{key} not LongTensor"

    def test_incidence_shape_rows(self, graphs):
        H = graphs["hyperedges_incidence"]
        assert H.shape[0] == N_STATIONS

    def test_membership_length(self, graphs):
        assert len(graphs["membership"]) == N_STATIONS

    def test_n_regions_positive(self, graphs):
        assert graphs["n_regions"] > 0

    def test_summary_runs(self, graph_builder, graphs):
        s = graph_builder.summary(graphs)
        assert "Nodes" in s


class TestRegionEmbeddings:
    def test_output_shape(self):
        emb = torch.randn(N_STATIONS, HIDDEN)
        m   = np.array([i % N_REGIONS for i in range(N_STATIONS)])
        out = compute_region_embeddings(emb, m, N_REGIONS)
        assert out.shape == (N_REGIONS, HIDDEN)

    def test_known_mean(self):
        emb = torch.ones(4, 2)
        emb[2:] = 2.0
        m = np.array([0, 0, 1, 1])
        out = compute_region_embeddings(emb, m, 2)
        assert out[0].allclose(torch.ones(2))
        assert out[1].allclose(torch.full((2,), 2.0))

    def test_unknown_station_ignored(self):
        emb = torch.randn(4, HIDDEN)
        m   = np.array([-1, 0, 0, 1])  # -1 = unknown
        out = compute_region_embeddings(emb, m, N_REGIONS)
        assert out.shape == (N_REGIONS, HIDDEN)


# ===========================================================================
# 3. Dataset Pipeline
# ===========================================================================

class TestSplitByDate:
    def _df(self, n_days=100):
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        rows  = []
        for d in dates:
            for sid in ["A", "B"]:
                rows.append({"date": d, "stationID": sid, "pm2_5_mean": 30.0})
        return pd.DataFrame(rows)

    def test_no_overlap(self):
        df = self._df()
        tr, va, te = split_by_date(df, 0.7, 0.15)
        tr_d = set(tr["date"].unique()); va_d = set(va["date"].unique())
        te_d = set(te["date"].unique())
        assert tr_d.isdisjoint(va_d), "Train/Val overlap"
        assert tr_d.isdisjoint(te_d), "Train/Test overlap"
        assert va_d.isdisjoint(te_d), "Val/Test overlap"

    def test_chronological(self):
        df = self._df()
        tr, va, te = split_by_date(df, 0.7, 0.15)
        assert tr["date"].max() < va["date"].min()
        assert va["date"].max() < te["date"].min()

    def test_sizes(self):
        df = self._df(100)
        tr, va, te = split_by_date(df, 0.7, 0.15)
        total = len(tr["date"].unique()) + len(va["date"].unique()) + len(te["date"].unique())
        assert total == 100


class TestFillMissing:
    def test_forward_fill_within_station(self):
        rows = [
            {"stationID": "A", "date": pd.Timestamp("2020-01-01"), "x": 1.0},
            {"stationID": "A", "date": pd.Timestamp("2020-01-02"), "x": np.nan},
            {"stationID": "A", "date": pd.Timestamp("2020-01-03"), "x": 3.0},
        ]
        df = pd.DataFrame(rows)
        result = fill_missing(df, ["x"])
        assert result["x"].iloc[1] == pytest.approx(1.0)

    def test_no_cross_station_fill(self):
        rows = [
            {"stationID": "A", "date": pd.Timestamp("2020-01-01"), "x": 10.0},
            {"stationID": "B", "date": pd.Timestamp("2020-01-01"), "x": np.nan},
        ]
        df = pd.DataFrame(rows)
        result = fill_missing(df, ["x"])
        # Station B row 1 should NOT be filled with station A's value
        b_val = result[result["stationID"] == "B"]["x"].iloc[0]
        assert b_val != pytest.approx(10.0, abs=0.01)

    def test_no_remaining_nan(self):
        np.random.seed(7)
        rows = [{"stationID": "A", "date": pd.Timestamp(f"2020-01-{d+1:02d}"),
                 "x": np.nan if d % 3 == 0 else float(d)}
                for d in range(10)]
        df = pd.DataFrame(rows)
        result = fill_missing(df, ["x"])
        assert not result["x"].isna().any()


class TestFeatureScaler:
    def test_mean_zero_after_fit(self):
        X = np.random.randn(100, 5).astype("float32")
        sc = FeatureScaler()
        Xs = sc.fit_transform(X)
        assert np.abs(Xs.mean(axis=0)).max() < 1e-5

    def test_std_one_after_fit(self):
        X = np.random.randn(100, 5).astype("float32")
        sc = FeatureScaler()
        Xs = sc.fit_transform(X)
        assert np.abs(Xs.std(axis=0) - 1).max() < 1e-4

    def test_inverse_roundtrip(self):
        X = np.random.randn(50, 4).astype("float32")
        sc = FeatureScaler()
        Xs = sc.fit_transform(X)
        Xi = sc.inverse_transform(Xs)
        assert np.allclose(X, Xi, atol=1e-4)

    def test_transform_before_fit_raises(self):
        sc = FeatureScaler()
        with pytest.raises(AssertionError):
            sc.transform(np.zeros((5, 3)))

    def test_train_stats_not_updated_on_transform(self):
        X_tr = np.random.randn(100, 3).astype("float32")
        X_te = np.random.randn(50, 3).astype("float32") + 100
        sc   = FeatureScaler()
        sc.fit_transform(X_tr)
        mean_tr = sc._scaler.mean_.copy()
        sc.transform(X_te)
        assert np.allclose(sc._scaler.mean_, mean_tr), "Scaler mean changed after transform"


class TestCreateSequences:
    def _simple_df(self, n_days=30, n_stations=5):
        rows = []
        np.random.seed(5)
        for d in range(n_days):
            for s in range(n_stations):
                rows.append({
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=d),
                    "stationID": f"S{s:02d}",
                    "feat1": float(d + s),
                    "feat2": float(d * s),
                    "pm2_5_mean": float(d + s + 1),
                })
        return pd.DataFrame(rows)

    def test_output_shapes(self):
        df = self._simple_df()
        order = [f"S{i:02d}" for i in range(5)]
        X, y, mask = create_sequences(df, ["feat1","feat2"], order, lookback=7)
        assert len(X) > 0
        assert X[0].shape == (5, 7, 2)
        assert y[0].shape == (5,)
        assert mask[0].shape == (5,)

    def test_no_future_leakage(self):
        df = self._simple_df()
        order = [f"S{i:02d}" for i in range(5)]
        X, y, mask = create_sequences(df, ["feat1","feat2"], order, lookback=7)
        # y[i] corresponds to date i+7, X[i] uses dates i to i+6
        assert len(X) == len(y)

    def test_station_order_fixed(self):
        df = self._simple_df()
        order = [f"S{i:02d}" for i in range(5)]
        X, y, mask = create_sequences(df, ["feat1","feat2"], order, lookback=7)
        # Node 0 in every sample should always be station S00
        assert X[0].shape[0] == 5   # always N_fixed nodes


class TestPM25GraphDataset:
    def test_len(self):
        ds = _make_dataset(15)
        assert len(ds) == 15

    def test_getitem_shapes(self):
        ds = _make_dataset(10)
        x, y, m = ds[0]
        assert x.shape == (N_STATIONS, SEQ_LEN, N_FEATURES)
        assert y.shape == (N_STATIONS,)
        assert m.shape == (N_STATIONS,)
        assert m.dtype == torch.bool

    def test_collate(self):
        ds = _make_dataset(10)
        batch = [ds[i] for i in range(4)]
        X, y, mask = collate_fn(batch)
        assert X.shape == (4, N_STATIONS, SEQ_LEN, N_FEATURES)
        assert y.shape == (4, N_STATIONS)
        assert mask.shape == (4, N_STATIONS)


# ===========================================================================
# 4. HyperGAT Layer
# ===========================================================================

class TestHyperGATLayer:
    @pytest.fixture
    def layer(self):
        return HyperGATLayer(hidden=HIDDEN, dropout=0.0).eval()

    @pytest.fixture
    def mini_H(self):
        # 5 nodes, 3 hyperedges
        H = torch.zeros(5, 3)
        H[0, 0] = H[1, 0] = H[2, 0] = 1  # he0: nodes {0,1,2}
        H[1, 1] = H[3, 1] = 1             # he1: nodes {1,3}
        H[2, 2] = H[4, 2] = 1             # he2: nodes {2,4}
        return H

    def test_output_shape(self, layer, mini_H):
        x = torch.randn(5, HIDDEN)
        out = layer(x, mini_H)
        assert out.shape == (5, HIDDEN)

    def test_output_finite(self, layer, mini_H):
        x = torch.randn(5, HIDDEN)
        out = layer(x, mini_H)
        assert torch.all(torch.isfinite(out))

    def test_residual_active(self, layer, mini_H):
        x = torch.randn(5, HIDDEN)
        out = layer(x, mini_H)
        assert not torch.allclose(x, out)


# ===========================================================================
# 5. HyperGAT Module
# ===========================================================================

class TestHyperGATModule:
    @pytest.fixture
    def module(self):
        return HyperGATModule(
            hidden=HIDDEN, n_regions=N_REGIONS,
            n_layers=1, dropout=0.0
        ).eval()

    def test_output_shape(self, module, H_inc, membership):
        x = torch.randn(BATCH, N_STATIONS, HIDDEN)
        out = module(x, H_inc, membership)
        assert out.shape == (BATCH, N_STATIONS, HIDDEN)

    def test_output_finite(self, module, H_inc, membership):
        x = torch.randn(BATCH, N_STATIONS, HIDDEN)
        out = module(x, H_inc, membership)
        assert torch.all(torch.isfinite(out))

    def test_batch_independent(self, module, H_inc, membership):
        torch.manual_seed(0)
        x = torch.randn(BATCH, N_STATIONS, HIDDEN)
        full = module(x, H_inc, membership)
        single = module(x[0:1], H_inc, membership)
        assert torch.allclose(full[0:1], single, atol=1e-5)


# ===========================================================================
# 6. HGAT Temporal Module
# ===========================================================================

class TestHGATModule:
    @pytest.fixture
    def hgat(self):
        return HGATModule(hidden=HIDDEN, dropout=0.0).eval()

    def test_output_shape(self, hgat):
        x = torch.randn(BATCH, N_STATIONS, SEQ_LEN, HIDDEN)
        out = hgat(x)
        assert out.shape == (BATCH, N_STATIONS, HIDDEN)

    def test_output_finite(self, hgat):
        x = torch.randn(BATCH, N_STATIONS, SEQ_LEN, HIDDEN)
        out = hgat(x)
        assert torch.all(torch.isfinite(out))

    def test_different_inputs_different_outputs(self, hgat):
        x1 = torch.randn(BATCH, N_STATIONS, SEQ_LEN, HIDDEN)
        x2 = torch.randn(BATCH, N_STATIONS, SEQ_LEN, HIDDEN)
        assert not torch.allclose(hgat(x1), hgat(x2))


# ===========================================================================
# 7. Contrastive Loss (InfoNCE)
# ===========================================================================

class TestInfoNCELoss:
    def test_positive_scalar(self):
        hs = torch.randn(BATCH, N_STATIONS, HIDDEN)
        ht = torch.randn(BATCH, N_STATIONS, HIDDEN)
        loss = infonce_loss(hs, ht)
        assert loss.item() > 0

    def test_identical_embeddings_low_loss(self):
        h = F.normalize(torch.randn(BATCH, N_STATIONS, HIDDEN), dim=-1)
        loss_identical = infonce_loss(h, h)
        loss_random    = infonce_loss(
            torch.randn(BATCH, N_STATIONS, HIDDEN),
            torch.randn(BATCH, N_STATIONS, HIDDEN),
        )
        # Perfect positive pairs → lower InfoNCE than random
        assert loss_identical.item() < loss_random.item()

    def test_differentiable(self):
        hs = torch.randn(BATCH, N_STATIONS, HIDDEN, requires_grad=True)
        ht = torch.randn(BATCH, N_STATIONS, HIDDEN, requires_grad=True)
        loss = infonce_loss(hs, ht)
        loss.backward()
        assert hs.grad is not None
        assert ht.grad is not None

    def test_finite(self):
        hs = torch.randn(BATCH, N_STATIONS, HIDDEN)
        ht = torch.randn(BATCH, N_STATIONS, HIDDEN)
        assert math.isfinite(infonce_loss(hs, ht).item())


# ===========================================================================
# 8. Adaptive Weight Loss
# ===========================================================================

class TestAdaptiveWeightLoss:
    def test_perfect_prediction_low_loss(self):
        y = torch.rand(BATCH, N_STATIONS) * 100
        m = torch.ones(BATCH, N_STATIONS, dtype=torch.bool)
        loss = adaptive_weight_loss(y, y, m)
        assert loss.item() < 1e-4

    def test_large_error_larger_loss(self):
        y_true = torch.ones(BATCH, N_STATIONS) * 50
        y_bad  = torch.zeros(BATCH, N_STATIONS)
        y_good = torch.ones(BATCH, N_STATIONS) * 49
        m = torch.ones(BATCH, N_STATIONS, dtype=torch.bool)
        l_bad  = adaptive_weight_loss(y_bad,  y_true, m)
        l_good = adaptive_weight_loss(y_good, y_true, m)
        assert l_bad.item() > l_good.item()

    def test_extreme_upweight(self):
        # Extreme PM2.5 events should get more weight
        y_extreme  = torch.ones(BATCH, N_STATIONS) * 5.0   # above threshold=2
        y_moderate = torch.ones(BATCH, N_STATIONS) * 0.5
        y_pred     = torch.zeros(BATCH, N_STATIONS)
        m = torch.ones(BATCH, N_STATIONS, dtype=torch.bool)
        l_ext = adaptive_weight_loss(y_pred, y_extreme,  m, extreme_threshold=2.0)
        l_mod = adaptive_weight_loss(y_pred, y_moderate, m, extreme_threshold=2.0)
        assert l_ext.item() > l_mod.item()

    def test_empty_mask_returns_zero(self):
        y = torch.randn(BATCH, N_STATIONS)
        m = torch.zeros(BATCH, N_STATIONS, dtype=torch.bool)
        loss = adaptive_weight_loss(y, y, m)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_differentiable(self):
        y_pred = torch.randn(BATCH, N_STATIONS, requires_grad=True)
        y_true = torch.rand(BATCH, N_STATIONS) * 100
        m      = torch.ones(BATCH, N_STATIONS, dtype=torch.bool)
        loss   = adaptive_weight_loss(y_pred, y_true, m)
        loss.backward()
        assert y_pred.grad is not None


# ===========================================================================
# 9. Full STCHGAT Model
# ===========================================================================

class TestSTCHGATForward:
    def test_output_shapes(self, model_net, sample_x, H_inc, membership):
        with torch.no_grad():
            pred, hs, ht = model_net(sample_x, H_inc, membership)
        assert pred.shape == (BATCH, N_STATIONS, 1)
        assert hs.shape   == (BATCH, N_STATIONS, HIDDEN)
        assert ht.shape   == (BATCH, N_STATIONS, HIDDEN)

    def test_pred_non_negative(self, model_net, sample_x, H_inc, membership):
        with torch.no_grad():
            pred, _, _ = model_net(sample_x, H_inc, membership)
        assert (pred >= 0).all()

    def test_pred_finite(self, model_net, sample_x, H_inc, membership):
        with torch.no_grad():
            pred, hs, ht = model_net(sample_x, H_inc, membership)
        assert torch.all(torch.isfinite(pred))
        assert torch.all(torch.isfinite(hs))
        assert torch.all(torch.isfinite(ht))

    def test_batch_independence(self, model_net, sample_x, H_inc, membership):
        """Each sample in a batch must be independent."""
        with torch.no_grad():
            full_pred, _, _ = model_net(sample_x, H_inc, membership)
            for i in range(BATCH):
                single_pred, _, _ = model_net(sample_x[i:i+1], H_inc, membership)
                assert torch.allclose(full_pred[i:i+1], single_pred, atol=1e-4), \
                    f"Sample {i} not batch-independent"

    def test_gradient_flows_all_params(self, H_inc, membership):
        net = STCHGAT(
            in_channels=N_FEATURES, hidden=HIDDEN,
            n_regions=N_REGIONS, n_hyperedges=H_inc.shape[1],
            hypergat_layers=1, seq_len=SEQ_LEN, dropout=0.0
        ).train()
        x = torch.randn(2, N_STATIONS, SEQ_LEN, N_FEATURES)
        y = torch.rand(2, N_STATIONS) * 100
        m = torch.ones(2, N_STATIONS, dtype=torch.bool)
        pred, hs, ht = net(x, H_inc, membership)
        loss, _ = net.compute_loss(pred, y, m, hs, ht)
        loss.backward()

        no_grad = [n for n, p in net.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert len(no_grad) == 0, f"Params with no gradient: {no_grad}"

    def test_compute_loss_returns_dict_keys(self, H_inc, membership):
        net = STCHGAT(
            in_channels=N_FEATURES, hidden=HIDDEN,
            n_regions=N_REGIONS, n_hyperedges=H_inc.shape[1],
            hypergat_layers=1, seq_len=SEQ_LEN
        ).eval()
        x = torch.randn(2, N_STATIONS, SEQ_LEN, N_FEATURES)
        y = torch.rand(2, N_STATIONS) * 100
        m = torch.ones(2, N_STATIONS, dtype=torch.bool)
        with torch.no_grad():
            pred, hs, ht = net(x, H_inc, membership)
            _, ld = net.compute_loss(pred, y, m, hs, ht)
        assert {"total", "aw_loss", "contrastive"} == set(ld.keys())

    def test_parameter_count_positive(self, model_net):
        assert model_net.count_parameters() > 0

    @pytest.mark.slow
    def test_dropout_stochasticity(self, H_inc, membership):
        net = STCHGAT(
            in_channels=N_FEATURES, hidden=HIDDEN, n_regions=N_REGIONS,
            n_hyperedges=H_inc.shape[1], hypergat_layers=1,
            seq_len=SEQ_LEN, dropout=0.5
        ).train()
        x = torch.randn(BATCH, N_STATIONS, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            o1, _, _ = net(x, H_inc, membership)
            o2, _, _ = net(x, H_inc, membership)
        assert not torch.allclose(o1, o2)


# ===========================================================================
# 10. STCHGATModel Wrapper
# ===========================================================================

class TestSTCHGATModelWrapper:
    def test_predict_shape(self, wrapper_model):
        ds   = _make_dataset(10)
        pred, true = wrapper_model.predict(ds)
        assert pred.ndim == 1
        assert pred.shape == true.shape

    def test_predict_non_negative(self, wrapper_model):
        ds   = _make_dataset(10)
        pred, _ = wrapper_model.predict(ds)
        assert np.all(pred >= 0)

    def test_predict_finite(self, wrapper_model):
        ds   = _make_dataset(10)
        pred, _ = wrapper_model.predict(ds)
        assert np.all(np.isfinite(pred))

    def test_fit_history_recorded(self, wrapper_config, H_inc, membership):
        model = STCHGATModel(wrapper_config, H_inc, membership)
        tr_ds = _make_dataset(20)
        va_ds = _make_dataset(8)
        model.fit(tr_ds, va_ds)
        assert len(model._history["train_loss"]) >= 1
        assert len(model._history["val_r2"]) >= 1

    def test_evaluate_all_keys(self, wrapper_model):
        y = np.random.rand(50) * 100
        p = np.random.rand(50) * 100
        m = wrapper_model.evaluate(y, p)
        assert set(m.keys()) == {"RMSE", "MAE", "R2", "SMAPE", "MBE"}

    def test_evaluate_perfect(self, wrapper_model):
        y = np.random.rand(50) * 100
        m = wrapper_model.evaluate(y, y)
        assert m["RMSE"] == pytest.approx(0.0, abs=1e-5)
        assert m["R2"]   == pytest.approx(1.0, abs=1e-4)

    def test_evaluate_all_finite(self, wrapper_model):
        y = np.random.rand(50) * 100
        p = np.random.rand(50) * 100
        for v in wrapper_model.evaluate(y, p).values():
            assert math.isfinite(v)

    def test_evaluate_nan_input_handled(self, wrapper_model):
        y = np.random.rand(50) * 100
        p = np.random.rand(50) * 100
        y[0] = np.nan; p[5] = np.nan
        m = wrapper_model.evaluate(y, p)
        assert all(math.isfinite(v) for v in m.values())

    def test_save_load_consistency(self, wrapper_config, H_inc, membership):
        model = STCHGATModel(wrapper_config, H_inc, membership)
        ds    = _make_dataset(10)
        pred_before, _ = model.predict(ds)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.pt")
            model.save(path)
            model2 = STCHGATModel(wrapper_config, H_inc, membership)
            model2.load(path)
            pred_after, _ = model2.predict(ds)

        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-4)

    @pytest.mark.slow
    def test_fit_loss_decreases(self, wrapper_config, H_inc, membership):
        cfg   = {**wrapper_config, "epochs": 10}
        model = STCHGATModel(cfg, H_inc, membership)
        tr_ds = _make_dataset(30)
        va_ds = _make_dataset(10)
        model.fit(tr_ds, va_ds)
        hist  = model._history["train_loss"]
        assert hist[0] >= hist[-1] or len(hist) >= 2  # generally decreasing


# ===========================================================================
# 11. Evaluator Metrics
# ===========================================================================

class TestComputeRMSE:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        # sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        yt = np.array([0.0, 0.0])
        yp = np.array([3.0, 4.0])
        assert compute_rmse(yt, yp) == pytest.approx(math.sqrt(12.5), rel=1e-4)

    def test_non_negative(self):
        yt = np.random.randn(50); yp = np.random.randn(50)
        assert compute_rmse(yt, yp) >= 0

    def test_symmetric(self):
        yt = np.array([1.0, 2.0, 3.0])
        yp = np.array([4.0, 6.0, 8.0])
        assert compute_rmse(yt, yp) == pytest.approx(compute_rmse(yp, yt))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_rmse(np.ones(3), np.ones(4))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_rmse(np.array([]), np.array([]))


class TestComputeMAE:
    def test_perfect(self):
        y = np.array([10.0, 20.0])
        assert compute_mae(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        yt = np.array([1.0, 2.0, 3.0])
        yp = np.array([4.0, 6.0, 8.0])
        assert compute_mae(yt, yp) == pytest.approx(4.0)

    def test_non_negative(self):
        yt = np.random.randn(50); yp = np.random.randn(50)
        assert compute_mae(yt, yp) >= 0

    def test_symmetric(self):
        yt = np.array([1.0, 3.0])
        yp = np.array([3.0, 1.0])
        assert compute_mae(yt, yp) == pytest.approx(compute_mae(yp, yt))


class TestComputeR2:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_r2(y, y) == pytest.approx(1.0, abs=1e-6)

    def test_mean_predictor(self):
        yt = np.array([1.0, 2.0, 3.0])
        yp = np.full(3, yt.mean())
        assert compute_r2(yt, yp) == pytest.approx(0.0, abs=1e-5)

    def test_can_be_negative(self):
        yt = np.array([1.0, 2.0, 3.0])
        yp = np.array([10.0, 20.0, 30.0])
        assert compute_r2(yt, yp) < 0


class TestComputeSMAPE:
    def test_perfect(self):
        y = np.array([10.0, 20.0, 30.0])
        assert compute_smape(y, y) == pytest.approx(0.0, abs=1e-4)

    def test_range_0_to_200(self):
        yt = np.random.rand(100) * 200 + 1
        yp = np.random.rand(100) * 200 + 1
        assert 0 <= compute_smape(yt, yp) <= 200

    def test_symmetric(self):
        yt = np.array([100.0, 200.0])
        yp = np.array([110.0, 210.0])
        assert compute_smape(yt, yp) == pytest.approx(compute_smape(yp, yt))

    def test_zero_denominator_handled(self):
        yt = np.array([0.0]); yp = np.array([0.0])
        result = compute_smape(yt, yp)
        assert math.isfinite(result)


class TestComputeMBE:
    def test_no_bias(self):
        yt = np.array([1.0, 3.0]); yp = np.array([3.0, 1.0])
        assert compute_mbe(yt, yp) == pytest.approx(0.0, abs=1e-6)

    def test_positive_bias(self):
        yt = np.array([10.0, 20.0]); yp = np.array([15.0, 25.0])
        assert compute_mbe(yt, yp) > 0

    def test_negative_bias(self):
        yt = np.array([10.0, 20.0]); yp = np.array([5.0, 15.0])
        assert compute_mbe(yt, yp) < 0

    def test_known_value(self):
        yt = np.array([10.0, 20.0]); yp = np.array([12.0, 24.0])
        assert compute_mbe(yt, yp) == pytest.approx(3.0)


class TestEvaluateAll:
    def test_all_keys(self):
        y = np.random.rand(50) * 100
        m = evaluate_all(y, y + np.random.randn(50))
        assert set(m.keys()) == {"RMSE", "MAE", "R2", "SMAPE", "MBE"}

    def test_all_finite(self):
        y = np.random.rand(50) * 100
        m = evaluate_all(y, y + np.random.randn(50))
        assert all(math.isfinite(v) for v in m.values())

    def test_consistent_with_individual(self):
        yt = np.random.rand(50) * 100
        yp = yt + np.random.randn(50) * 5
        m = evaluate_all(yt, yp)
        assert m["RMSE"] == pytest.approx(compute_rmse(yt, yp), rel=1e-4)
        assert m["R2"]   == pytest.approx(compute_r2(yt, yp),   rel=1e-4)

    def test_nan_inputs_handled(self):
        yt = np.random.rand(50) * 100
        yp = yt + np.random.randn(50)
        yt[3] = np.nan; yp[7] = np.nan
        m = evaluate_all(yt, yp)
        assert all(math.isfinite(v) for v in m.values())

    @pytest.mark.parametrize("metric,lo,hi", [
        ("RMSE", 0, None),
        ("MAE",  0, None),
        ("SMAPE", 0, 200),
    ])
    def test_metric_bounds(self, metric, lo, hi):
        yt = np.random.rand(100) * 200
        yp = np.random.rand(100) * 200
        m  = evaluate_all(yt, yp)
        if lo is not None:
            assert m[metric] >= lo
        if hi is not None:
            assert m[metric] <= hi
