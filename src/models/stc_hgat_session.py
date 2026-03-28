"""
stc_hgat_session.py
-------------------
STC-HGAT with Session-Based Enhancements

Extends ImprovedSTCHGAT with:
1. Session type embeddings (weekday/weekend/holiday/fire_season)
2. Daily session boundaries
3. Cross-window attention
4. All Phase 3 improvements (gated fusion, cross-attention, multi-scale temporal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import numpy as np

from src.models.stc_hgat_model import (
    HyperGATModule, PositionalEncoding,
    infonce_loss
)
from src.models.stc_hgat_improved import (
    GatedFusion, CrossAttentionFusion, MultiScaleTemporalBlock
)
from src.models.session_enhancements import SessionEnhancedHGAT


class SessionSTCHGAT(nn.Module):
    """
    STC-HGAT with Session-Based Enhancements
    
    Combines:
    - Base STC-HGAT architecture (HyperGAT + HGAT + Contrastive)
    - Phase 3 improvements (Gated Fusion + Cross-Attention + Multi-Scale)
    - Session enhancements (Session Types + Boundaries + Cross-Window)
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_stations: int = 79,
        num_regions: int = 5,
        num_hypergat_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        forecast_horizons: list = [1, 3, 7],
        # Phase 3 improvements
        use_gated_fusion: bool = True,
        use_cross_attention: bool = True,
        use_multiscale_temporal: bool = True,
        # Session enhancements
        use_session_types: bool = True,
        use_session_boundaries: bool = True,
        use_cross_window: bool = True,
        num_session_types: int = 4
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_stations = num_stations
        self.num_regions = num_regions
        self.forecast_horizons = forecast_horizons
        
        # Config flags
        self.use_gated_fusion = use_gated_fusion
        self.use_cross_attention = use_cross_attention
        self.use_multiscale_temporal = use_multiscale_temporal
        self.use_session_types = use_session_types
        self.use_session_boundaries = use_session_boundaries
        self.use_cross_window = use_cross_window
        
        # Feature embedding
        self.feature_embed = nn.Linear(num_features, hidden_dim)
        
        # Spatial module: HyperGAT
        self.hypergat = HyperGATModule(
            hidden=hidden_dim,
            n_regions=num_regions,
            n_layers=num_hypergat_layers,
            dropout=dropout
        )
        
        # Temporal module: Session-Enhanced HGAT
        self.hgat = SessionEnhancedHGAT(
            hidden_dim=hidden_dim,
            num_session_types=num_session_types,
            num_heads=num_heads,
            dropout=dropout,
            use_session_types=use_session_types,
            use_session_boundaries=use_session_boundaries,
            use_cross_window=use_cross_window
        )
        
        # Multi-scale temporal (Phase 3)
        if use_multiscale_temporal:
            self.multiscale_temporal = MultiScaleTemporalBlock(
                hidden_dim=hidden_dim,
                scales=[1, 3, 7]
            )
        
        # Fusion module
        if use_gated_fusion:
            self.fusion = GatedFusion(hidden_dim)
        
        # Cross-attention between spatial and temporal (Phase 3)
        if use_cross_attention:
            self.cross_attn = CrossAttentionFusion(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Position encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=30)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: Tensor,
        H_inc: Optional[Tensor] = None,
        membership: Optional[np.ndarray] = None,
        session_types: Optional[Tensor] = None,
        hour_of_day: Optional[Tensor] = None,
        previous_window: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x : (B, N, T, F) - Input features
        H_inc : Hypergraph incidence matrix (optional)
        membership : Region membership (optional)
        session_types : (B,) - Session type for each batch
        hour_of_day : (B, T) - Hour of day for each timestep
        previous_window : (B, N, T, F) - Previous window for cross-attention
        
        Returns
        -------
        pred : (B, N) - Predictions
        h_spatial : (B, N, H) - Spatial embeddings
        h_temporal : (B, N, H) - Temporal embeddings
        """
        B, N, T, F = x.shape
        
        # Feature embedding
        h = self.feature_embed(x)  # (B, N, T, H)
        
        # Multi-scale temporal processing (Phase 3)
        if self.use_multiscale_temporal:
            h = self.multiscale_temporal(h)  # (B, N, T, H)
        
        # Spatial processing: HyperGAT
        h_spatial_list = []
        for b in range(B):
            h_b = h[b].mean(dim=1)  # (N, H) - mean over time
            
            if H_inc is not None and membership is not None:
                h_spatial_b = self.hypergat(h_b, H_inc, membership)
            else:
                h_spatial_b = h_b
            
            h_spatial_list.append(h_spatial_b)
        
        h_spatial = torch.stack(h_spatial_list, dim=0)  # (B, N, H)
        
        # Temporal processing: Session-Enhanced HGAT
        # Prepare previous window embeddings if using cross-window attention
        prev_emb = None
        if self.use_cross_window and previous_window is not None:
            prev_emb = self.feature_embed(previous_window)  # (B, N, T, H)
            if self.use_multiscale_temporal:
                prev_emb = self.multiscale_temporal(prev_emb)
        
        h_temporal = self.hgat(
            h,
            session_types=session_types,
            hour_of_day=hour_of_day,
            previous_window=prev_emb
        )  # (B, N, H)
        
        # Fusion
        if self.use_gated_fusion:
            h_fused = self.fusion(h_spatial, h_temporal)
        else:
            h_fused = h_spatial + h_temporal
        
        # Cross-attention (Phase 3)
        if self.use_cross_attention:
            h_fused = self.cross_attn(h_spatial, h_temporal, h_temporal)
        
        # Position encoding - reshape h_fused to (B, N, T, H) for PositionalEncoding
        # h_fused is (B, N, H), need to add time dimension
        B, N, H_dim = h_fused.shape
        h_fused_expanded = h_fused.unsqueeze(2).expand(B, N, h.shape[2], H_dim)  # (B, N, T, H)
        h_final = self.pos_encoding(h_fused_expanded)  # (B, N, H)
        
        # Output projection
        pred = self.output_proj(h_final).squeeze(-1)  # (B, N)
        
        return pred, h_spatial, h_temporal
    
    def compute_loss(
        self,
        pred: Tensor,
        y: Tensor,
        h_spatial: Tensor,
        h_temporal: Tensor,
        lambda_contrastive: float = 0.1,
        temperature: float = 0.1
    ) -> Tuple[Tensor, dict]:
        """
        Compute combined loss: MSE + Contrastive
        
        Parameters
        ----------
        pred : (B, N) - Predictions
        y : (B, N) - Targets
        h_spatial : (B, N, H) - Spatial embeddings
        h_temporal : (B, N, H) - Temporal embeddings
        lambda_contrastive : Weight for contrastive loss
        temperature : Temperature for InfoNCE
        
        Returns
        -------
        total_loss : Combined loss
        loss_dict : Dictionary with loss components
        """
        # Reconstruction loss
        l_r = F.mse_loss(pred, y)
        
        # Contrastive loss
        l_c = infonce_loss(h_spatial, h_temporal, temperature=temperature)
        
        # Combined loss
        total_loss = l_r + lambda_contrastive * l_c
        
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': l_r.item(),
            'contrastive': l_c.item()
        }
        
        return total_loss, loss_dict
