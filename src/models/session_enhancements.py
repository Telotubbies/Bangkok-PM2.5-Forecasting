"""
session_enhancements.py
-----------------------
Session-based enhancements for STC-HGAT model

Addresses missing session-based concepts from the paper:
1. Session type embeddings (weekday/weekend/holiday/fire_season)
2. Daily session boundaries
3. Cross-window attention for long-range dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import numpy as np


class SessionTypeEmbedding(nn.Module):
    """
    Learnable embeddings for different session types
    
    Session Types:
    0 - Weekday (Mon-Fri, normal pollution)
    1 - Weekend (Sat-Sun, different traffic patterns)
    2 - Holiday (public holidays, reduced activity)
    3 - Fire Season (March-April, high fire activity)
    """
    
    def __init__(self, hidden_dim: int, num_session_types: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_session_types = num_session_types
        
        # Learnable session type embeddings
        self.session_emb = nn.Embedding(num_session_types, hidden_dim)
        
        # Initialize
        nn.init.normal_(self.session_emb.weight, std=0.02)
    
    def forward(self, session_types: Tensor) -> Tensor:
        """
        Parameters
        ----------
        session_types : (B,) - Session type index for each batch
        
        Returns
        -------
        embeddings : (B, H) - Session type embeddings
        """
        return self.session_emb(session_types)


class DailySessionBoundary(nn.Module):
    """
    Model daily session boundaries explicitly
    
    Treats each day as a "session" with:
    - Session start embedding
    - Session end embedding
    - Within-session position encoding
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Session boundary markers
        self.session_start = nn.Parameter(torch.randn(hidden_dim))
        self.session_end = nn.Parameter(torch.randn(hidden_dim))
        
        # Position within session (0-23 for hours)
        self.position_emb = nn.Embedding(24, hidden_dim)
        
        nn.init.normal_(self.session_start, std=0.02)
        nn.init.normal_(self.session_end, std=0.02)
    
    def forward(self, node_emb: Tensor, hour_of_day: Tensor) -> Tensor:
        """
        Parameters
        ----------
        node_emb : (B, N, T, H) - Node embeddings
        hour_of_day : (B, T) - Hour of day (0-23) for each timestep
        
        Returns
        -------
        enhanced_emb : (B, N, T, H) - Enhanced with session boundaries
        """
        B, N, T, H = node_emb.shape
        
        # Add position within session
        pos_emb = self.position_emb(hour_of_day)  # (B, T, H)
        pos_emb = pos_emb.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, T, H)
        
        # Add session start marker at hour 0
        start_mask = (hour_of_day == 0).float()  # (B, T)
        start_mask = start_mask.unsqueeze(1).unsqueeze(-1).expand(-1, N, -1, H)  # (B, N, T, H)
        start_emb = self.session_start.view(1, 1, 1, H).expand(B, N, T, H)
        
        # Add session end marker at hour 23
        end_mask = (hour_of_day == 23).float()  # (B, T)
        end_mask = end_mask.unsqueeze(1).unsqueeze(-1).expand(-1, N, -1, H)  # (B, N, T, H)
        end_emb = self.session_end.view(1, 1, 1, H).expand(B, N, T, H)
        
        # Combine
        enhanced = node_emb + pos_emb + (start_mask * start_emb) + (end_mask * end_emb)
        
        return enhanced


class CrossWindowAttention(nn.Module):
    """
    Cross-window attention for long-range dependencies
    
    Allows current window to attend to previous window
    Captures patterns that span across session boundaries
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm and feedforward
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        current_window: Tensor, 
        previous_window: Optional[Tensor] = None
    ) -> Tensor:
        """
        Parameters
        ----------
        current_window : (B, N, T, H) - Current time window
        previous_window : (B, N, T, H) or None - Previous time window
        
        Returns
        -------
        enhanced : (B, N, T, H) - Enhanced with cross-window context
        """
        if previous_window is None:
            return current_window
        
        B, N, T, H = current_window.shape
        
        # Reshape for attention: (B*N, T, H)
        curr = current_window.view(B * N, T, H)
        prev = previous_window.view(B * N, T, H)
        
        # Cross attention: current attends to previous
        attn_out, _ = self.cross_attn(
            query=curr,
            key=prev,
            value=prev
        )  # (B*N, T, H)
        
        # Residual + norm
        curr = self.norm1(curr + attn_out)
        
        # Feedforward
        ffn_out = self.ffn(curr)
        curr = self.norm2(curr + ffn_out)
        
        # Reshape back
        enhanced = curr.view(B, N, T, H)
        
        return enhanced


class SessionEnhancedHGAT(nn.Module):
    """
    Enhanced HGAT with session-based improvements
    
    Combines:
    1. Session type embeddings
    2. Daily session boundaries
    3. Cross-window attention
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_session_types: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_session_types: bool = True,
        use_session_boundaries: bool = True,
        use_cross_window: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_session_types = use_session_types
        self.use_session_boundaries = use_session_boundaries
        self.use_cross_window = use_cross_window
        
        # Session type embeddings
        if use_session_types:
            self.session_type_emb = SessionTypeEmbedding(hidden_dim, num_session_types)
        
        # Daily session boundaries
        if use_session_boundaries:
            self.session_boundary = DailySessionBoundary(hidden_dim)
        
        # Cross-window attention
        if use_cross_window:
            self.cross_window_attn = CrossWindowAttention(hidden_dim, num_heads, dropout)
        
        # Original HGAT components (from stc_hgat_model.py)
        self.v_is = nn.Parameter(torch.empty(hidden_dim))
        self.v_si = nn.Parameter(torch.empty(hidden_dim))
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        
        nn.init.normal_(self.v_is, std=0.02)
        nn.init.normal_(self.v_si, std=0.02)
    
    def forward(
        self,
        node_time_emb: Tensor,
        session_types: Optional[Tensor] = None,
        hour_of_day: Optional[Tensor] = None,
        previous_window: Optional[Tensor] = None
    ) -> Tensor:
        """
        Parameters
        ----------
        node_time_emb : (B, N, T, H) - Temporal node embeddings
        session_types : (B,) - Session type for each batch
        hour_of_day : (B, T) - Hour of day (0-23) for each timestep
        previous_window : (B, N, T, H) - Previous time window for cross-attention
        
        Returns
        -------
        temporal_emb : (B, N, H) - Aggregated temporal embeddings
        """
        B, N, T, H = node_time_emb.shape
        
        # Enhancement 1: Add session type embeddings
        if self.use_session_types and session_types is not None:
            session_emb = self.session_type_emb(session_types)  # (B, H)
            session_emb = session_emb.view(B, 1, 1, H).expand(-1, N, T, -1)
            node_time_emb = node_time_emb + session_emb
        
        # Enhancement 2: Add session boundaries
        if self.use_session_boundaries and hour_of_day is not None:
            node_time_emb = self.session_boundary(node_time_emb, hour_of_day)
        
        # Enhancement 3: Cross-window attention
        if self.use_cross_window and previous_window is not None:
            node_time_emb = self.cross_window_attn(node_time_emb, previous_window)
        
        # Original HGAT logic (from stc_hgat_model.py:271-295)
        # Stage 1: Items → Session
        h0 = node_time_emb.mean(dim=2)  # (B, N, H)
        h0_exp = h0.unsqueeze(2).expand(-1, -1, T, -1)
        
        e_is = F.leaky_relu((node_time_emb * h0_exp * self.v_is).sum(-1))
        beta_is = F.softmax(e_is, dim=2)
        h_s1 = (beta_is.unsqueeze(-1) * node_time_emb).sum(2)
        
        # Stage 2: Session → Items
        h_s1_exp = h_s1.unsqueeze(2).expand(-1, -1, T, -1)
        e_si = F.leaky_relu((node_time_emb * h_s1_exp * self.v_si).sum(-1))
        beta_si = F.softmax(e_si, dim=2)
        h_t1 = (beta_si.unsqueeze(-1) * node_time_emb).sum(2)
        
        return self.norm(self.drop(h_t1) + h0)


def get_session_type(timestamps: np.ndarray) -> np.ndarray:
    """
    Determine session type for each timestamp
    
    Parameters
    ----------
    timestamps : array of datetime64 - Timestamps
    
    Returns
    -------
    session_types : array of int - Session type indices
        0 - Weekday
        1 - Weekend
        2 - Holiday
        3 - Fire Season (March-April)
    """
    import pandas as pd
    
    # Convert to pandas datetime
    dt = pd.to_datetime(timestamps)
    
    # Initialize as weekday
    session_types = np.zeros(len(dt), dtype=np.int64)
    
    # Weekend (Sat=5, Sun=6)
    weekend_mask = dt.dayofweek >= 5
    session_types[weekend_mask] = 1
    
    # Fire season (March-April)
    fire_season_mask = (dt.month >= 3) & (dt.month <= 4)
    session_types[fire_season_mask] = 3
    
    # TODO: Add holiday detection
    # For now, holidays are not implemented
    
    return session_types


def get_hour_of_day(timestamps: np.ndarray) -> np.ndarray:
    """
    Extract hour of day from timestamps
    
    Parameters
    ----------
    timestamps : array of datetime64 - Timestamps
    
    Returns
    -------
    hours : array of int - Hour of day (0-23)
    """
    import pandas as pd
    
    dt = pd.to_datetime(timestamps)
    return dt.hour.values
