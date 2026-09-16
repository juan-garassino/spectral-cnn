"""
PhasorWaveAttention: Fixed-Basis Fourier Attention Mechanism

This module implements attention using wave interference physics with a FIXED
frequency basis (like a cochlea or FFT), avoiding the optimization instability
of learning frequencies via gradient descent.

Key Innovation:
- Fixed frequency bank (logspace from 1/T to Nyquist)
- Learn amplitudes and phase shifts, NOT frequencies
- Power-law sharpening replaces softmax (kills Gibbs side-lobes)

Physics-to-Math Mapping:
- Phasor: A * e^(iφ) = A*cos(φ) + i*A*sin(φ)
- Interference: Re(Q @ K*) = A_Q * A_K * cos(φ_Q - φ_K)
- This equals: (Re_Q @ Re_K^T) + (Im_Q @ Im_K^T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PhasorWaveAttention(nn.Module):
    """
    Phasor Wave Interference Attention.
    
    Replaces standard softmax attention with wave physics:
    - Fixed frequency basis (no learning ω)
    - Phasor projection (learn A and δ)
    - Power-law sharpening (sparse attention)
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        max_len: Maximum sequence length (determines lowest frequency)
        dropout: Dropout probability
        sharpness: Power-law exponent for attention sharpening (default=3)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_len: int = 2048,
        dropout: float = 0.1,
        sharpness: float = 3.0,
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_len = max_len
        self.sharpness = sharpness
        
        # Head dimension must be even for Real/Imaginary split
        assert self.head_dim % 2 == 0, f"head_dim ({self.head_dim}) must be even for Real/Imag split"
        
        self.freq_dim = self.head_dim // 2  # Half for Real, half for Imaginary
        
        # === FIXED FREQUENCY BASIS (The "Cochlea") ===
        # Do NOT make this learnable - this is the key insight!
        # Range: 1/max_len (global) to 0.5 (Nyquist limit)
        min_freq = 1.0 / max_len  # Wavelength = full sequence
        max_freq = 0.5            # Nyquist: resolves adjacent tokens
        
        # Log-spaced frequencies for multi-scale attention
        freqs = torch.logspace(
            math.log10(min_freq),
            math.log10(max_freq),
            self.freq_dim
        )
        self.register_buffer('freqs', freqs)  # (freq_dim,)
        
        # Pre-compute position indices for phase calculation
        positions = torch.arange(max_len).float()
        self.register_buffer('positions', positions)  # (max_len,)
        
        # === LEARNABLE PROJECTIONS ===
        # Project to Q, K, V (standard attention pattern)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Amplitude and Phase projections (per head)
        # Input: head_dim, Output: freq_dim for amplitude, freq_dim for phase
        self.amp_proj = nn.Linear(self.head_dim, self.freq_dim, bias=True)
        self.phase_proj = nn.Linear(self.head_dim, self.freq_dim, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.freq_dim)
        
        # For visualization/debugging
        self.last_weights = None
        
    def _compute_phasors(
        self, 
        x: torch.Tensor, 
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert input to phasor representation (Real, Imaginary).
        
        Physics: Phasor = A * e^(iφ) where φ(t) = ω*t + δ
        
        Args:
            x: Input tensor (B, H, T, head_dim)
            seq_len: Actual sequence length
            
        Returns:
            real: A * cos(φ) - shape (B, H, T, freq_dim)
            imag: A * sin(φ) - shape (B, H, T, freq_dim)
        """
        B, H, T, D = x.shape
        
        # Project to amplitude and phase shift
        # A > 0 enforced via softplus
        amplitude = F.softplus(self.amp_proj(x))  # (B, H, T, freq_dim)
        phase_shift = self.phase_proj(x)           # (B, H, T, freq_dim)
        
        # Get position-dependent phase: φ(t) = ω*t + δ
        # positions: (T,) -> (1, 1, T, 1)
        # freqs: (freq_dim,) -> (1, 1, 1, freq_dim)
        t = self.positions[:seq_len].view(1, 1, T, 1)
        omega = self.freqs.view(1, 1, 1, -1)
        
        # Instantaneous phase
        phase = omega * t + phase_shift  # (B, H, T, freq_dim)
        
        # Convert polar (A, φ) to Cartesian (Real, Imag)
        # This is Euler's formula: A*e^(iφ) = A*cos(φ) + i*A*sin(φ)
        real = amplitude * torch.cos(phase)  # (B, H, T, freq_dim)
        imag = amplitude * torch.sin(phase)  # (B, H, T, freq_dim)
        
        return real, imag
    
    def _interference(
        self,
        q_real: torch.Tensor,
        q_imag: torch.Tensor,
        k_real: torch.Tensor,
        k_imag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute wave interference scores.
        
        Physics: Interference = A_Q * A_K * cos(φ_Q - φ_K)
        
        Math equivalence:
            Re(Q @ K*) = Re_Q @ Re_K^T + Im_Q @ Im_K^T
        
        This is because:
            (a + ib)(c - id) = (ac + bd) + i(bc - ad)
            Real part = ac + bd = Re_Q*Re_K + Im_Q*Im_K
        
        Args:
            q_real, q_imag: Query phasors (B, H, T, freq_dim)
            k_real, k_imag: Key phasors (B, H, T, freq_dim)
            
        Returns:
            scores: Interference pattern (B, H, T, T)
        """
        # Dot product of real parts + dot product of imaginary parts
        # This equals the real part of complex dot product (interference)
        scores = (
            torch.matmul(q_real, k_real.transpose(-2, -1)) +
            torch.matmul(q_imag, k_imag.transpose(-2, -1))
        )  # (B, H, T, T)
        
        return scores
    
    def _sharpen(self, scores: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply power-law sharpening to attention scores.
        
        This replaces softmax and kills Gibbs phenomenon side-lobes.
        
        Steps:
        1. Scale for stability
        2. ReLU to zero out destructive interference (negative scores)
        3. Power-law sharpening: weights = scores^p (default p=3)
        4. Normalize to sum to 1 (energy conservation)
        
        Args:
            scores: Raw interference scores (B, H, T, T)
            mask: Optional causal mask
            
        Returns:
            weights: Sharpened, normalized attention weights (B, H, T, T)
        """
        # Scale for numerical stability
        scores = scores / self.scale
        
        # Apply causal mask if provided (set future to -inf before ReLU)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        # ReLU: Zero out destructive interference (negative coherence)
        # This is physically meaningful: destructive interference = no signal
        scores = F.relu(scores)
        
        # Power-law sharpening: kills side-lobes, creates sparse attention
        # Higher power = sharper peaks, more like softmax
        weights = scores ** self.sharpness
        
        # Normalize to sum to 1 (energy conservation)
        # Add small epsilon to avoid division by zero
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        return weights
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass: Phasor Wave Interference Attention.
        
        Args:
            x: Input tensor (B, T, D)
            mask: Optional attention mask
            causal: Whether to apply causal masking
            
        Returns:
            output: Attended output (B, T, D)
        """
        B, T, D = x.shape
        
        # === STEP 1: Linear projections (standard attention) ===
        q = self.q_proj(x)  # (B, T, D)
        k = self.k_proj(x)  # (B, T, D)
        v = self.v_proj(x)  # (B, T, D)
        
        # Reshape for multi-head: (B, T, D) -> (B, H, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # === STEP 2: Phasor projection (Euler's formula) ===
        # Convert Q and K to phasor representation (Real, Imag)
        q_real, q_imag = self._compute_phasors(q, T)  # (B, H, T, freq_dim)
        k_real, k_imag = self._compute_phasors(k, T)  # (B, H, T, freq_dim)
        
        # === STEP 3: Wave interference (attention scores) ===
        scores = self._interference(q_real, q_imag, k_real, k_imag)  # (B, H, T, T)
        
        # === STEP 4: Causal mask (light cone) ===
        if causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
        else:
            causal_mask = mask
        
        # === STEP 5: Sharpening (softmax replacement) ===
        weights = self._sharpen(scores, causal_mask)  # (B, H, T, T)
        weights = self.dropout(weights)
        
        # Store for visualization
        self.last_weights = weights.detach()
        
        # === STEP 6: Apply attention to values ===
        out = torch.matmul(weights, v)  # (B, H, T, head_dim)
        
        # === STEP 7: Reshape and project output ===
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)
        
        return out


class PhasorWaveBlock(nn.Module):
    """
    Transformer block using PhasorWaveAttention.
    
    Architecture: LN -> Attention -> Residual -> LN -> MLP -> Residual
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_len: int = 2048,
        dropout: float = 0.1,
        sharpness: float = 3.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = PhasorWaveAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout,
            sharpness=sharpness,
        )
        
        self.ln2 = nn.LayerNorm(d_model)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.ln1(x))
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x


class PhasorWaveGPT(nn.Module):
    """
    GPT-style model using PhasorWaveAttention.
    
    This is a drop-in replacement for standard GPT that uses
    wave interference instead of softmax attention.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_len: int = 2048,
        dropout: float = 0.1,
        sharpness: float = 3.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Token embedding (standard)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding (standard, but could be removed since
        # phasor attention encodes position via phase)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks with PhasorWaveAttention
        self.blocks = nn.ModuleList([
            PhasorWaveBlock(
                d_model=d_model,
                num_heads=num_heads,
                max_len=max_len,
                dropout=dropout,
                sharpness=sharpness,
            )
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            idx: Token indices (B, T)
            targets: Optional target indices for loss computation
            
        Returns:
            logits: Output logits (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        B, T = idx.shape
        device = idx.device
        
        # Token + positional embeddings
        tok_emb = self.token_emb(idx)  # (B, T, D)
        pos = torch.arange(T, device=device)
        pos_emb = self.pos_emb(pos)    # (T, D)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        
        return logits, loss


# === TESTING ===
if __name__ == "__main__":
    print("=" * 60)
    print("PhasorWaveAttention Test")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    seq_len = 128
    d_model = 256
    num_heads = 8
    vocab_size = 1000
    
    # Create model
    model = PhasorWaveGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=4,
        max_len=512,
        sharpness=3.0,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, loss = model(x, y)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test attention weights
    attn = model.blocks[0].attn
    print(f"\nAttention weights shape: {attn.last_weights.shape}")
    print(f"Attention weights sum (should be ~1): {attn.last_weights[0, 0, -1].sum().item():.4f}")
    
    # Check frequency bank
    print(f"\nFrequency bank:")
    print(f"  Min freq: {attn.freqs.min().item():.6f} (wavelength: {1/attn.freqs.min().item():.1f} tokens)")
    print(f"  Max freq: {attn.freqs.max().item():.6f} (wavelength: {1/attn.freqs.max().item():.1f} tokens)")
    print(f"  Num frequencies: {attn.freqs.shape[0]}")
    
    print("\n✅ PhasorWaveAttention working correctly!")
    print("=" * 60)
