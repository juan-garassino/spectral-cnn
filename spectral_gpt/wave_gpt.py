"""
Wave-Native GPT: Language Modeling via Continuous Wave Functions

Philosophy: "Everything is a mass on a spring"
- Tokens are embedded as wave packets (frequency, phase, amplitude)
- Computation happens via wave interference
- Discretization only at output (measurement/collapse)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

# ==========================================
# Configuration
# ==========================================

@dataclass
class WaveGPTConfig:
    vocab_size: int
    d_model: int           # Embedding dimension (also wave dimension)
    num_layers: int
    num_heads: int
    num_waves: int         # Number of wave components per token
    num_harmonics: int = 4 # Harmonics per wave (1f, 2f, 3f, 4f)
    block_size: int = 256  # Context length
    dropout: float = 0.1
    pure_wave_attention: bool = False  # True = NO SOFTMAX, pure interference
    pure_wave_kernel: str = "elu_plus_one" # Kernel for pure wave: 'elu_plus_one', 'sigmoid', 'exp'
    pure_wave_mode: str = "quadratic"      # 'quadratic' (N^2, exact kernel) or 'linear' (N, decomposable)
    model_type: str = "wave"               # "wave", "standard", or "pure_wave"
    use_interference_attention: bool = False  # True = physics-based interference attention (Req 2.1-2.5)
    use_wave_embeddings: bool = True       # True = WavePacketEmbedding, False = StandardEmbedding (Req 7.5)
    pure_wave_mode_v2: bool = False        # True = PURE WAVE-TO-WAVE (no embeddings anywhere!)
    
    # === NON-LINEAR PHYSICS TRINITY (Breaks Linearity Plateau) ===
    # These solve the problem of smooth sine waves being unable to model sharp logic
    use_overdrive: bool = True       # Soft-clipping saturation → square waves (binary 0/1 states)
    use_fm_synthesis: bool = True    # Frequency modulation → context shifts logic frequencies
    use_interferometer: bool = True  # Phase-based gated residuals → boolean logic via interference
    mlp_expansion: int = 4           # Hidden dimension multiplier (shifts params to reasoning core)

# ==========================================
# Standard Transformer Components (The Control)
# ==========================================

class StandardEmbedding(nn.Module):
    """Standard Token + Positional Embedding"""
    def __init__(self, vocab_size, d_model, block_size, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device
        
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0) # (1, T)
        
        tok_emb = self.token_embedding(idx) # (B, T, C)
        pos_emb = self.position_embedding(pos) # (1, T, C)
        
        x = self.dropout(tok_emb + pos_emb)
        return x

class StandardCausalSelfAttention(nn.Module):
    """Vanilla Multi-Head Attention with Causal Mask"""
    def __init__(self, d_model, num_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = num_heads
        self.n_embd = d_model
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class StandardBlock(nn.Module):
    """Transformer Block: LN -> Attn -> LN -> MLP"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = StandardCausalSelfAttention(config.d_model, config.num_heads, config.block_size, config.dropout)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PureWaveBlock(nn.Module):
    """
    Pure Wave-Native Transformer Block.
    
    NO EMBEDDINGS! Everything operates on wave parameters:
    - Input: (frequencies, phases, amplitudes)
    - Attention: Wave interference between wave parameters
    - MLP: Wave parameter transformations
    - Output: (frequencies, phases, amplitudes)
    
    This is truly wave-native computation!
    """
    def __init__(self, config):
        super().__init__()
        
        # Wave-native attention
        self.wave_attn = InterferenceAttention(
            num_waves=config.num_waves,
            num_harmonics=config.num_harmonics,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Wave-native MLP (operates on wave parameters)
        self.wave_mlp = WaveNativeMLP(
            num_waves=config.num_waves,
            num_harmonics=config.num_harmonics,
            dropout=config.dropout
        )
        
        # Wave normalization (instead of LayerNorm)
        self.wave_norm1 = WaveNormalization(config.num_waves, config.num_harmonics)
        self.wave_norm2 = WaveNormalization(config.num_waves, config.num_harmonics)
        
    def forward(self, wave_freqs, wave_phases, wave_amps):
        """
        Pure wave computation - no embeddings anywhere!
        
        Args:
            wave_freqs: (B, T, num_waves)
            wave_phases: (B, T, num_waves)  
            wave_amps: (B, T, num_waves, num_harmonics)
            
        Returns:
            Same shapes but contextualized through wave physics
        """
        # Normalize waves
        norm_freqs, norm_phases, norm_amps = self.wave_norm1(wave_freqs, wave_phases, wave_amps)
        
        # Wave interference attention
        attn_freqs, attn_phases, attn_amps = self.wave_attn(norm_freqs, norm_phases, norm_amps)
        
        # Residual connection in wave space
        res_freqs = wave_freqs + attn_freqs
        res_phases = wave_phases + attn_phases  
        res_amps = wave_amps + attn_amps
        
        # Normalize again
        norm_freqs2, norm_phases2, norm_amps2 = self.wave_norm2(res_freqs, res_phases, res_amps)
        
        # Wave MLP
        mlp_freqs, mlp_phases, mlp_amps = self.wave_mlp(norm_freqs2, norm_phases2, norm_amps2)
        
        # Final residual connection
        output_freqs = res_freqs + mlp_freqs
        output_phases = res_phases + mlp_phases
        output_amps = res_amps + mlp_amps
        
        return output_freqs, output_phases, output_amps


class WaveNativeMLP(nn.Module):
    """MLP that operates directly on wave parameters"""
    def __init__(self, num_waves, num_harmonics, dropout=0.1):
        super().__init__()
        
        # Separate MLPs for each wave parameter type
        self.freq_mlp = nn.Sequential(
            nn.Linear(num_waves, 4 * num_waves),
            nn.GELU(),
            nn.Linear(4 * num_waves, num_waves),
            nn.Dropout(dropout)
        )
        
        self.phase_mlp = nn.Sequential(
            nn.Linear(num_waves, 4 * num_waves),
            nn.GELU(), 
            nn.Linear(4 * num_waves, num_waves),
            nn.Dropout(dropout)
        )
        
        amp_dim = num_waves * num_harmonics
        self.amp_mlp = nn.Sequential(
            nn.Linear(amp_dim, 4 * amp_dim),
            nn.GELU(),
            nn.Linear(4 * amp_dim, amp_dim),
            nn.Dropout(dropout)
        )
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
    def forward(self, wave_freqs, wave_phases, wave_amps):
        B, T, W = wave_freqs.shape
        H = wave_amps.shape[-1]
        
        # Transform each wave parameter type
        new_freqs = self.freq_mlp(wave_freqs)
        new_phases = self.phase_mlp(wave_phases)
        
        # Flatten amplitudes for MLP
        amps_flat = wave_amps.view(B, T, -1)
        new_amps_flat = self.amp_mlp(amps_flat)
        new_amps = new_amps_flat.view(B, T, W, H)
        
        return new_freqs, new_phases, new_amps


class WaveNormalization(nn.Module):
    """Normalization for wave parameters (instead of LayerNorm)"""
    def __init__(self, num_waves, num_harmonics):
        super().__init__()
        
        # Learnable scales for each parameter type
        self.freq_scale = nn.Parameter(torch.ones(num_waves))
        self.phase_scale = nn.Parameter(torch.ones(num_waves))
        self.amp_scale = nn.Parameter(torch.ones(num_waves, num_harmonics))
        
        self.eps = 1e-8
        
    def forward(self, wave_freqs, wave_phases, wave_amps):
        # RMS normalization for each parameter type
        freq_rms = wave_freqs.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        norm_freqs = wave_freqs / freq_rms * self.freq_scale
        
        phase_rms = wave_phases.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        norm_phases = wave_phases / phase_rms * self.phase_scale
        
        amp_rms = wave_amps.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=self.eps)
        norm_amps = wave_amps / amp_rms * self.amp_scale
        
        return norm_freqs, norm_phases, norm_amps



# ==========================================
# Wave Packet Embedding
# ==========================================

class WavePacketEmbedding(nn.Module):
    """
    Physics-aware token embeddings based on mass-frequency relationships.
    
    Tokens are quantized resonant modes following natural physical laws:
    - Token mass follows Zipfian distribution: Mass(i) = 1/(i+1)
    - Base frequency: ω_0 = 1.0 / sqrt(Mass) (heavy tokens = low freq, light tokens = high freq)
    - Harmonic frequencies: ω_n = n * ω_0 (initialized as integer multiples, but LEARNABLE)
    - Harmonic amplitudes: A_n = 1/n (initialized with power law decay, but LEARNABLE)
    
    Training Strategy:
    - Supports annealing via standard_embed_ratio parameter
    - out = (1-r)*wave + r*standard where r decays from 1.0 to 0.0
    
    Key change from original: frequencies and amplitudes are now LEARNABLE parameters
    initialized with physics-based values, allowing the model to adapt while starting
    from a physically meaningful prior.
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
    """
    def __init__(self, vocab_size, d_model, num_waves=16, num_harmonics=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
        # === Requirement 1.1: Zipfian mass distribution ===
        # Mass(i) = 1/(i+1) - stored as buffer (reference only)
        token_indices = torch.arange(vocab_size, dtype=torch.float32)
        masses = 1.0 / (token_indices + 1.0)  # Zipfian: Mass(i) = 1/(i+1)
        self.register_buffer('masses', masses)  # (vocab_size,)
        
        # === Requirement 1.2: MULTI-SCALE frequency spectrum ===
        # Create a "sea of low frequencies with spikes for local attention"
        # 
        # Strategy: Most waves are LOW frequency (global context), but some are HIGH (local spikes)
        # Distribution: 70% low freq, 20% mid freq, 10% high freq
        # This creates the desired "sea with spikes" pattern
        
        # Define frequency bands for multi-scale attention:
        # - Low freq (0.1-1.0): peaks every 6-60 tokens → global/document context  
        # - Mid freq (1.0-5.0): peaks every 1-6 tokens → paragraph context
        # - High freq (5.0-20.0): peaks every 0.3-1 tokens → local/word spikes
        
        n_low = int(0.7 * num_waves)    # 70% low frequencies (the "sea")
        n_mid = int(0.2 * num_waves)    # 20% mid frequencies  
        n_high = num_waves - n_low - n_mid  # 10% high frequencies (the "spikes")
        
        # Generate frequency bands
        low_freqs = torch.linspace(0.1, 1.0, n_low)      # Sea of low frequencies
        mid_freqs = torch.linspace(1.0, 5.0, n_mid)      # Medium frequencies
        high_freqs = torch.linspace(5.0, 20.0, n_high)   # Local attention spikes
        
        # Combine into full spectrum (sea + spikes)
        base_spectrum = torch.cat([low_freqs, mid_freqs, high_freqs])
        
        # Shuffle to mix low/mid/high across wave components (breaks patterns)
        perm = torch.randperm(num_waves)
        base_spectrum = base_spectrum[perm]
        
        # Add small token-dependent variation (each token gets slightly different spectrum)
        # Common tokens (low mass) get slightly lower frequencies → more global attention
        # Rare tokens (high mass) get slightly higher frequencies → more local attention
        raw_center_freq = 1.0 / torch.sqrt(masses)  # (vocab_size,)
        token_variation = (raw_center_freq - raw_center_freq.mean()) / raw_center_freq.std()
        token_variation = token_variation * 0.1  # Small variation ±10%
        
        # Each token gets the base spectrum with small individual variation
        # Shape: (vocab_size, num_waves)
        init_base_frequencies = base_spectrum.unsqueeze(0) * (1.0 + token_variation.unsqueeze(1))
        self.base_freqs = nn.Parameter(init_base_frequencies.clone())  # LEARNABLE!
        
        # === Requirement 1.3: Harmonic quantization ===
        # Harmonic multipliers - keep as buffer (structural, not learned)
        harmonic_mults = torch.arange(1, num_harmonics + 1, dtype=torch.float32)  # [1, 2, 3, 4, ...]
        self.register_buffer('harmonic_mults', harmonic_mults)
        
        # === Requirement 1.4: Power law amplitude decay ===
        # LEARNABLE amplitudes initialized with physics prior: A_n = 1/n
        # But add some randomness to create more interesting wave shapes
        base_amplitudes = 1.0 / harmonic_mults  # [1, 0.5, 0.333, 0.25, ...]
        
        # Add small random variations to break uniformity and create unique wave shapes
        # Each token-wave combination gets slightly different harmonic content
        random_variations = torch.randn(vocab_size, num_waves, num_harmonics) * 0.2
        init_harmonic_amplitudes = base_amplitudes.view(1, 1, -1) * (1.0 + random_variations)
        
        # Ensure amplitudes stay positive and reasonable
        init_harmonic_amplitudes = torch.clamp(init_harmonic_amplitudes, min=0.1, max=2.0)
        
        self.harmonic_amps = nn.Parameter(init_harmonic_amplitudes)  # NOW LEARNABLE!
        
        # Debug: Print frequency distribution
        print(f"🌊 Frequency spectrum initialized:")
        print(f"   Low freq (sea): {low_freqs.min():.3f} - {low_freqs.max():.3f} Hz ({n_low} waves)")
        print(f"   Mid freq: {mid_freqs.min():.3f} - {mid_freqs.max():.3f} Hz ({n_mid} waves)")  
        print(f"   High freq (spikes): {high_freqs.min():.3f} - {high_freqs.max():.3f} Hz ({n_high} waves)")
        print(f"   Total range: {base_spectrum.min():.3f} - {base_spectrum.max():.3f} Hz")
        
        # === Phases: Random initialization with structure ===
        # Golden angle creates too rigid a pattern. Use random init with some structure:
        # - Random base phase per token (allows learning unique token "signatures")
        # - Small wave-dependent offset (breaks symmetry between waves)
        
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        golden_angle = 2 * math.pi / (phi ** 2)
        
        # Random phase per token, plus small structured offset per wave
        random_token_phase = torch.rand(vocab_size, 1) * 2 * math.pi
        wave_offset = torch.arange(num_waves, dtype=torch.float32).unsqueeze(0) * golden_angle * 0.1
        
        # Add small random noise to break any remaining patterns
        noise = torch.randn(vocab_size, num_waves) * 0.3
        
        init_phases = (random_token_phase + wave_offset + noise) % (2 * math.pi)
        self.phases = nn.Parameter(init_phases)  # LEARNABLE!
        
        # Project wave state to d_model dimension
        # num_waves * num_harmonics * 2 (sin + cos) + num_waves (phase) + num_waves (freq)
        wave_dim = num_waves * num_harmonics * 2 + num_waves * 2
        self.wave_to_embed = nn.Linear(wave_dim, d_model)
        
        # Initialize wave_to_embed with slightly larger weights for phase/freq pathways
        # This ensures gradients flow back to phases and frequencies
        # NOTE: 10x was too aggressive and caused instability, using 3x instead
        with torch.no_grad():
            sin_cos_dim = num_waves * num_harmonics * 2
            self.wave_to_embed.weight[:, sin_cos_dim:] *= 3.0  # 3x larger weights (was 10x)
        
        # Learnable scale for phase contribution (starts smaller for stability)
        self.phase_scale = nn.Parameter(torch.tensor(0.5))  # Start at 0.5, not 1.0
        
        # Positional wave modulation
        self.pos_freq = nn.Parameter(torch.randn(1, 1, num_waves) * 0.1)
        
        # === Requirement 1.5: Standard embedding for annealing ===
        self.simple_embed = nn.Embedding(vocab_size, d_model)
        
        # Use learned scaling instead of LayerNorm (LN kills gradients to phases/freqs)
        # RMSNorm-style: just scale by learned parameter, no mean subtraction
        self.output_scale = nn.Parameter(torch.ones(d_model))
        
    def forward(self, token_ids, standard_embed_ratio=0.0):
        """
        Compute wave packet embeddings with optional annealing.
        
        Args:
            token_ids: (B, T) tensor of token indices
            standard_embed_ratio: Mixing ratio for standard embeddings
                                  0.0 = pure wave, 1.0 = pure standard
                                  Requirement 1.5: out = (1-r)*wave + r*standard
        
        Returns:
            (B, T, d_model) wave packet embeddings
        """
        B, T = token_ids.shape
        device = token_ids.device
        
        # Get wave parameters for each token
        base_f = self.base_freqs[token_ids]    # (B, T, num_waves)
        phases = self.phases[token_ids]         # (B, T, num_waves)
        harm_a = self.harmonic_amps[token_ids]  # (B, T, num_waves, num_harmonics)
        
        # Positional modulation
        positions = torch.arange(T, device=device).float().view(1, T, 1)
        pos_phase = positions * self.pos_freq
        
        # Generate harmonics: ω_n = n * ω_0 (Requirement 1.3)
        # base_f: (B, T, W) -> expand to (B, T, W, H)
        freqs = base_f.unsqueeze(-1) * self.harmonic_mults  # (B, T, W, H)
        
        # === PHASE 2: TEMPORAL EVOLUTION ===
        # Position in sequence = Time t
        # Phase evolves according to wave equation: φ(t) = ω*t + φ_0
        # This is pure physics - no hardcoded envelopes!
        wave_phase = freqs * positions.unsqueeze(-1) + phases.unsqueeze(-1) + pos_phase.unsqueeze(-1)
        
        # === WAVE PACKET GENERATION ===
        # The token is "struck" and generates oscillations at its natural frequencies
        # No artificial envelopes - the wave shape emerges from harmonic superposition
        sin_waves = harm_a * torch.sin(wave_phase)  # (B, T, W, H)
        cos_waves = harm_a * torch.cos(wave_phase)  # (B, T, W, H)
        
        # === DIRECT PHASE PATHWAY for stronger gradients ===
        # The sin/cos pathway dilutes phase gradients. Add direct phase info.
        # Normalize phases to [-1, 1] range and scale for stability
        phase_direct = self.phase_scale * ((phases / math.pi) - 1.0)  # (B, T, num_waves)
        
        # Also add frequency info directly (normalized to similar scale as phases)
        freq_direct = torch.log1p(base_f) / 5.0  # Normalize to ~[0, 1] range
        
        # Flatten: (B, T, W*H*2 + W + W)
        wave_state = torch.cat([
            sin_waves.reshape(B, T, -1),
            cos_waves.reshape(B, T, -1),
            phase_direct,  # Direct phase pathway!
            freq_direct    # Direct frequency pathway!
        ], dim=-1)
        
        # Project to embedding dimension
        wave_embed = self.wave_to_embed(wave_state)  # (B, T, d_model)
        
        # === Requirement 1.5: Embedding annealing ===
        # out = (1-r)*wave + r*standard
        if standard_embed_ratio > 0.0:
            simple_embed = self.simple_embed(token_ids)
            r = standard_embed_ratio
            embeddings = (1.0 - r) * wave_embed + r * simple_embed
        else:
            embeddings = wave_embed
        
        # Scale output (RMSNorm-style, preserves gradients unlike LayerNorm)
        rms = embeddings.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        embeddings = embeddings / rms * self.output_scale
        
        # === PURE WAVE OUTPUT ===
        # Instead of projecting to embedding space, return raw wave parameters!
        # This enables pure wave-to-wave computation throughout the network
        
        if standard_embed_ratio > 0.0:
            # For annealing, we still need embeddings temporarily
            simple_embed = self.simple_embed(token_ids)
            r = standard_embed_ratio
            embeddings = (1.0 - r) * wave_embed + r * simple_embed
            
            # Scale output (RMSNorm-style, preserves gradients unlike LayerNorm)
            rms = embeddings.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
            embeddings = embeddings / rms * self.output_scale
            
            return embeddings
        else:
            # Pure wave mode: return wave parameters directly!
            return base_f, phases, harm_a  # (freqs, phases, amplitudes)
    
    def get_token_mass(self, token_id):
        """Get the mass for a specific token (Requirement 1.1)."""
        return self.masses[token_id]
    
    def get_base_frequency(self, token_id):
        """Get the base frequency for a specific token (Requirement 1.2)."""
        return self.base_freqs[token_id, 0]  # Same for all waves
    
    def get_harmonic_frequencies(self, token_id):
        """Get all harmonic frequencies for a token (Requirement 1.3)."""
        base_freq = self.base_freqs[token_id, 0]
        return base_freq * self.harmonic_mults
    
    def get_harmonic_amplitudes(self):
        """Get the harmonic amplitude decay pattern (Requirement 1.4)."""
        return 1.0 / self.harmonic_mults


# ==========================================
# Wave Interference Attention (Legacy - Hybrid)
# ==========================================

class WaveInterferenceAttention(nn.Module):
    """
    Attention via wave interference instead of dot product.
    
    Key insight: When waves meet, they interfere:
    - Constructive: waves in phase → amplify (high attention)
    - Destructive: waves out of phase → cancel (low attention)
    
    This is natural for waves and avoids the discrete dot product.
    """
    def __init__(self, d_model, num_heads, num_waves=16, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_waves = num_waves
        
        # Project to wave space for Q, K, V
        # Each becomes a wave state
        self.q_proj = nn.Linear(d_model, num_heads * num_waves)
        self.k_proj = nn.Linear(d_model, num_heads * num_waves)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Phase modulation for Q and K (learned)
        self.q_phase = nn.Parameter(torch.zeros(1, num_heads, 1, num_waves))
        self.k_phase = nn.Parameter(torch.zeros(1, num_heads, 1, num_waves))
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature for attention sharpness
        self.temperature = nn.Parameter(torch.ones(1) * (num_waves ** 0.5))
        
    def forward(self, x):
        """Wave interference attention"""
        B, T, C = x.shape
        
        # Project to wave space
        q_waves = self.q_proj(x).view(B, T, self.num_heads, self.num_waves)  # (B, T, H, W)
        k_waves = self.k_proj(x).view(B, T, self.num_heads, self.num_waves)  # (B, T, H, W)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)         # (B, T, H, D)
        
        # Transpose for attention: (B, H, T, ...)
        q_waves = q_waves.transpose(1, 2)  # (B, H, T, W)
        k_waves = k_waves.transpose(1, 2)  # (B, H, T, W)
        v = v.transpose(1, 2)              # (B, H, T, D)
        
        # Add learnable phase shifts
        q_waves = q_waves + self.q_phase
        k_waves = k_waves + self.k_phase
        
        # Wave interference: compute via cosine similarity of wave states
        # When waves have same phase → cos(0) = 1 (constructive)
        # When waves are out of phase → cos(π) = -1 (destructive)
        
        # Normalize waves
        q_norm = F.normalize(q_waves, dim=-1)
        k_norm = F.normalize(k_waves, dim=-1)
        
        # Interference pattern: dot product of normalized waves
        # This gives cosine similarity = wave interference strength
        interference = torch.matmul(q_norm, k_norm.transpose(-2, -1)) / self.temperature  # (B, H, T, T)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        interference = interference.masked_fill(causal_mask, float('-inf'))
        
        # Softmax to get attention weights (interference probabilities)
        attn = F.softmax(interference, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values (wave superposition of values)
        out = torch.matmul(attn, v)  # (B, H, T, D)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        
        return out


# ==========================================
# Physics-Based Interference Attention (Memory-Efficient)
# ==========================================

class InterferenceAttention(nn.Module):
    """
    Pure Wave-to-Wave Interference Attention.
    
    NO EMBEDDING SPACE! Direct wave parameter transformations.
    
    Input: Wave parameters (frequencies, phases, amplitudes)
    Output: Transformed wave parameters after interference
    
    This is truly wave-native - every operation has clear physical meaning:
    - Wave parameter transformations (learned resonance coupling)
    - Direct wave interference computation
    - Wave superposition for output
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """
    def __init__(
        self,
        num_waves: int = 16,
        num_harmonics: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        self.num_heads = num_heads
        
        # === DIRECT WAVE-TO-WAVE TRANSFORMATIONS ===
        # Transform input wave parameters to attention wave parameters
        self.freq_transform = nn.Linear(num_waves, num_heads * num_waves)
        self.phase_transform = nn.Linear(num_waves, num_heads * num_waves)
        self.amp_transform = nn.Linear(num_waves * num_harmonics, num_heads * num_waves)
        
        # Wave-to-wave value transformation (no embedding!)
        self.value_freq_transform = nn.Linear(num_waves, num_waves)
        self.value_phase_transform = nn.Linear(num_waves, num_waves)
        self.value_amp_transform = nn.Linear(num_waves * num_harmonics, num_waves * num_harmonics)
        
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-8
        
        # Learnable wave physics parameters
        self.temperature = nn.Parameter(torch.ones(1))
        self.interference_strength = nn.Parameter(torch.tensor(1.0))
        
        # Initialize transformations to preserve multi-scale structure
        with torch.no_grad():
            # Initialize frequency transform to spread across scales
            freq_init = torch.linspace(0.1, 5.0, num_waves).repeat(num_heads)
            self.freq_transform.bias.data = freq_init
        
    def forward(self, wave_freqs: torch.Tensor, wave_phases: torch.Tensor, wave_amps: torch.Tensor) -> tuple:
        """
        Pure Wave-to-Wave Interference Attention.
        
        Input: Raw wave parameters from embedding layer
        - wave_freqs: (B, T, num_waves) - base frequencies per token
        - wave_phases: (B, T, num_waves) - phases per token  
        - wave_amps: (B, T, num_waves, num_harmonics) - harmonic amplitudes
        
        Output: Transformed wave parameters after attention
        - Same shapes as input but contextualized through wave interference
        
        NO EMBEDDING SPACE - pure wave physics!
        """
        B, T, W = wave_freqs.shape
        H = wave_amps.shape[-1]  # num_harmonics
        device = wave_freqs.device
        dtype = wave_freqs.dtype
        
        # === WAVE-TO-WAVE TRANSFORMATIONS ===
        # Transform input waves to attention space waves
        attn_freqs = F.softplus(self.freq_transform(wave_freqs)) + self.eps  # (B, T, H*W)
        attn_freqs = attn_freqs.view(B, T, self.num_heads, self.num_waves).transpose(1, 2)  # (B, H, T, W)
        
        attn_phases = self.phase_transform(wave_phases).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        
        # Flatten amplitudes for transformation
        wave_amps_flat = wave_amps.view(B, T, -1)  # (B, T, W*H)
        attn_amps = F.softplus(self.amp_transform(wave_amps_flat)) + self.eps
        attn_amps = attn_amps.view(B, T, self.num_heads, self.num_waves).transpose(1, 2)  # (B, H, T, W)
        
        # === VALUE WAVE TRANSFORMATIONS ===
        # Transform input waves for values (wave-to-wave, no embedding!)
        value_freqs = F.softplus(self.value_freq_transform(wave_freqs)) + self.eps  # (B, T, W)
        value_phases = self.value_phase_transform(wave_phases)  # (B, T, W)
        value_amps = F.softplus(self.value_amp_transform(wave_amps_flat)) + self.eps
        value_amps = value_amps.view(B, T, W, H)  # (B, T, W, H)
        
        # === WAVE INTERFERENCE (Pure Physics!) ===
        # Position indices for phase evolution
        positions = torch.arange(T, device=device, dtype=dtype)
        
        # Phase at each position: θ(t) = ω*t + φ_0
        # Different frequencies create different phase evolution rates!
        theta = freq * positions.view(1, 1, T, 1) + phase  # (B, H, T, W)
        
        # Create complex phasors: A * e^(iθ)
        # This encodes both amplitude and phase in one complex number
        phasor = amp * torch.exp(1j * theta.to(torch.complex64))  # (B, H, T, W)
        
        # === INTERFERENCE PATTERN ===
        # When query phasor meets key phasor, they interfere:
        # score[q,k] = Re(sum_w(phasor_q[w] * conj(phasor_k[w])))
        #            = sum_w(A_q * A_k * cos(θ_q - θ_k))
        #
        # This NATURALLY creates:
        # - High scores when phases align (constructive interference)
        # - Low scores when phases oppose (destructive interference)
        # - BEATING patterns when frequencies differ!
        #   Beat period = 2π / |ω_q - ω_k| → learned from data!
        
        # === FULL INTERFERENCE EQUATION ===
        # I = A_Q² + A_K² + 2*A_Q*A_K*cos(Δφ)
        # 
        # The phasor dot product gives us: sum_w(A_q * A_k * cos(θ_q - θ_k))
        # This is the interference term: 2*A_Q*A_K*cos(Δφ) (summed over waves)
        
        interference_term = torch.matmul(phasor, phasor.conj().transpose(-2, -1)).real  # (B, H, T, T)
        
        # Compute amplitude squared terms for full intensity
        amp_sq = (amp ** 2).sum(dim=-1)  # (B, H, T) - sum over waves
        amp_q_sq = amp_sq.unsqueeze(-1)  # (B, H, T, 1)
        amp_k_sq = amp_sq.unsqueeze(-2)  # (B, H, 1, T)
        
        # Full interference intensity: I = A_Q² + A_K² + 2*A_Q*A_K*cos(Δφ)
        intensity = amp_q_sq + amp_k_sq + 2 * interference_term  # (B, H, T, T)
        
        # === PHYSICS-BASED NORMALIZATION ===
        # Normalize by maximum potential energy: (A_Q + A_K)²
        # This gives transmission coefficient in [0, 1] based on resonance
        amp_sum = amp.sum(dim=-1)  # (B, H, T) - total amplitude per position
        amp_q = amp_sum.unsqueeze(-1)  # (B, H, T, 1)
        amp_k = amp_sum.unsqueeze(-2)  # (B, H, 1, T)
        max_energy = (amp_q + amp_k) ** 2 + self.eps  # (B, H, T, T)
        
        # Transmission coefficient: how much energy passes through
        transmission = intensity / max_energy  # (B, H, T, T), range ~ [0, 1]
        
        # Scale by learned parameters
        scores = self.interference_strength * transmission * self.temperature
        
        # === Causal masking ===
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, 0.0)  # Zero out future (not -inf!)
        
        # === NO SOFTMAX! ===
        # Multiple contexts can be fully active simultaneously
        # Just normalize to ensure stability
        row_sum = scores.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        attn_weights = scores / row_sum  # Normalize by row sum
        attn_weights = self.dropout(attn_weights)
        
        # Store for visualization
        self.last_attention_weights = attn_weights.detach()
        self.last_interference = interference.detach()
        self.last_frequencies = freq.detach()
        self.last_phases = phase.detach()
        self.last_amplitudes = amp.detach()
        
        # === WAVE SUPERPOSITION (Instead of matrix multiply with values) ===
        # Apply attention weights to wave parameters directly!
        
        # Expand value waves for multi-head attention
        value_freqs_expanded = value_freqs.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, T, W)
        value_phases_expanded = value_phases.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, T, W)
        value_amps_expanded = value_amps.unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)  # (B, H, T, W, H)
        
        # Wave superposition: weighted combination of wave parameters
        output_freqs = torch.matmul(attn_weights, value_freqs_expanded)  # (B, H, T, W)
        output_phases = torch.matmul(attn_weights, value_phases_expanded)  # (B, H, T, W)
        
        # For amplitudes, we need to handle the extra harmonic dimension
        value_amps_flat = value_amps_expanded.view(B, self.num_heads, T, -1)  # (B, H, T, W*H)
        output_amps_flat = torch.matmul(attn_weights, value_amps_flat)  # (B, H, T, W*H)
        output_amps = output_amps_flat.view(B, self.num_heads, T, W, H)  # (B, H, T, W, H)
        
        # === MULTI-HEAD WAVE FUSION ===
        # Combine multiple attention heads back to single wave representation
        final_freqs = output_freqs.mean(dim=1)  # (B, T, W) - average across heads
        final_phases = output_phases.mean(dim=1)  # (B, T, W)
        final_amps = output_amps.mean(dim=1)  # (B, T, W, H)
        
        return final_freqs, final_phases, final_amps
    
    def get_interference_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw wave interference scores for visualization."""
        with torch.no_grad():
            _ = self.forward(x)
        
        if hasattr(self, 'last_interference'):
            return self.last_interference
        else:
            return torch.zeros(x.size(0), self.num_heads, x.size(1), x.size(1), device=x.device)
    
    def get_interference_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized attention weights (after softmax)."""
        with torch.no_grad():
            _ = self.forward(x)
        
        if hasattr(self, 'last_attention_weights'):
            return self.last_attention_weights
        else:
            T = x.size(1)
            return torch.zeros(x.size(0), self.num_heads, T, T, device=x.device)
    
    def get_wave_components(self, x: torch.Tensor) -> dict:
        """Get learned wave components for visualization."""
        with torch.no_grad():
            _ = self.forward(x)
        
        return {
            'frequencies': self.last_frequencies if hasattr(self, 'last_frequencies') else None,
            'phases': self.last_phases if hasattr(self, 'last_phases') else None,
            'amplitudes': self.last_amplitudes if hasattr(self, 'last_amplitudes') else None,
            'interference': self.last_interference if hasattr(self, 'last_interference') else None,
            'attention_weights': self.last_attention_weights if hasattr(self, 'last_attention_weights') else None,
        }
    
# ==========================================
# PURE Wave Attention (NO SOFTMAX!)
# ==========================================

class PureWaveAttention(nn.Module):
    """
    TRUE wave interference attention - NO SOFTMAX, NO DOT PRODUCT.
    
    This is the pure wave paradigm:
    - Attention weights come directly from wave interference
    - Negative values = destructive interference = SUPPRESSION
    - No softmax normalization - pure field dynamics
    
    Key differences from standard attention:
    1. No softmax: interference can be negative (destructive)
    2. No Q/K dot product: uses phase-based interference
    3. Bounded naturally: cosine similarity is in [-1, 1]
    """
    def __init__(self, d_model, num_heads, num_waves=16, dropout=0.1, kernel='elu_plus_one', mode='quadratic'):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_waves = num_waves
        self.kernel = kernel
        self.mode = mode
        
        # Wave projections - map to frequency/phase space
        self.q_freq = nn.Linear(d_model, num_heads * num_waves)
        self.k_freq = nn.Linear(d_model, num_heads * num_waves)
        self.q_phase = nn.Linear(d_model, num_heads * num_waves)
        self.k_phase = nn.Linear(d_model, num_heads * num_waves)
        
        # Value projection (still needed to carry information)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Learnable interference strength per head
        self.interference_scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Pure wave interference attention.
        """
        B, T, C = x.shape
        
        # Project to wave parameters
        q_f = self.q_freq(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        k_f = self.k_freq(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        q_p = self.q_phase(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        k_p = self.k_phase(x).view(B, T, self.num_heads, self.num_waves).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute wave components
        t_pos = torch.arange(T, device=x.device).float().view(1, 1, T, 1)
        q_waves = torch.sin(q_f * t_pos + q_p)
        k_waves = torch.sin(k_f * t_pos + k_p)
        
        # Normalize
        q_norm = F.normalize(q_waves, dim=-1)
        k_norm = F.normalize(k_waves, dim=-1)
        
        # Scale q/k to have variance 1 before kernel
        # interference = q @ k * sqrt(W). Equivalent to q*W^0.25 @ k*W^0.25? 
        # For linear mode, we need to distribute the scaling.
        scale_factor = (self.num_waves ** 0.25)
        q_norm = q_norm * scale_factor
        k_norm = k_norm * scale_factor
        
        if self.mode == 'linear':
            # === Linear Attention: O(N) ===
            # Decompose kernel: phi(Q) @ phi(K)^T
            
            # 1. Apply kernel to Q and K directly
            if self.kernel == 'elu_plus_one':
                q_prime = F.elu(q_norm) + 1.0
                k_prime = F.elu(k_norm) + 1.0
            elif self.kernel == 'sigmoid':
                q_prime = torch.sigmoid(q_norm)
                k_prime = torch.sigmoid(k_norm)
            elif self.kernel == 'exp':
                q_prime = torch.exp(q_norm)
                k_prime = torch.exp(k_norm)
            else: # identity (not recommended for linear mode as cumsum will drift)
                q_prime = F.elu(q_norm) + 1.0
                k_prime = F.elu(k_norm) + 1.0
            
            # 2. Compute KV state = cumsum(K' * V)
            # Result: (B, H, T, W, D)
            kv = torch.einsum('bhtw, bhtd -> bhtwd', k_prime, v)
            S = torch.cumsum(kv, dim=2)
            
            # 3. Compute Z state (normalization denominator)
            # Result: (B, H, T, W)
            Z = torch.cumsum(k_prime, dim=2)
            
            # 4. Numerator = Q' . S (sum over W)
            numerator = torch.einsum('bhtw, bhtwd -> bhtd', q_prime, S)
            
            # 5. Denominator = Q' . Z (sum over W)
            denominator = torch.einsum('bhtw, bhtw -> bht', q_prime, Z)
            denominator = denominator.unsqueeze(-1)
            
            out = numerator / (denominator + 1e-6)
            
        else:
            # === Quadratic Attention: O(N^2) ===
            # Explicit interference matrix calculation
            
            # Interference = (q @ k) 
            # Note: q_norm, k_norm already scaled by W^0.25. Product is scaled by W^0.5.
            interference = torch.matmul(q_norm, k_norm.transpose(-2, -1))
            
            # Scale per head (learnable)
            interference = interference * self.interference_scale
            
            # Masking
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            
            # Activation
            if self.kernel == 'elu_plus_one':
                interference = interference.masked_fill(causal_mask, -float('inf'))
                attn = F.elu(interference) + 1.0
            elif self.kernel == 'sigmoid':
                interference = interference.masked_fill(causal_mask, -float('inf'))
                attn = torch.sigmoid(interference)
            elif self.kernel == 'exp':
                interference = interference.masked_fill(causal_mask, -float('inf'))
                attn = torch.exp(interference)
            else:
                interference = interference.masked_fill(causal_mask, 0.0)
                attn = interference
            
            # Normalize by number of attended positions (approx Z)
            num_attended = torch.arange(1, T + 1, device=x.device).float().view(1, 1, T, 1)
            attn = attn / num_attended.sqrt()
            
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        # Output projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        
        return out


# ==========================================
# Wave MLP (Resonance Network)
# ==========================================

class WaveResonanceMLP(nn.Module):
    """
    MLP that preserves wave nature using harmonic activations.
    
    Instead of ReLU (which clips), use periodic activations
    that maintain the oscillatory structure.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Hybrid activation: GELU for expressivity + sin for wave nature
        # Pure sin traps gradients; pure GELU loses wave character
        h = self.fc1(x)
        h = F.gelu(h) + 0.1 * torch.sin(h)  # GELU dominant, sin adds harmonics
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return h


# ==========================================
# Wave Block
# ==========================================

class WaveBlock(nn.Module):
    """Single transformer block operating in wave space"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # Choose attention type based on config
        if getattr(config, 'use_interference_attention', False):
            # Physics-based interference attention (Requirements 2.1-2.5)
            self.attn = InterferenceAttention(
                config.d_model, config.num_heads,
                config.num_waves, config.dropout
            )
        elif getattr(config, 'pure_wave_attention', False):
            # PURE wave attention - NO SOFTMAX!
            self.attn = PureWaveAttention(
                config.d_model, config.num_heads, 
                config.num_waves, config.dropout,
                kernel=getattr(config, 'pure_wave_kernel', 'elu_plus_one')
            )
        else:
            # Hybrid wave attention (with softmax)
            self.attn = WaveInterferenceAttention(
                config.d_model, config.num_heads, 
                config.num_waves, config.dropout
            )
        
        self.mlp = WaveResonanceMLP(
            config.d_model, config.d_model * 4, config.dropout
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ==========================================
# Collapse Head (Wave → Token)
# ==========================================

class CollapseHead(nn.Module):
    """
    Wave function collapse: continuous wave state → discrete token.
    
    Like quantum measurement: the continuous superposition
    "collapses" to a probability distribution over tokens.
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.ln(x)
        logits = self.proj(x)  # Wave collapse → token probabilities
        return logits


# ==========================================
# Wave-Native GPT
# ==========================================

# Note: WaveGPTConfig is defined as a dataclass at the top of the file.
# This placeholder class is kept for backward compatibility with code that
# instantiates WaveGPTConfig with positional arguments.
class WaveGPTConfigCompat:
    """Compatibility class for WaveGPTConfig with positional arguments."""
    def __init__(self, vocab_size, d_model, num_heads, num_waves, dropout, num_harmonics, num_layers, block_size, model_type="wave", pure_wave_attention=False, pure_wave_kernel='elu_plus_one', pure_wave_mode='quadratic', use_interference_attention=False, use_wave_embeddings=True):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_waves = num_waves
        self.dropout = dropout
        self.num_harmonics = num_harmonics
        self.num_layers = num_layers
        self.block_size = block_size
        self.model_type = model_type
        self.pure_wave_attention = pure_wave_attention
        self.pure_wave_kernel = pure_wave_kernel
        self.pure_wave_mode = pure_wave_mode
        self.use_interference_attention = use_interference_attention
        self.use_wave_embeddings = use_wave_embeddings

# Placeholder for StandardEmbedding and StandardBlock
# These classes were referenced in the diff but not provided.
# For the code to be syntactically correct, I'll add minimal placeholder definitions.
class StandardEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, dropout):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        token_emb = self.token_embeddings(idx)
        pos_emb = self.position_embeddings(pos)
        x = self.dropout(token_emb + pos_emb)
        return x

class StandardBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=config.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=self._generate_square_subsequent_mask(x.size(1)).to(x.device))[0]
        x = x + self.mlp(self.ln2(x))
        return x
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# ==========================================
# PURE WAVE GPT - Wave-to-Wave Throughout!
# ==========================================

class PureWaveGPT(nn.Module):
    """
    Pure Wave-Native GPT: Wave-to-Wave computation throughout.
    
    NO EMBEDDING SPACE! Everything is waves:
    - Token → Wave excitation (frequencies, phases, amplitudes)
    - Wave → Wave attention (interference physics)
    - Wave → Wave MLP (resonance filtering)
    - Wave → Logits (measurement/collapse)
    
    All patterns emerge from learned wave physics:
    - Decay and re-emergence from frequency beating
    - Attention patterns from phase interference
    - Context from wave superposition
    """
    
    def __init__(self, config: WaveGPTConfig):
        super().__init__()
        self.config = config
        
        # === WAVE EXCITATION (Token → Wave) ===
        self.wave_excitation = PureWaveExcitation(
            vocab_size=config.vocab_size,
            num_waves=config.num_waves,
            num_harmonics=config.num_harmonics,
            block_size=config.block_size
        )
        
        # === WAVE TRANSFORMER LAYERS ===
        self.wave_layers = nn.ModuleList([
            PureWaveLayer(
                num_waves=config.num_waves,
                num_harmonics=config.num_harmonics,
                num_heads=config.num_heads,
                dropout=config.dropout,
                # Non-Linear Physics Trinity
                use_overdrive=config.use_overdrive,
                use_fm_synthesis=config.use_fm_synthesis,
                use_interferometer=config.use_interferometer,
                mlp_expansion=config.mlp_expansion
            )
            for _ in range(config.num_layers)
        ])
        
        # === WAVE COLLAPSE (Wave → Logits) ===
        self.wave_collapse = WaveCollapse(
            vocab_size=config.vocab_size,
            num_waves=config.num_waves,
            num_harmonics=config.num_harmonics
        )
        
    def forward(self, token_ids, targets=None):
        """
        Enhanced pure wave forward pass with wave regularization.
        
        Token → Wave → Wave → ... → Wave → Logits
        """
        B, T = token_ids.shape
        
        # 1. EXCITATION: Token → Wave parameters
        wave_state = self.wave_excitation(token_ids)
        # wave_state = (freqs, phases, amps) - pure wave representation!
        
        # Store initial wave state for regularization
        initial_wave_state = WaveState(
            wave_state.freqs.clone(),
            wave_state.phases.clone(), 
            wave_state.amps.clone()
        )
        
        # 2. WAVE LAYERS: Wave → Wave transformations
        for layer in self.wave_layers:
            wave_state = layer(wave_state)
        
        # 3. COLLAPSE: Wave → Logits (measurement)
        logits = self.wave_collapse(wave_state)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Primary loss: cross-entropy
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # === WAVE PHYSICS REGULARIZATION ===
            wave_reg_loss = 0.0
            
            # 1. Frequency stability: prevent frequencies from becoming too extreme
            freq_stability = torch.mean((wave_state.freqs - initial_wave_state.freqs) ** 2) * 0.01
            
            # 2. Phase coherence: encourage smooth phase evolution
            if T > 1:
                phase_diff = wave_state.phases[:, 1:] - wave_state.phases[:, :-1]
                phase_coherence = torch.mean(torch.sin(phase_diff) ** 2) * 0.005  # Penalize abrupt phase jumps
            else:
                phase_coherence = 0.0
            
            # 3. Amplitude conservation: prevent amplitude explosion/collapse
            amp_conservation = torch.mean((wave_state.amps.mean() - initial_wave_state.amps.mean()) ** 2) * 0.01
            
            # 4. Wave coupling regularization: encourage meaningful wave interactions
            if hasattr(self.wave_excitation, 'wave_coupling'):
                coupling_reg = torch.mean(self.wave_excitation.wave_coupling ** 2) * 0.001
            else:
                coupling_reg = 0.0
            
            wave_reg_loss = freq_stability + phase_coherence + amp_conservation + coupling_reg
            
            # Total loss
            loss = ce_loss + wave_reg_loss
            
            # Store loss components for monitoring
            if hasattr(self, 'loss_components'):
                self.loss_components = {
                    'ce_loss': ce_loss.item(),
                    'freq_stability': freq_stability.item() if isinstance(freq_stability, torch.Tensor) else freq_stability,
                    'phase_coherence': phase_coherence.item() if isinstance(phase_coherence, torch.Tensor) else phase_coherence,
                    'amp_conservation': amp_conservation.item() if isinstance(amp_conservation, torch.Tensor) else amp_conservation,
                    'coupling_reg': coupling_reg.item() if isinstance(coupling_reg, torch.Tensor) else coupling_reg,
                    'total_wave_reg': wave_reg_loss.item() if isinstance(wave_reg_loss, torch.Tensor) else wave_reg_loss
                }
        
        return logits, loss
    
    def get_wave_state(self, token_ids, layer_idx=None):
        """Get wave state at a specific layer for visualization."""
        wave_state = self.wave_excitation(token_ids)
        
        if layer_idx is None:
            return wave_state
        
        for i, layer in enumerate(self.wave_layers):
            wave_state = layer(wave_state)
            if i == layer_idx:
                return wave_state
        
        return wave_state
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens one by one for PureWaveGPT
        """
        for _ in range(max_new_tokens):
            # Crop to block size if needed
            if idx.size(1) > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size:]
            else:
                idx_cond = idx
                
            # Forward
            logits, _ = self(idx_cond)
            
            # Select last step
            logits = logits[:, -1, :] / temperature
            
            # Top-K sampling (optional)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx # Return full sequence


class PureWaveExcitation(nn.Module):
    """
    Token → Wave Excitation with DC MODES for Global Attention.
    
    BREAKTHROUGH: Solving "Diagonal Blindness" with Pure Physics!
    
    The Problem:
    - Phase evolution θ = ω*t + φ causes Δθ = ω*(i-j) between positions
    - For non-zero ω, distant tokens have rapidly rotating phase → destructive interference
    - Result: Model can only attend locally (diagonal attention)
    
    The Solution: DC MODES (Zero-Frequency Waves)
    - If ω = 0, then Δθ = φ_i - φ_j (position-independent!)
    - Interference depends ONLY on learned phase content, not distance
    - Creates "wormholes" for instant long-range attention
    
    Wave Spectrum:
    - DC Modes (ω=0): First N_DC waves - global attention via phase content
    - Deep Bass (ω~0.001): Ultra-low freq for document-level patterns  
    - Full Spectrum: Log-spaced from 0.001 to 2.0 Hz
    """
    
    def __init__(self, vocab_size, num_waves, num_harmonics, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        self.block_size = block_size
        
        # === DC MODES: Zero-Frequency Waves for Global Attention ===
        # These waves have ω=0, so phase difference is purely content-based
        self.n_dc_modes = max(4, num_waves // 8)  # ~12.5% of waves are DC
        n_ac_modes = num_waves - self.n_dc_modes
        
        # === DEEP BASS: Lower frequency floor for longer attention ===
        # Min freq = 1/block_size ensures we can attend to full context
        min_freq = 1.0 / block_size  # e.g., 1/256 = 0.004 Hz
        min_freq = max(0.001, min_freq)  # Floor at 0.001 Hz (1000 token period)
        max_freq = 2.0  # Keep phoneme resolution
        
        # Create AC spectrum (non-zero frequencies)
        ac_spectrum = torch.logspace(math.log10(min_freq), math.log10(max_freq), n_ac_modes)
        
        # DC modes are exactly zero (will be masked from gradients)
        dc_spectrum = torch.zeros(self.n_dc_modes)
        
        # Combine: [DC modes | AC modes]
        base_spectrum = torch.cat([dc_spectrum, ac_spectrum])
        
        # EVERY token gets this spectrum
        init_freqs = base_spectrum.unsqueeze(0).expand(vocab_size, -1).clone()
        
        # Add variation ONLY to AC modes (DC must stay at zero)
        ac_variation = torch.randn(vocab_size, n_ac_modes) * 0.1
        init_freqs[:, self.n_dc_modes:] = init_freqs[:, self.n_dc_modes:] * (1.0 + ac_variation)
        
        # Ensure AC frequencies stay positive
        init_freqs[:, self.n_dc_modes:] = torch.clamp(init_freqs[:, self.n_dc_modes:], min=0.001)
        
        self.base_freqs = nn.Parameter(init_freqs)
        
        # === DC MODE MASK: Prevent gradients from drifting DC waves ===
        # This mask zeros out gradients for DC mode frequencies
        dc_mask = torch.ones(num_waves)
        dc_mask[:self.n_dc_modes] = 0.0  # DC modes: no gradient
        self.register_buffer('dc_freq_mask', dc_mask)
        
        print(f"🌊 WAVE SPECTRUM with DC MODES for Global Attention:")
        print(f"   🔴 DC Modes (ω=0): {self.n_dc_modes} waves - GLOBAL attention via phase content")
        print(f"   🟡 Deep Bass: {min_freq:.4f} Hz (period ~{1/min_freq:.0f} tokens)")
        print(f"   🟢 Full Range: {min_freq:.4f} - {max_freq:.1f} Hz ({n_ac_modes} AC waves)")
        print(f"   Total: {num_waves} waves per token")
        
        # === HARMONIC MULTIPLIERS ===
        harmonic_mults = torch.arange(1, num_harmonics + 1, dtype=torch.float32)
        self.register_buffer('harmonic_mults', harmonic_mults)
        
        # === PHASES - CRITICAL for DC modes! ===
        # For DC modes (ω=0), phase IS the only information carrier
        # Initialize DC mode phases with more variance for diversity
        init_phases = torch.rand(vocab_size, num_waves) * 2 * math.pi
        
        # DC modes get EXTRA phase diversity (they carry all the semantic info)
        dc_phase_boost = torch.randn(vocab_size, self.n_dc_modes) * 0.5
        init_phases[:, :self.n_dc_modes] = init_phases[:, :self.n_dc_modes] + dc_phase_boost
        
        self.phases = nn.Parameter(init_phases)
        
        print(f"   DC mode phases: BOOSTED variance for semantic diversity")
        print(f"   Total learnable phases: {vocab_size} × {num_waves} = {vocab_size * num_waves:,}")
        
        # === AMPLITUDES ===
        base_amps = 1.0 / harmonic_mults
        init_amps = base_amps.view(1, 1, -1).expand(vocab_size, num_waves, -1).clone()
        
        # DC modes get HIGHER initial amplitude (they're important for global attention)
        init_amps[:, :self.n_dc_modes, :] = init_amps[:, :self.n_dc_modes, :] * 1.5
        
        amp_variation = torch.randn(vocab_size, num_waves, num_harmonics) * 0.2
        init_amps = init_amps * (1.0 + amp_variation)
        init_amps = torch.clamp(init_amps, min=0.01, max=2.0)
        
        self.amplitudes = nn.Parameter(init_amps)
        
        print(f"   DC mode amplitudes: BOOSTED 1.5x for global attention")
        print(f"   Total learnable amplitudes: {vocab_size} × {num_waves} × {num_harmonics} = {vocab_size * num_waves * num_harmonics:,}")
        
        # === CROSS-WAVE COUPLING ===
        self.wave_coupling = nn.Parameter(
            torch.eye(num_waves) + torch.randn(num_waves, num_waves) * 0.05
        )
        
        # === HARMONIC COUPLING ===
        self.harmonic_coupling = nn.Parameter(
            torch.eye(num_harmonics) + torch.randn(num_harmonics, num_harmonics) * 0.05
        )
        
        total_params = (vocab_size * num_waves) + (vocab_size * num_waves) + (vocab_size * num_waves * num_harmonics)
        print(f"🎯 Total wave excitation parameters: {total_params:,} (all learnable!)")
        
        # === GRADIENT HOOK: Zero out DC mode frequency gradients ===
        # This prevents training from drifting DC modes away from ω=0
        def _zero_dc_freq_grads(grad):
            # grad shape: (vocab_size, num_waves)
            # Zero out gradients for DC mode frequencies (first n_dc_modes)
            grad_mask = self.dc_freq_mask.view(1, -1)  # (1, num_waves)
            return grad * grad_mask
        
        self.base_freqs.register_hook(_zero_dc_freq_grads)
        print(f"   ✅ Gradient hook registered: DC mode frequencies locked at ω=0")
        
    def forward(self, token_ids):
        """
        Wave excitation: Token → Full Wave Spectrum with DC MODE ENFORCEMENT.
        
        EVERY token gets ALL frequencies, ALL harmonics.
        Position encoding comes from temporal phase evolution.
        
        DC MODES (first n_dc_modes waves):
        - Frequency is EXACTLY ZERO (no position-dependent phase)
        - Phase difference Δθ = φ_i - φ_j depends ONLY on token content
        - Creates "wormholes" for instant long-range attention
        
        AC MODES (remaining waves):
        - Standard phase evolution: θ = ω*t + φ₀
        - Position-dependent attention patterns
        
        Returns: WaveState tuple (freqs, phases, amps)
        """
        B, T = token_ids.shape
        device = token_ids.device
        
        # === LOOKUP WAVE PARAMETERS ===
        # Each token has its own FULL spectrum (all frequencies, all harmonics)
        token_freqs = self.base_freqs[token_ids]      # (B, T, num_waves)
        token_phases = self.phases[token_ids]          # (B, T, num_waves)
        token_amps = self.amplitudes[token_ids]        # (B, T, num_waves, num_harmonics)
        
        # === ENFORCE DC MODE FREQUENCIES = 0 BEFORE COUPLING ===
        # DC modes must stay at exactly zero - mask them out
        dc_mask = self.dc_freq_mask.view(1, 1, -1)  # (1, 1, num_waves)
        token_freqs_masked = token_freqs * dc_mask  # DC modes → 0
        
        # === CROSS-WAVE COUPLING (AC modes only) ===
        # Waves can influence each other (learned interactions)
        # But DC modes must remain at zero after coupling
        coupled_freqs = torch.matmul(token_freqs_masked, self.wave_coupling)  # (B, T, num_waves)
        
        # === RE-ENFORCE DC MODE CONSTRAINT AFTER COUPLING ===
        # Wave coupling can leak non-zero values into DC modes - force them back to zero
        coupled_freqs = coupled_freqs * dc_mask  # DC modes → 0 again
        
        # === HARMONIC COUPLING ===
        # Harmonics can influence each other (learned harmonic relationships)
        coupled_amps = torch.matmul(token_amps, self.harmonic_coupling)  # (B, T, num_waves, num_harmonics)
        
        # === TEMPORAL PHASE EVOLUTION ===
        # This is the ONLY position encoding: φ(t) = ω*t + φ₀
        # Position in sequence = Time
        positions = torch.arange(T, device=device, dtype=torch.float32).view(1, T, 1)
        
        # Phase evolves according to wave equation
        # DC MODES: ω=0 → θ = 0*t + φ₀ = φ₀ (position-independent!)
        # AC MODES: θ = ω*t + φ₀ (position-dependent)
        evolved_phases = coupled_freqs * positions + token_phases  # (B, T, num_waves)
        
        # === FINAL FREQUENCY PROCESSING ===
        # AC modes: softplus ensures positive frequencies
        # DC modes: must stay at EXACTLY ZERO
        ac_freqs = F.softplus(coupled_freqs) + 1e-6  # All positive
        final_freqs = ac_freqs * dc_mask  # DC modes → 0, AC modes → positive
        
        # Ensure amplitudes stay positive
        final_amps = F.softplus(coupled_amps)
        
        return WaveState(final_freqs, evolved_phases, final_amps)


class WaveState:
    """Container for wave parameters - the fundamental representation."""
    def __init__(self, freqs, phases, amps):
        self.freqs = freqs    # (B, T, num_waves)
        self.phases = phases  # (B, T, num_waves)
        self.amps = amps      # (B, T, num_waves, num_harmonics)
    
    def to_phasors(self):
        """Convert to complex phasors for interference computation."""
        # Sum amplitudes across harmonics for total amplitude per wave
        total_amp = self.amps.sum(dim=-1)  # (B, T, num_waves)
        phasors = total_amp * torch.exp(1j * self.phases.to(torch.complex64))
        return phasors  # (B, T, num_waves) complex


class PureWaveLayer(nn.Module):
    """
    Pure Wave Transformer Layer with Non-Linear Physics Trinity.
    
    UPGRADED ARCHITECTURE to break the "Linearity Plateau":
    
    Wave → Wave computation:
    1. Wave Interference Attention
    2. GATED Residual Connection (PhaseInterferometer - Logic NOT via π shift)
    3. Wave FM Modulation (Context modulates Logic frequencies)
    4. Wave MLP with Saturation (4x hidden, creates harmonics)
    5. GATED Residual Connection
    
    The "Non-Linear Physics Trinity":
    - WaveSaturation: tanh overdrive INSIDE MLP creates square-wave logic
    - WaveFM: Low-freq context modulates high-freq logic frequencies
    - PhaseInterferometer: Gated residuals via destructive interference
    
    This shifts parameter mass from I/O to the Reasoning Core.
    """
    
    def __init__(
        self, 
        num_waves, 
        num_harmonics, 
        num_heads, 
        dropout=0.1,
        use_overdrive: bool = True,      # Now ON by default
        use_fm_synthesis: bool = True,   # Now ON by default
        use_interferometer: bool = True, # Now ON by default
        mlp_expansion: int = 4           # 4x hidden expansion (standard Transformer)
    ):
        super().__init__()
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
        # === WAVE INTERFERENCE ATTENTION ===
        self.wave_attention = PureWaveInterference(
            num_waves=num_waves,
            num_harmonics=num_harmonics,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # === FM MODULATION (Context → Logic frequency shift) ===
        # Low-freq waves (0 to num_waves//4) modulate high-freq waves
        self.use_fm = use_fm_synthesis
        if use_fm_synthesis:
            self.wave_fm = WaveFMContextual(num_waves, num_harmonics)
        
        # === UPGRADED MLP with Saturation & 4x Expansion ===
        self.wave_mlp = PureWaveMLP(
            num_waves=num_waves,
            num_harmonics=num_harmonics,
            dropout=dropout,
            use_overdrive=use_overdrive,
            use_fm_synthesis=False,  # FM is now at layer level
            use_interferometer=False,  # Interferometer is now gated residual
            mlp_expansion=mlp_expansion  # 4x hidden!
        )
        
        # === PHASE INTERFEROMETER GATED RESIDUALS ===
        # Replace standard residual with physics-based gating
        self.use_gated_residual = use_interferometer
        if use_interferometer:
            self.attn_gate = PhaseInterferometerGate(num_waves, num_harmonics)
            self.mlp_gate = PhaseInterferometerGate(num_waves, num_harmonics)
        
        # Wave normalization (RMS-style, preserves wave physics)
        self.norm1 = WaveRMSNorm(num_waves, num_harmonics)
        self.norm2 = WaveRMSNorm(num_waves, num_harmonics)
        
    def forward(self, wave_state: WaveState) -> WaveState:
        """Wave → Wave transformation with Non-Linear Physics Trinity."""
        
        # === ATTENTION BLOCK ===
        norm_state = self.norm1(wave_state)
        attn_state = self.wave_attention(norm_state)
        
        # GATED RESIDUAL (PhaseInterferometer) or standard residual
        if self.use_gated_residual:
            res_state = self.attn_gate(wave_state, attn_state)
        else:
            res_freqs = wave_state.freqs + attn_state.freqs
            res_phases = wave_state.phases + attn_state.phases
            res_amps = wave_state.amps + attn_state.amps
            res_state = WaveState(res_freqs, res_phases, res_amps)
        
        # === FM MODULATION (Context shifts Logic frequencies) ===
        if self.use_fm:
            res_state = self.wave_fm(res_state)
        
        # === MLP BLOCK (with internal Saturation) ===
        norm_state2 = self.norm2(res_state)
        mlp_state = self.wave_mlp(norm_state2)
        
        # GATED RESIDUAL or standard residual
        if self.use_gated_residual:
            out_state = self.mlp_gate(res_state, mlp_state)
        else:
            out_freqs = res_state.freqs + mlp_state.freqs
            out_phases = res_state.phases + mlp_state.phases
            out_amps = res_state.amps + mlp_state.amps
            out_state = WaveState(out_freqs, out_phases, out_amps)
        
        return out_state


class WaveFMContextual(nn.Module):
    """
    Contextual FM Modulation - Low frequencies modulate high frequencies.
    
    Physics: freq_new = freq_old + modulation_index * low_freq_amplitude
    
    This allows "context" (low-freq global patterns) to physically shift
    the "meaning" (frequency) of "logic" tokens (high-freq local patterns).
    
    Split: waves 0 to num_waves//4 are modulators, rest are carriers.
    """
    
    def __init__(self, num_waves, num_harmonics):
        super().__init__()
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
        # Split: 25% modulators (low freq context), 75% carriers (high freq logic)
        self.n_mod = max(1, num_waves // 4)
        self.n_carrier = num_waves - self.n_mod
        
        # Learnable modulation index per carrier
        self.mod_index = nn.Parameter(torch.ones(self.n_carrier) * 0.5)
        
        # Learnable routing: which modulators affect which carriers
        # Shape: (n_carrier, n_mod) - each carrier gets weighted sum of modulators
        self.mod_routing = nn.Parameter(
            torch.randn(self.n_carrier, self.n_mod) * 0.1
        )
        
    def forward(self, wave_state: WaveState) -> WaveState:
        """Apply contextual FM: low-freq modulates high-freq."""
        B, T, W = wave_state.freqs.shape
        
        # Split into modulators and carriers
        mod_freqs = wave_state.freqs[..., :self.n_mod]      # (B, T, n_mod)
        mod_amps = wave_state.amps[..., :self.n_mod, :]     # (B, T, n_mod, H)
        
        carrier_freqs = wave_state.freqs[..., self.n_mod:]  # (B, T, n_carrier)
        
        # Compute modulation signal: amplitude-weighted sum of modulator frequencies
        # mod_signal = Σ (routing * mod_amp * mod_freq)
        mod_amp_total = mod_amps.sum(dim=-1)  # (B, T, n_mod)
        mod_signal = mod_amp_total * mod_freqs  # (B, T, n_mod)
        
        # Route to carriers: (B, T, n_mod) @ (n_mod, n_carrier) -> (B, T, n_carrier)
        routed_mod = torch.matmul(mod_signal, self.mod_routing.T)
        
        # Apply FM: shift carrier frequencies
        # freq_new = freq_old + mod_index * modulation_signal
        modulated_freqs = carrier_freqs + self.mod_index * routed_mod
        
        # Reconstruct full frequency tensor
        new_freqs = torch.cat([mod_freqs, modulated_freqs], dim=-1)
        
        # Phases and amplitudes unchanged in basic FM
        return WaveState(new_freqs, wave_state.phases, wave_state.amps)


class PhaseInterferometerGate(nn.Module):
    """
    Phase Interferometer Gated Residual - Logic via Destructive Interference.
    
    Replaces standard residual: out = input + delta
    With gated residual: out = input + gate * delta
    
    Where gate is computed via phase interference:
    - If control_phase ≈ 0 relative to input: constructive → gate ≈ 1 (PASS)
    - If control_phase ≈ π relative to input: destructive → gate ≈ 0 (BLOCK/NOT)
    
    This enables true boolean logic via physics, not learned weights.
    """
    
    def __init__(self, num_waves, num_harmonics):
        super().__init__()
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
        # Learnable control phase per wave (initialized near 0 = pass by default)
        self.control_phase = nn.Parameter(torch.zeros(num_waves))
        
        # Gate sharpness (higher = more binary)
        self.gate_sharpness = nn.Parameter(torch.ones(1) * 5.0)
        
        # Learnable gate bias
        self.gate_bias = nn.Parameter(torch.zeros(num_waves))
        
    def forward(self, input_state: WaveState, delta_state: WaveState) -> WaveState:
        """
        Gated residual via phase interference.
        
        out = input + gate * delta
        gate = f(cos(input_phase - control_phase))
        """
        # Compute phase difference between input and control
        phase_diff = input_state.phases - self.control_phase  # (B, T, W)
        
        # Interference: cos(Δφ) gives constructive (+1) or destructive (-1)
        interference = torch.cos(phase_diff)  # (B, T, W)
        
        # Convert to gate [0, 1] via sigmoid
        # When interference = +1 (constructive): gate → 1
        # When interference = -1 (destructive): gate → 0
        gate = torch.sigmoid(
            self.gate_sharpness * (interference + self.gate_bias)
        )  # (B, T, W)
        
        # Apply gated residual
        out_freqs = input_state.freqs + gate * delta_state.freqs
        out_phases = input_state.phases + gate * delta_state.phases
        out_amps = input_state.amps + gate.unsqueeze(-1) * delta_state.amps
        
        return WaveState(out_freqs, out_phases, out_amps)
        out_freqs = res_state.freqs + mlp_state.freqs
        out_phases = res_state.phases + mlp_state.phases
        out_amps = res_state.amps + mlp_state.amps
        
        return WaveState(out_freqs, out_phases, out_amps)


class PureWaveInterference(nn.Module):
    """
    FULLY MULTI-HEAD Wave Superposition Interference.
    
    Each head has its own complete wave physics with FULL SPECTRUM:
    - Emitter projection waves (per head, all frequencies)
    - Receiver projection waves (per head, all frequencies)  
    - Field projection waves (per head, all frequencies)
    - Output projection waves (per head, all frequencies)
    
    KEY PRINCIPLE: Every head starts with ALL frequencies (0.005-2.0 Hz).
    The model LEARNS which patterns each head should specialize in.
    No artificial frequency constraints - pure learning from data!
    
    The heads are combined via learned softmax weights.
    """
    
    def __init__(self, num_waves, num_harmonics, num_heads, dropout=0.1):
        super().__init__()
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        self.num_heads = num_heads
        
        # === EMITTER PROJECTION WAVES (per head) ===
        self.emit_proj_freqs = nn.Parameter(
            self._init_projection_freqs(num_heads, num_waves)
        )
        self.emit_proj_phases = nn.Parameter(
            torch.rand(num_heads, num_waves) * 2 * math.pi
        )
        self.emit_proj_amps = nn.Parameter(
            torch.ones(num_heads, num_waves, num_harmonics) * 0.5
        )
        
        # === RECEIVER PROJECTION WAVES (per head) ===
        self.recv_proj_freqs = nn.Parameter(
            self._init_projection_freqs(num_heads, num_waves)
        )
        self.recv_proj_phases = nn.Parameter(
            torch.rand(num_heads, num_waves) * 2 * math.pi
        )
        self.recv_proj_amps = nn.Parameter(
            torch.ones(num_heads, num_waves, num_harmonics) * 0.5
        )
        
        # === FIELD PROJECTION WAVES (per head) - NOW MULTI-HEAD! ===
        self.field_proj_freqs = nn.Parameter(
            self._init_projection_freqs(num_heads, num_waves)
        )
        self.field_proj_phases = nn.Parameter(
            torch.rand(num_heads, num_waves) * 2 * math.pi
        )
        self.field_proj_amps = nn.Parameter(
            torch.ones(num_heads, num_waves, num_harmonics) * 0.5
        )
        
        # === OUTPUT PROJECTION WAVES (per head) - NOW MULTI-HEAD! ===
        self.output_proj_freqs = nn.Parameter(
            torch.zeros(num_heads, num_waves)
        )
        self.output_proj_phases = nn.Parameter(
            torch.zeros(num_heads, num_waves)
        )
        self.output_proj_amps = nn.Parameter(
            torch.zeros(num_heads, num_waves, num_harmonics)
        )
        
        # === SUPERPOSITION WEIGHTS (per head) ===
        self.emit_mix = nn.Parameter(torch.ones(num_heads) * 0.5)
        self.recv_mix = nn.Parameter(torch.ones(num_heads) * 0.5)
        self.field_mix = nn.Parameter(torch.ones(num_heads) * 0.5)
        self.output_mix = nn.Parameter(torch.ones(num_heads) * 0.5)
        
        # === HEAD COMBINATION WEIGHTS ===
        # Learned weights for combining head outputs
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        
        # === PHYSICS PARAMETERS (per head) ===
        self.interference_strength = nn.Parameter(torch.ones(num_heads))
        
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-8
        
        # Store coupling for visualization
        self.coupling = None
        
    def _init_projection_freqs(self, num_heads, num_waves):
        """
        Initialize projection wave frequencies - ALL HEADS GET FULL SPECTRUM.
        
        Every head starts with the same full frequency range (0.005 - 2.0 Hz).
        Small random perturbations break symmetry so heads can specialize.
        
        The model learns which frequencies each head should focus on!
        """
        # Full spectrum for ALL heads
        base_spectrum = torch.logspace(math.log10(0.005), math.log10(2.0), num_waves)
        
        # Every head gets the full spectrum
        freqs = base_spectrum.unsqueeze(0).expand(num_heads, -1).clone()
        
        # Small random perturbation to break symmetry (heads will diverge during training)
        freqs = freqs * (1.0 + torch.randn(num_heads, num_waves) * 0.05)
        
        return freqs
        
    def forward(self, wave_state: WaveState) -> WaveState:
        """
        FULLY MULTI-HEAD Wave Interference.
        
        Each head has independent:
        1. EMIT: Emitter projection waves
        2. RECEIVE: Receiver projection waves  
        3. FIELD: Field content waves
        4. OUTPUT: Output projection waves
        
        Heads are combined via learned weights.
        """
        B, T, W = wave_state.freqs.shape
        H = self.num_heads
        NH = self.num_harmonics
        device = wave_state.freqs.device
        
        # === STEP 1: CREATE EMITTER WAVES (per head) ===
        # Shape: (B, H, T, W)
        emit_freqs = wave_state.freqs.unsqueeze(1) + self.emit_mix.view(1, H, 1, 1) * self.emit_proj_freqs.unsqueeze(0).unsqueeze(2)
        emit_phases = wave_state.phases.unsqueeze(1) + self.emit_mix.view(1, H, 1, 1) * self.emit_proj_phases.unsqueeze(0).unsqueeze(2)
        emit_amps = wave_state.amps.unsqueeze(1) + self.emit_mix.view(1, H, 1, 1, 1) * self.emit_proj_amps.unsqueeze(0).unsqueeze(2)
        
        emit_freqs = F.softplus(emit_freqs) + self.eps
        emit_amps = F.softplus(emit_amps)
        
        # === STEP 2: CREATE RECEIVER WAVES (per head) ===
        recv_freqs = wave_state.freqs.unsqueeze(1) + self.recv_mix.view(1, H, 1, 1) * self.recv_proj_freqs.unsqueeze(0).unsqueeze(2)
        recv_phases = wave_state.phases.unsqueeze(1) + self.recv_mix.view(1, H, 1, 1) * self.recv_proj_phases.unsqueeze(0).unsqueeze(2)
        recv_amps = wave_state.amps.unsqueeze(1) + self.recv_mix.view(1, H, 1, 1, 1) * self.recv_proj_amps.unsqueeze(0).unsqueeze(2)
        
        recv_freqs = F.softplus(recv_freqs) + self.eps
        recv_amps = F.softplus(recv_amps)
        
        # === STEP 3: CREATE FIELD WAVES (per head) - NOW MULTI-HEAD! ===
        field_freqs = wave_state.freqs.unsqueeze(1) + self.field_mix.view(1, H, 1, 1) * self.field_proj_freqs.unsqueeze(0).unsqueeze(2)
        field_phases = wave_state.phases.unsqueeze(1) + self.field_mix.view(1, H, 1, 1) * self.field_proj_phases.unsqueeze(0).unsqueeze(2)
        field_amps = wave_state.amps.unsqueeze(1) + self.field_mix.view(1, H, 1, 1, 1) * self.field_proj_amps.unsqueeze(0).unsqueeze(2)
        
        field_freqs = F.softplus(field_freqs) + self.eps
        field_amps = F.softplus(field_amps)
        
        # === STEP 4: TEMPORAL PHASE EVOLUTION ===
        positions = torch.arange(T, device=device, dtype=torch.float32)
        
        # Phase at each position: θ = ω*t + φ
        emit_theta = emit_freqs * positions.view(1, 1, T, 1) + emit_phases  # (B, H, T, W)
        recv_theta = recv_freqs * positions.view(1, 1, T, 1) + recv_phases  # (B, H, T, W)
        
        # === STEP 5: WAVE INTERFERENCE (per head) - OPTIMIZED REAL-ONLY ===
        # Avoid complex tensor overhead by using trigonometric identity:
        # Re(A*e^(iθ₁) @ B*e^(-iθ₂)) = A*B*cos(θ₁-θ₂) = A*B*(cos(θ₁)cos(θ₂) + sin(θ₁)sin(θ₂))
        # This is equivalent to: (A*cos(θ₁)) @ (B*cos(θ₂))ᵀ + (A*sin(θ₁)) @ (B*sin(θ₂))ᵀ
        
        emit_amp_total = emit_amps.sum(dim=-1)  # (B, H, T, W)
        recv_amp_total = recv_amps.sum(dim=-1)
        
        # Real-only phasor components (no complex tensors!)
        emit_cos = emit_amp_total * torch.cos(emit_theta)  # (B, H, T, W)
        emit_sin = emit_amp_total * torch.sin(emit_theta)  # (B, H, T, W)
        recv_cos = recv_amp_total * torch.cos(recv_theta)  # (B, H, T, W)
        recv_sin = recv_amp_total * torch.sin(recv_theta)  # (B, H, T, W)
        
        # Interference = Re(emit @ recv*) using real arithmetic only
        # This avoids complex64 tensor creation and conjugate operations
        interference = (
            torch.matmul(emit_cos, recv_cos.transpose(-2, -1)) +
            torch.matmul(emit_sin, recv_sin.transpose(-2, -1))
        ) / (W ** 0.5)  # (B, H, T, T)
        
        # === STEP 6: FULL INTENSITY with per-head strength ===
        emit_energy = (emit_amp_total ** 2).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        recv_energy = (recv_amp_total ** 2).sum(dim=-1, keepdim=True).transpose(-2, -1)  # (B, H, 1, T)
        
        # Per-head interference strength
        intensity = emit_energy + recv_energy + 2 * interference * self.interference_strength.view(1, H, 1, 1)
        
        # === STEP 7: PHYSICS-BASED NORMALIZATION ===
        emit_total = emit_amp_total.sum(dim=-1, keepdim=True)
        recv_total = recv_amp_total.sum(dim=-1, keepdim=True).transpose(-2, -1)
        max_intensity = (emit_total + recv_total) ** 2 + self.eps
        
        coupling = intensity / max_intensity  # (B, H, T, T)
        
        # === STEP 8: CAUSAL MASK (Light Cone) ===
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        coupling = coupling.masked_fill(causal_mask, 0.0)
        
        # Normalize coupling weights per head
        coupling = coupling / (coupling.sum(dim=-1, keepdim=True) + self.eps)
        coupling = self.dropout(coupling)
        
        # Store for visualization
        self.coupling = coupling.detach()
        
        # === STEP 9: WAVE SUPERPOSITION (per head) ===
        # Apply coupling to field content (field is now per-head!)
        # field_freqs: (B, H, T, W), coupling: (B, H, T, T)
        out_freqs = torch.matmul(coupling, field_freqs)  # (B, H, T, W)
        out_phases = torch.matmul(coupling, field_phases)  # (B, H, T, W)
        
        # Handle amplitudes: (B, H, T, W, NH) -> flatten -> matmul -> reshape
        field_amps_flat = field_amps.view(B, H, T, -1)  # (B, H, T, W*NH)
        out_amps_flat = torch.matmul(coupling, field_amps_flat)  # (B, H, T, W*NH)
        out_amps = out_amps_flat.view(B, H, T, W, NH)  # (B, H, T, W, NH)
        
        # === STEP 10: ADD OUTPUT PROJECTION (per head) ===
        out_freqs = out_freqs + self.output_mix.view(1, H, 1, 1) * self.output_proj_freqs.unsqueeze(0).unsqueeze(2)
        out_phases = out_phases + self.output_mix.view(1, H, 1, 1) * self.output_proj_phases.unsqueeze(0).unsqueeze(2)
        out_amps = out_amps + self.output_mix.view(1, H, 1, 1, 1) * self.output_proj_amps.unsqueeze(0).unsqueeze(2)
        
        # === STEP 11: COMBINE HEADS via learned weights ===
        # Normalize head weights
        head_w = F.softmax(self.head_weights, dim=0).view(1, H, 1, 1)
        
        final_freqs = (out_freqs * head_w).sum(dim=1)  # (B, T, W)
        final_phases = (out_phases * head_w).sum(dim=1)  # (B, T, W)
        final_amps = (out_amps * head_w.unsqueeze(-1)).sum(dim=1)  # (B, T, W, NH)
        
        return WaveState(final_freqs, final_phases, final_amps)


# ==========================================
# Gibbs Phenomenon Solutions: Sharp Logic via Physics
# ==========================================

class WaveOverdrive(nn.Module):
    """
    Overdrive Mechanism (Wave Saturation) - Solves Gibbs Phenomenon.
    
    Physics Principle: Analog saturation/clipping (like guitar amp overdrive).
    
    Mechanism: Push wave amplitudes through soft-clipping (tanh with learnable gain).
    This turns smooth sine waves into approximate SQUARE WAVES (pulse trains).
    
    Goal: Physical way to represent binary 0/1 states using saturated waves.
    
    The key insight: tanh(gain * x) approaches sign(x) as gain → ∞
    - Low gain: smooth, linear response
    - High gain: sharp, square-wave-like response
    """
    
    def __init__(self, num_waves, num_harmonics):
        super().__init__()
        
        # Learnable gain per wave (controls sharpness)
        # Initialize to moderate values - training will find optimal sharpness
        self.amp_gain = nn.Parameter(torch.ones(num_waves, num_harmonics) * 2.0)
        self.freq_gain = nn.Parameter(torch.ones(num_waves) * 1.5)
        
        # Learnable bias (DC offset before saturation)
        self.amp_bias = nn.Parameter(torch.zeros(num_waves, num_harmonics))
        self.freq_bias = nn.Parameter(torch.zeros(num_waves))
        
        # Output scaling (compensate for tanh compression)
        self.amp_scale = nn.Parameter(torch.ones(num_waves, num_harmonics))
        self.freq_scale = nn.Parameter(torch.ones(num_waves))
        
    def forward(self, wave_state: WaveState) -> WaveState:
        """Apply soft-clipping saturation to create sharp transitions."""
        # Saturate amplitudes: smooth → square-ish
        saturated_amps = torch.tanh(
            self.amp_gain * (wave_state.amps + self.amp_bias)
        ) * self.amp_scale
        
        # Saturate frequencies (optional, creates frequency quantization)
        saturated_freqs = torch.tanh(
            self.freq_gain * (wave_state.freqs + self.freq_bias)
        ) * self.freq_scale
        
        # Phases pass through unchanged (saturation doesn't apply to angles)
        return WaveState(saturated_freqs, wave_state.phases, saturated_amps)


class WaveFMSynthesis(nn.Module):
    """
    FM Synthesis (Frequency Modulation) - Solves Gibbs Phenomenon.
    
    Physics Principle: Frequency Modulation (like FM radio, Yamaha DX7).
    
    Mechanism: Low-frequency "modulator" waves modulate the frequency of 
    high-frequency "carrier" waves. This creates complex sidebands and
    sharp transients INSTANTLY without infinite harmonics.
    
    Goal: Context waves (low freq) can dynamically switch logic waves (high freq).
    
    FM equation: y(t) = A * sin(ω_c*t + β*sin(ω_m*t))
    where β (modulation index) controls sideband richness.
    """
    
    def __init__(self, num_waves, num_harmonics):
        super().__init__()
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
        # Split waves into modulators (low freq) and carriers (high freq)
        self.n_modulators = num_waves // 3  # ~33% are modulators
        self.n_carriers = num_waves - self.n_modulators
        
        # Modulation index (β) - controls sideband richness
        # Higher β = more sidebands = sharper transients
        self.mod_index = nn.Parameter(torch.ones(self.n_carriers) * 2.0)
        
        # Modulation routing: which modulators affect which carriers
        # Learnable mixing matrix
        self.mod_routing = nn.Parameter(
            torch.randn(self.n_carriers, self.n_modulators) * 0.5
        )
        
        # Carrier frequency scaling
        self.carrier_scale = nn.Parameter(torch.ones(self.n_carriers))
        
    def forward(self, wave_state: WaveState) -> WaveState:
        """Apply FM synthesis: modulators modulate carrier frequencies."""
        B, T, W = wave_state.freqs.shape
        
        # Split into modulators and carriers
        mod_freqs = wave_state.freqs[..., :self.n_modulators]  # (B, T, n_mod)
        mod_phases = wave_state.phases[..., :self.n_modulators]
        mod_amps = wave_state.amps[..., :self.n_modulators, :]
        
        carrier_freqs = wave_state.freqs[..., self.n_modulators:]  # (B, T, n_car)
        carrier_phases = wave_state.phases[..., self.n_modulators:]
        carrier_amps = wave_state.amps[..., self.n_modulators:, :]
        
        # Compute modulator signal: sum of modulator waves
        # mod_signal = Σ A_m * sin(φ_m)  (simplified, using current phase)
        mod_signal = (mod_amps.sum(dim=-1) * torch.sin(mod_phases))  # (B, T, n_mod)
        
        # Route modulation to carriers via learned routing matrix
        # (B, T, n_mod) @ (n_mod, n_car).T -> (B, T, n_car)
        routed_mod = torch.matmul(mod_signal, self.mod_routing.T)
        
        # Apply FM: modulate carrier phases
        # New phase = original_phase + β * modulator_signal
        fm_phase_shift = self.mod_index * routed_mod  # (B, T, n_car)
        modulated_phases = carrier_phases + fm_phase_shift
        
        # Scale carrier frequencies (FM also affects perceived frequency)
        modulated_freqs = carrier_freqs * self.carrier_scale
        
        # Reconstruct full wave state
        new_freqs = torch.cat([mod_freqs, modulated_freqs], dim=-1)
        new_phases = torch.cat([mod_phases, modulated_phases], dim=-1)
        new_amps = wave_state.amps  # Amplitudes unchanged in basic FM
        
        return WaveState(new_freqs, new_phases, new_amps)


class WaveInterferometerGate(nn.Module):
    """
    Interferometer Gate (Optical Logic) - Solves Gibbs Phenomenon.
    
    Physics Principle: Mach-Zehnder Interferometer (constructive/destructive interference).
    
    Mechanism: A "control" wave shifts a "signal" wave's phase by π (180°).
    - Phase aligned (Δφ ≈ 0): Constructive interference → signal PASSES (Logic TRUE)
    - Phase opposed (Δφ ≈ π): Destructive interference → signal CANCELS (Logic FALSE)
    
    Goal: True boolean logic (AND/OR/NOT) via phase cancellation, not multiplication.
    
    This is how optical computers work! Pure physics, no artificial gating.
    """
    
    def __init__(self, num_waves, num_harmonics):
        super().__init__()
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
        # Split waves into control and signal
        self.n_control = num_waves // 2
        self.n_signal = num_waves - self.n_control
        
        # Learnable control-to-signal routing
        # Determines which control waves gate which signal waves
        self.gate_routing = nn.Parameter(
            torch.eye(min(self.n_control, self.n_signal)).unsqueeze(0)
        )
        
        # Phase shift amount (learnable, initialized to π for full cancellation)
        self.phase_shift = nn.Parameter(torch.ones(self.n_signal) * math.pi)
        
        # Gate sharpness (how binary the gating is)
        # Higher = sharper transition between pass/block
        self.gate_sharpness = nn.Parameter(torch.ones(self.n_signal) * 5.0)
        
        # Bias for gate activation
        self.gate_bias = nn.Parameter(torch.zeros(self.n_signal))
        
    def forward(self, wave_state: WaveState) -> WaveState:
        """Apply interferometer gating: control waves gate signal waves."""
        B, T, W = wave_state.freqs.shape
        
        # Split into control and signal waves
        ctrl_amps = wave_state.amps[..., :self.n_control, :]  # (B, T, n_ctrl, H)
        ctrl_phases = wave_state.phases[..., :self.n_control]  # (B, T, n_ctrl)
        
        sig_freqs = wave_state.freqs[..., self.n_control:]  # (B, T, n_sig)
        sig_phases = wave_state.phases[..., self.n_control:]
        sig_amps = wave_state.amps[..., self.n_control:, :]
        
        # Compute control signal strength (total amplitude of control waves)
        ctrl_strength = ctrl_amps.sum(dim=-1)  # (B, T, n_ctrl)
        
        # Route control to signal via gate routing
        # Expand routing for batch: (1, n_ctrl, n_sig) or learned
        if self.gate_routing.shape[0] == 1:
            routing = self.gate_routing.expand(1, self.n_control, self.n_signal)
        else:
            routing = self.gate_routing
            
        # Pad routing if dimensions don't match
        if routing.shape[1] < self.n_control:
            routing = F.pad(routing, (0, 0, 0, self.n_control - routing.shape[1]))
        if routing.shape[2] < self.n_signal:
            routing = F.pad(routing, (0, self.n_signal - routing.shape[2]))
        routing = routing[:, :self.n_control, :self.n_signal]
        
        # Compute gate activation: how much each control affects each signal
        # (B, T, n_ctrl) @ (1, n_ctrl, n_sig) -> (B, T, n_sig)
        gate_activation = torch.matmul(
            ctrl_strength.unsqueeze(-2),  # (B, T, 1, n_ctrl)
            routing.transpose(-2, -1)      # (1, n_sig, n_ctrl) -> need (1, n_ctrl, n_sig)
        ).squeeze(-2)  # (B, T, n_sig)
        
        # Apply sigmoid for smooth gating (0 = block, 1 = pass)
        # gate = σ(sharpness * (activation - bias))
        gate = torch.sigmoid(
            self.gate_sharpness * (gate_activation + self.gate_bias)
        )  # (B, T, n_sig)
        
        # Interferometer logic:
        # - gate ≈ 1: no phase shift → constructive → signal passes
        # - gate ≈ 0: π phase shift → destructive → signal cancels
        
        # Phase shift based on gate (inverted: low gate = high shift)
        phase_shift = (1 - gate) * self.phase_shift  # (B, T, n_sig)
        
        # Apply phase shift to signal
        gated_phases = sig_phases + phase_shift
        
        # Amplitude modulation via interference
        # I = A₁² + A₂² + 2A₁A₂cos(Δφ)
        # When Δφ = π, cos(π) = -1, so I = (A₁ - A₂)² → cancellation
        # Simplified: amplitude scales with cos(phase_shift/2)²
        interference_factor = torch.cos(phase_shift / 2).unsqueeze(-1) ** 2
        gated_amps = sig_amps * interference_factor
        
        # Reconstruct: control waves pass through, signal waves are gated
        ctrl_freqs = wave_state.freqs[..., :self.n_control]
        ctrl_amps_full = wave_state.amps[..., :self.n_control, :]  # (B, T, n_ctrl, H)
        
        new_freqs = torch.cat([ctrl_freqs, sig_freqs], dim=-1)  # (B, T, W)
        new_phases = torch.cat([wave_state.phases[..., :self.n_control], gated_phases], dim=-1)  # (B, T, W)
        new_amps = torch.cat([ctrl_amps_full, gated_amps], dim=-2)  # (B, T, W, H) - concat on wave dim!
        
        return WaveState(new_freqs, new_phases, new_amps)


class PureWaveMLP(nn.Module):
    """
    Wave-Native MLP with Non-Linear Physics - UPGRADED for Expressivity.
    
    KEY CHANGES to break "Linearity Plateau":
    1. 4x hidden expansion (shifts params from I/O to reasoning core)
    2. Saturation (tanh overdrive) INSIDE the MLP creates harmonics
    3. Cross-parameter mixing (freqs influence phases, etc.)
    
    Architecture per parameter type:
        input → Linear(D, 4D) → Saturation → GELU → Linear(4D, D) → output
    
    The Saturation layer creates square-wave-like responses that enable
    sharp, contextual decisions instead of smooth interpolation.
    """
    
    def __init__(
        self, 
        num_waves, 
        num_harmonics, 
        dropout=0.1,
        use_overdrive: bool = True,      # Saturation ON by default
        use_fm_synthesis: bool = False,  # FM now at layer level
        use_interferometer: bool = False, # Interferometer now gated residual
        mlp_expansion: int = 4           # 4x hidden expansion
    ):
        super().__init__()
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        self.use_overdrive = use_overdrive
        
        # Hidden dimensions with 4x expansion
        freq_hidden = num_waves * mlp_expansion
        phase_hidden = num_waves * mlp_expansion
        amp_dim = num_waves * num_harmonics
        amp_hidden = amp_dim * mlp_expansion
        
        # === FREQUENCY MLP with Saturation ===
        self.freq_up = nn.Linear(num_waves, freq_hidden)
        self.freq_down = nn.Linear(freq_hidden, num_waves)
        
        # === PHASE MLP with Saturation ===
        self.phase_up = nn.Linear(num_waves, phase_hidden)
        self.phase_down = nn.Linear(phase_hidden, num_waves)
        
        # === AMPLITUDE MLP with Saturation (largest - most params) ===
        self.amp_up = nn.Linear(amp_dim, amp_hidden)
        self.amp_down = nn.Linear(amp_hidden, amp_dim)
        
        # === CROSS-PARAMETER MIXING ===
        # Frequencies influence phases (physical: freq determines phase evolution)
        self.freq_to_phase = nn.Linear(num_waves, num_waves)
        # Amplitudes influence frequencies (physical: energy affects resonance)
        self.amp_to_freq = nn.Linear(amp_dim, num_waves)
        
        # === SATURATION (Overdrive) PARAMETERS ===
        if use_overdrive:
            # Learnable gain per hidden unit (controls sharpness)
            self.freq_gain = nn.Parameter(torch.ones(freq_hidden) * 2.0)
            self.phase_gain = nn.Parameter(torch.ones(phase_hidden) * 2.0)
            self.amp_gain = nn.Parameter(torch.ones(amp_hidden) * 2.0)
            
            # Learnable bias before saturation
            self.freq_sat_bias = nn.Parameter(torch.zeros(freq_hidden))
            self.phase_sat_bias = nn.Parameter(torch.zeros(phase_hidden))
            self.amp_sat_bias = nn.Parameter(torch.zeros(amp_hidden))
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize for stability
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training with proper scale."""
        with torch.no_grad():
            # Standard Xavier/Glorot initialization for up projections
            # This ensures gradients flow properly through the network
            for layer in [self.freq_up, self.phase_up, self.amp_up]:
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            
            # Smaller for down projections (residual-friendly, but not too small)
            for layer in [self.freq_down, self.phase_down, self.amp_down]:
                nn.init.xavier_uniform_(layer.weight)
                layer.weight.data *= 0.5  # Slightly smaller for residual
                if layer.bias is not None:
                    layer.bias.data.zero_()
            
            # Cross-parameter mixing - moderate initialization
            nn.init.xavier_uniform_(self.freq_to_phase.weight)
            self.freq_to_phase.weight.data *= 0.1
            nn.init.xavier_uniform_(self.amp_to_freq.weight)
            self.amp_to_freq.weight.data *= 0.1
        
    def _saturate(self, x, gain, bias):
        """
        Apply tanh saturation with learnable gain.
        
        tanh(gain * (x + bias)) approaches sign(x) as gain → ∞
        This creates square-wave-like responses for sharp logic.
        """
        return torch.tanh(gain * (x + bias))
    
    def forward(self, wave_state: WaveState) -> WaveState:
        B, T, W = wave_state.freqs.shape
        H = self.num_harmonics
        
        # Flatten amplitudes
        amps_flat = wave_state.amps.view(B, T, -1)  # (B, T, W*H)
        
        # === FREQUENCY PATH ===
        # Up project
        freq_h = self.freq_up(wave_state.freqs)  # (B, T, freq_hidden)
        
        # Saturation (creates harmonics, enables sharp decisions)
        if self.use_overdrive:
            freq_h = self._saturate(freq_h, self.freq_gain, self.freq_sat_bias)
        
        # Non-linearity
        freq_h = F.gelu(freq_h)
        freq_h = self.dropout(freq_h)
        
        # Down project
        new_freqs = self.freq_down(freq_h)  # (B, T, W)
        
        # Cross-parameter: amplitudes influence frequencies
        amp_influence = self.amp_to_freq(amps_flat)
        new_freqs = new_freqs + 0.1 * amp_influence
        
        # === PHASE PATH ===
        phase_h = self.phase_up(wave_state.phases)
        
        if self.use_overdrive:
            phase_h = self._saturate(phase_h, self.phase_gain, self.phase_sat_bias)
        
        phase_h = F.gelu(phase_h)
        phase_h = self.dropout(phase_h)
        
        new_phases = self.phase_down(phase_h)
        
        # Cross-parameter: frequencies influence phases
        freq_influence = self.freq_to_phase(wave_state.freqs)
        new_phases = new_phases + 0.1 * freq_influence
        
        # === AMPLITUDE PATH (largest, most expressive) ===
        amp_h = self.amp_up(amps_flat)
        
        if self.use_overdrive:
            amp_h = self._saturate(amp_h, self.amp_gain, self.amp_sat_bias)
        
        amp_h = F.gelu(amp_h)
        amp_h = self.dropout(amp_h)
        
        new_amps_flat = self.amp_down(amp_h)
        new_amps = new_amps_flat.view(B, T, W, H)
        
        return WaveState(new_freqs, new_phases, new_amps)


class WaveRMSNorm(nn.Module):
    """RMS Normalization for wave parameters."""
    
    def __init__(self, num_waves, num_harmonics):
        super().__init__()
        self.freq_scale = nn.Parameter(torch.ones(num_waves))
        self.phase_scale = nn.Parameter(torch.ones(num_waves))
        self.amp_scale = nn.Parameter(torch.ones(num_waves, num_harmonics))
        self.eps = 1e-8
        
    def forward(self, wave_state: WaveState) -> WaveState:
        # RMS normalize each parameter type
        freq_rms = wave_state.freqs.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        norm_freqs = wave_state.freqs / freq_rms * self.freq_scale
        
        phase_rms = wave_state.phases.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        norm_phases = wave_state.phases / phase_rms * self.phase_scale
        
        amp_rms = wave_state.amps.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=self.eps)
        norm_amps = wave_state.amps / amp_rms * self.amp_scale
        
        return WaveState(norm_freqs, norm_phases, norm_amps)


class WaveCollapse(nn.Module):
    """
    SIMPLIFIED Wave → Logits (Measurement/Collapse) - Parameter Efficient.
    
    CRITICAL FIX: The previous implementation was parameter-heavy (91.6% of model!).
    This simplified version shifts parameters to the reasoning core.
    
    Simple approach:
    1. Normalize wave parameters to reasonable ranges
    2. Concatenate all wave info into single vector
    3. Single linear projection to vocabulary (minimal parameters)
    
    This forces the model to do reasoning in the wave layers, not the head.
    """
    
    def __init__(self, vocab_size, num_waves, num_harmonics):
        super().__init__()
        
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        
        # === MINIMAL PROCESSING ===
        # Just normalize and concatenate - no heavy processing
        wave_dim = num_waves + num_waves + num_waves * num_harmonics  # freqs + phases + amps
        self.wave_norm = nn.LayerNorm(wave_dim)
        
        # === SINGLE PROJECTION (Minimal Parameters) ===
        # Direct projection with small bottleneck to minimize head parameters
        bottleneck_dim = min(256, wave_dim // 2)  # Very small bottleneck
        
        self.collapse_proj = nn.Sequential(
            nn.Linear(wave_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, vocab_size)
        )
        
        # Initialize with smaller weights for stability
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with proper scaling for gradient flow."""
        for layer in self.collapse_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                # Scale down for stable logits
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()
        
    def forward(self, wave_state: WaveState) -> torch.Tensor:
        B, T, W = wave_state.freqs.shape
        
        # === SIMPLE NORMALIZATION ===
        # Normalize wave parameters to reasonable ranges
        freq_norm = torch.tanh(wave_state.freqs * 0.1)  # Normalize frequencies
        phase_norm = torch.sin(wave_state.phases)        # Normalize phases to [-1, 1]
        amp_norm = torch.clamp(wave_state.amps, min=0.0, max=2.0)  # Clamp amplitudes
        
        # Flatten amplitudes
        amp_flat = amp_norm.view(B, T, -1)  # (B, T, W*H)
        
        # === CONCATENATE ALL WAVE PARAMETERS ===
        # Simple concatenation - let the reasoning layers do the work!
        wave_vector = torch.cat([
            freq_norm,    # (B, T, W)
            phase_norm,   # (B, T, W)  
            amp_flat      # (B, T, W*H)
        ], dim=-1)  # (B, T, W + W + W*H)
        
        # === MINIMAL PROJECTION ===
        # Normalize and project to vocabulary with minimal parameters
        wave_vector = self.wave_norm(wave_vector)
        logits = self.collapse_proj(wave_vector)  # (B, T, vocab_size)
        
        return logits


class WaveGPT(nn.Module):
    """
    Wave-Native GPT with independently toggleable physics components.
    
    Component Independence (Requirement 7.5):
    - use_wave_embeddings: Toggle WavePacketEmbedding vs StandardEmbedding
    - use_interference_attention: Toggle physics-based InterferenceAttention
    - pure_wave_attention: Toggle PureWaveAttention (no softmax)
    - model_type: "wave" uses WaveBlock, "standard" uses StandardBlock
    
    Each component can be toggled independently without breaking model functionality.
    """
    def __init__(self, config: WaveGPTConfig):
        super().__init__()
        self.config = config
        
        # Determine embedding type (Requirement 7.5: independent toggle)
        use_wave_embed = getattr(config, 'use_wave_embeddings', True)
        
        if config.model_type == "standard":
            # --- Standard Transformer ---
            # Standard model always uses standard embeddings
            self.embedding = StandardEmbedding(
                config.vocab_size, config.d_model, config.block_size, config.dropout
            )
            self.blocks = nn.ModuleList([
                StandardBlock(config) for _ in range(config.num_layers)
            ])
            self.ln_f = nn.LayerNorm(config.d_model) # Final LN
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            
        else:
            # --- Wave-Native Transformer ---
            # Embedding type can be toggled independently (Requirement 7.5)
            if use_wave_embed:
                self.embedding = WavePacketEmbedding(
                    config.vocab_size, config.d_model, 
                    config.num_waves, config.num_harmonics
                )
            else:
                # Use standard embedding with wave attention blocks
                self.embedding = StandardEmbedding(
                    config.vocab_size, config.d_model, config.block_size, config.dropout
                )
            
            self.blocks = nn.ModuleList([
                WaveBlock(config) for _ in range(config.num_layers)
            ])
            self.ln_f = nn.LayerNorm(config.d_model)
            self.head = CollapseHead(config.d_model, config.vocab_size)
        
        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, standard_embed_ratio=0.0):
        """
        Forward pass with optional embedding annealing.
        
        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target token indices (optional)
            standard_embed_ratio: Mixing ratio for standard embeddings (0.0 = pure wave, 1.0 = pure standard)
                                  Used for annealing during training (Requirement 1.5)
        
        Returns:
            logits: (B, T, vocab_size) output logits
            loss: scalar loss if targets provided, else None
        """
        B, T = idx.shape
        
        # 1. Embedding (with optional annealing for wave embeddings)
        # Check if embedding is WavePacketEmbedding (supports standard_embed_ratio)
        if isinstance(self.embedding, WavePacketEmbedding):
            # WavePacketEmbedding supports standard_embed_ratio for annealing
            x = self.embedding(idx, standard_embed_ratio=standard_embed_ratio)
        else:
            # StandardEmbedding doesn't use standard_embed_ratio
            x = self.embedding(idx)
        
        # 2. Blocks
        for block in self.blocks:
            x = block(x)
            
        # 3. Final LN
        x = self.ln_f(x)
        
        # 4. Collapse/Head
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens one by one
        """
        for _ in range(max_new_tokens):
            # Crop to block size if needed
            if idx.size(1) > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size:]
            else:
                idx_cond = idx
                
            # Forward
            logits, _ = self(idx_cond)
            
            # Select last step
            logits = logits[:, -1, :] / temperature
            
            # Top-K sampling (optional)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx # Return full sequence

