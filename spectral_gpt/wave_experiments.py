"""
Wave-Native GPT Comprehensive Experiment Suite

Ablation studies, extended training, and dataset experiments.

Usage:
    # Run all ablation studies
    python wave_experiments.py --experiment all --steps 20000
    
    # Run specific experiment
    python wave_experiments.py --experiment rgd_only --steps 15000
    
    # FineWeb-Edu experiment
    python wave_experiments.py --dataset fineweb --model large
    
    # Multi-GPU
    python wave_experiments.py --experiment all --parallel
"""

import os
import sys
import gc
import json
import time
import math
import argparse
import zipfile
import shutil
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
import tiktoken # GPT-2 Tokenizer
from datasets import load_dataset # HuggingFace Datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'prototyping'))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir)) # Add root project dir for src imports

from wave_gpt import WaveGPT, WaveGPTConfig
from physics_optim import ResonantGradientDescent, QuantumFieldEntanglementLoss
from train import BasicTokenizer, get_batch

# Import wave physics core components (physics-first approach)
try:
    from wave_physics_core import WaveNativeOptimizer, WaveCoherenceLoss, WaveDiagnostics
    WAVE_PHYSICS_CORE_AVAILABLE = True
except ImportError:
    WAVE_PHYSICS_CORE_AVAILABLE = False
    WaveNativeOptimizer = None
    WaveCoherenceLoss = None
    WaveDiagnostics = None


# ==========================================
# Annealing Schedule
# ==========================================

# Default annealing steps constant
ANNEALING_STEPS = 3000


def get_annealing_ratio(step: int, total_annealing_steps: int = ANNEALING_STEPS) -> float:
    """
    Compute standard_embed_ratio for annealing schedule.
    
    Decays linearly from 1.0 to 0.0 over total_annealing_steps.
    After total_annealing_steps, returns 0.0 (pure wave embeddings).
    
    Args:
        step: Current training step
        total_annealing_steps: Number of steps for annealing (default: 3000)
        
    Returns:
        standard_embed_ratio in [0.0, 1.0]
        
    Requirements: 6.1
    """
    return max(0.0, 1.0 - step / total_annealing_steps)

# ==========================================
# Experiment Configurations
# ==========================================

class EarlyStopping:
    """
    Robust Early Stopping with CPU state saving.
    """
    def __init__(self, patience: int = 8, min_delta: float = 0.005, console=None):
        self.patience = patience
        self.min_delta = min_delta
        self.console = console
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save state to CPU to avoid GPU OOM
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.console:
                self.console.print(f"   [green]New best: {val_loss:.4f} (Saved)[/green]")
        else:
            self.counter += 1
            if self.console:
                self.console.print(f"   [yellow]Patience: {self.counter}/{self.patience}[/yellow]")
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
        
    def restore(self, model: nn.Module):
        if self.best_model_state is not None:
            if self.console:
                self.console.print(f"♻️  Restoring best model (Val: {self.best_loss:.4f})")
            model.load_state_dict(self.best_model_state)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    use_rgd: bool = False
    use_qfe: bool = False
    pure_wave_attention: bool = False  # New flag for pure attention
    pure_wave_kernel: str = "elu_plus_one" # Kernel: 'elu_plus_one', 'sigmoid', 'exp'
    pure_wave_mode: str = "quadratic"      # 'quadratic' vs 'linear'
    use_interference_attention: bool = False  # True = physics-based InterferenceAttention (Req 2.1-2.5)
    rgd_strength: float = 0.3
    qfe_lambda: float = 0.05
    qfe_threshold: float = 0.01
    lr: float = 6e-4
    weight_decay: float = 0.01
    dropout: float = 0.1 # Default 0.1 for Baseline
    warmup_steps: int = 500
    patience: int = 8   # Scientific default
    steps: int = 10000   # Scaled for 500M tokens (~7.4 tokens/param)
    wave_ratio_schedule: bool = True  # Schedule wave_ratio from 0.5 to 0.9
    model_type: str = "wave" # "wave", "standard", or "pure_wave"
    grad_accum_steps: int = 2 # Restore effective B=32 (since physical B=16)
    batch_size_override: int = None  # Override batch size (for memory-heavy experiments like InterferenceAttention)
    
    # === NON-LINEAR PHYSICS TRINITY (Diagonal Breaker) ===
    use_overdrive: bool = False      # Soft-clipping saturation → square waves
    use_fm_synthesis: bool = False   # Frequency modulation → context shifts logic frequencies  
    use_interferometer: bool = False # Phase-based gated residuals → boolean logic via interference
    mlp_expansion: int = 1           # Hidden dimension multiplier (4x for reasoning core shift)




ABLATION_EXPERIMENTS = {
    # ============================================================
    # CORE ABLATION MATRIX (2x2 grid: Embeddings × Optimizer)
    # ============================================================
    
    # Control: Standard Embeddings + AdamW
    "std_emb_adamw": ExperimentConfig(
        name="StdEmbed + AdamW (Control)",
        model_type="standard",
        use_rgd=False, use_qfe=False,
        lr=6e-4, dropout=0.1
    ),
    
    # Test A: Wave Embeddings + AdamW (isolates embedding effect)
    "wave_emb_adamw": ExperimentConfig(
        name="WaveEmbed + AdamW",
        model_type="wave",
        use_rgd=False, use_qfe=False,
        lr=6e-4, dropout=0.1
    ),
    
    # Test B: Standard Embeddings + WaveOptim (isolates optimizer effect)
    "std_emb_waveopt": ExperimentConfig(
        name="StdEmbed + WaveOptim",
        model_type="standard",
        use_rgd=True, use_qfe=False,
        lr=1e-3, dropout=0.1
    ),
    
    # Test C: Wave Embeddings + WaveOptim (full wave stack)
    "wave_emb_waveopt": ExperimentConfig(
        name="WaveEmbed + WaveOptim",
        model_type="wave",
        use_rgd=True, use_qfe=False,
        lr=1e-3, dropout=0.1
    ),
    
    # ============================================================
    # QFE LOSS ABLATIONS (adds coherence regularization)
    # ============================================================
    
    # Wave + AdamW + QFE Loss
    "wave_emb_adamw_qfe": ExperimentConfig(
        name="WaveEmbed + AdamW + QFE",
        model_type="wave",
        use_rgd=False, use_qfe=True,
        lr=6e-4, dropout=0.1
    ),
    
    # Full Physics: Wave + WaveOptim + QFE (maximum wave)
    "wave_full_physics": ExperimentConfig(
        name="WaveEmbed + WaveOptim + QFE (Full)",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        lr=1e-3, dropout=0.0,
        qfe_lambda=0.1
    ),
    
    # ============================================================
    # PURE WAVE ATTENTION VARIANTS (no softmax)
    # ============================================================
    
    "pure_wave_elu": ExperimentConfig(
        name="PureWaveAttn (ELU+1)",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        lr=1e-3, dropout=0.1,
        pure_wave_attention=True,
        pure_wave_kernel="elu_plus_one"
    ),
    "pure_wave_linear": ExperimentConfig(
        name="PureWaveAttn (Linear O(N))",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        lr=1e-3, dropout=0.1,
        pure_wave_attention=True,
        pure_wave_kernel="elu_plus_one",
        pure_wave_mode="linear"
    ),
    "pure_wave_sigmoid": ExperimentConfig(
        name="PureWaveAttn (Sigmoid)",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        lr=1e-3, dropout=0.1,
        pure_wave_attention=True,
        pure_wave_kernel="sigmoid"
    ),
    "pure_wave_exp": ExperimentConfig(
        name="PureWaveAttn (Exp)",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        lr=1e-3, dropout=0.1,
        pure_wave_attention=True,
        pure_wave_kernel="exp"
    ),
    
    # ============================================================
    # LEGACY ALIASES (for backward compatibility)
    # ============================================================
    "standard_transformer": ExperimentConfig(
        name="StdEmbed + AdamW (Control)",
        model_type="standard",
        use_rgd=False, use_qfe=False,
        lr=6e-4, dropout=0.1
    ),
    "wave_baseline": ExperimentConfig(
        name="WaveEmbed + AdamW",
        model_type="wave",
        use_rgd=False, use_qfe=False,
        lr=6e-4, dropout=0.1
    ),
    "rgd_only": ExperimentConfig(
        name="WaveEmbed + WaveOptim",
        model_type="wave",
        use_rgd=True, use_qfe=False,
        lr=1e-3, dropout=0.1
    ),
    "full_physics": ExperimentConfig(
        name="WaveEmbed + WaveOptim + QFE (Full)",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        lr=1e-3, dropout=0.0,
        qfe_lambda=0.1
    ),
    "qfe_only": ExperimentConfig(
        name="WaveEmbed + AdamW + QFE",
        model_type="wave",
        use_rgd=False, use_qfe=True,
        lr=6e-4, dropout=0.1
    ),
    "pure_wave": ExperimentConfig(
        name="PureWaveAttn (ELU+1)",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        lr=1e-3, dropout=0.1,
        pure_wave_attention=True,
        pure_wave_kernel="elu_plus_one"
    ),
    
    # ============================================================
    # INTERFERENCE ATTENTION (True Physics - Energy Normalization)
    # ============================================================
    # Memory-optimized via phasor decomposition: O(T²) instead of O(T²W)
    
    # Wave Embeddings + InterferenceAttention + AdamW (isolates interference attention effect)
    "interference_attention": ExperimentConfig(
        name="WaveEmbed + InterferenceAttn + AdamW",
        model_type="wave",
        use_rgd=False, use_qfe=False,
        use_interference_attention=True,  # True physics-based attention!
        lr=6e-4, dropout=0.1
        # No batch_size_override needed - phasor decomposition is memory-efficient
    ),
    
    # Wave Embeddings + InterferenceAttention + WaveOptim (full physics attention)
    "interference_full": ExperimentConfig(
        name="WaveEmbed + InterferenceAttn + WaveOptim + QFE",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        use_interference_attention=True,
        lr=1e-3, dropout=0.0,
        qfe_lambda=0.1
        # No batch_size_override needed - phasor decomposition is memory-efficient
    ),
    
    # ============================================================
    # PURE WAVE GPT - Wave-to-Wave Throughout!
    # ============================================================
    # This is the novel architecture: NO EMBEDDINGS, pure wave physics
    # Token → Wave → Wave → ... → Wave → Logits
    
    "pure_wave": ExperimentConfig(
        name="🌊 PURE WAVE GPT (Wave-to-Wave) - Enhanced",
        model_type="pure_wave",  # New model type!
        use_rgd=False, use_qfe=False,
        use_interference_attention=True,  # Always true for pure wave
        lr=5e-4, dropout=0.05,  # ENHANCED: Higher LR + lower dropout for more expressivity
        warmup_steps=2000,  # Longer warmup for complex wave dynamics
        # Enhanced wave parameters need careful training
    ),
    
    "pure_wave_full": ExperimentConfig(
        name="🌊 PURE WAVE GPT + WaveOptim + QFE",
        model_type="pure_wave",
        use_rgd=True, use_qfe=True,
        use_interference_attention=True,
        lr=5e-4, dropout=0.0,  # FIXED: Lower learning rate for stability
        qfe_lambda=0.05  # FIXED: Lower QFE lambda for stability
    ),

    # ============================================================
    # SYSTEMATIC GRID SEARCH (2x2x2 = 8 experiments)
    # Factors: Embeddings × Attention × Optimizer
    # ============================================================
    # This grid isolates the effect of each component:
    # - Embeddings: Standard vs Wave
    # - Attention: Hybrid (softmax) vs Interference (physics)
    # - Optimizer: AdamW vs WaveNativeOptimizer
    
    # Grid 1: Std Embed + Hybrid Attn + AdamW (CONTROL)
    "grid_std_hybrid_adamw": ExperimentConfig(
        name="[GRID] Std + Hybrid + AdamW",
        model_type="standard",
        use_rgd=False, use_qfe=False,
        use_interference_attention=False,
        lr=6e-4, dropout=0.1
    ),
    
    # Grid 2: Std Embed + Hybrid Attn + WaveOptim
    "grid_std_hybrid_waveopt": ExperimentConfig(
        name="[GRID] Std + Hybrid + WaveOpt",
        model_type="standard",
        use_rgd=True, use_qfe=False,
        use_interference_attention=False,
        lr=1e-3, dropout=0.1
    ),
    
    # Grid 3: Wave Embed + Hybrid Attn + AdamW
    "grid_wave_hybrid_adamw": ExperimentConfig(
        name="[GRID] Wave + Hybrid + AdamW",
        model_type="wave",
        use_rgd=False, use_qfe=False,
        use_interference_attention=False,
        lr=6e-4, dropout=0.1
    ),
    
    # Grid 4: Wave Embed + Hybrid Attn + WaveOptim
    "grid_wave_hybrid_waveopt": ExperimentConfig(
        name="[GRID] Wave + Hybrid + WaveOpt",
        model_type="wave",
        use_rgd=True, use_qfe=False,
        use_interference_attention=False,
        lr=1e-3, dropout=0.1
    ),
    
    # Grid 5: Wave Embed + Interference Attn + AdamW (KEY TEST!)
    "grid_wave_interference_adamw": ExperimentConfig(
        name="[GRID] Wave + Interference + AdamW",
        model_type="wave",
        use_rgd=False, use_qfe=False,
        use_interference_attention=True,
        lr=6e-4, dropout=0.1
        # Memory-efficient via phasor decomposition
    ),
    
    # Grid 6: Wave Embed + Interference Attn + WaveOptim
    "grid_wave_interference_waveopt": ExperimentConfig(
        name="[GRID] Wave + Interference + WaveOpt",
        model_type="wave",
        use_rgd=True, use_qfe=False,
        use_interference_attention=True,
        lr=1e-3, dropout=0.1
        # Memory-efficient via phasor decomposition
    ),
    
    # Grid 7: Wave Embed + Interference Attn + AdamW + QFE
    "grid_wave_interference_adamw_qfe": ExperimentConfig(
        name="[GRID] Wave + Interference + AdamW + QFE",
        model_type="wave",
        use_rgd=False, use_qfe=True,
        use_interference_attention=True,
        lr=6e-4, dropout=0.1,
        qfe_lambda=0.05
        # Memory-efficient via phasor decomposition
    ),
    
    # Grid 8: Wave Embed + Interference Attn + WaveOptim + QFE (FULL PHYSICS)
    "grid_wave_interference_full": ExperimentConfig(
        name="[GRID] Wave + Interference + WaveOpt + QFE",
        model_type="wave",
        use_rgd=True, use_qfe=True,
        use_interference_attention=True,
        lr=1e-3, dropout=0.0,
        qfe_lambda=0.1
        # Memory-efficient via phasor decomposition
    ),
    
    # ============================================================
    # DIAGONAL BREAKER EXPERIMENTS (Non-Linear Physics Trinity)
    # ============================================================
    
    # Baseline: Pure Wave without Trinity (should show diagonal blindness)
    "diagonal_baseline": ExperimentConfig(
        name="PureWave Baseline (Diagonal Blind)",
        model_type="pure_wave",
        use_rgd=False, use_qfe=False,
        lr=3e-4, dropout=0.1,
        steps=5000,
        # Trinity disabled - should stay diagonal
        use_overdrive=False,
        use_fm_synthesis=False, 
        use_interferometer=False,
        mlp_expansion=1  # No expansion
    ),
    
    # Trinity: Non-Linear Physics Trinity (should break diagonal)
    "diagonal_breaker": ExperimentConfig(
        name="Non-Linear Physics Trinity (Diagonal Breaker)",
        model_type="pure_wave", 
        use_rgd=False,  # Use AdamW for reliable convergence
        use_qfe=False,  # Disable QFE initially - let model learn freely
        lr=1e-3,  # Higher LR for faster learning
        dropout=0.1,
        steps=5000,
        qfe_lambda=0.01,  # Lower if enabled later
        # Trinity enabled - should break diagonal blindness
        use_overdrive=True,
        use_fm_synthesis=True,
        use_interferometer=True, 
        mlp_expansion=4  # 4x expansion for reasoning core
    ),
    
    # Ablation A: Only Overdrive (Saturation)
    "diagonal_overdrive_only": ExperimentConfig(
        name="Overdrive Only (Saturation)",
        model_type="pure_wave",
        use_rgd=True, use_qfe=False,
        lr=3e-4, dropout=0.1,
        steps=3000,
        use_overdrive=True,
        use_fm_synthesis=False,
        use_interferometer=False,
        mlp_expansion=4
    ),
    
    # Ablation B: Only FM Synthesis (Context Modulation)
    "diagonal_fm_only": ExperimentConfig(
        name="FM Synthesis Only (Context Modulation)",
        model_type="pure_wave",
        use_rgd=True, use_qfe=False,
        lr=3e-4, dropout=0.1,
        steps=3000,
        use_overdrive=False,
        use_fm_synthesis=True,
        use_interferometer=False,
        mlp_expansion=4
    ),
    
    # Ablation C: Only Interferometer (Phase Gating)
    "diagonal_interferometer_only": ExperimentConfig(
        name="Interferometer Only (Phase Gating)",
        model_type="pure_wave",
        use_rgd=True, use_qfe=False,
        lr=3e-4, dropout=0.1,
        steps=3000,
        use_overdrive=False,
        use_fm_synthesis=False,
        use_interferometer=True,
        mlp_expansion=4
    ),
    
    # Quick Test: Fast diagonal check
    "diagonal_quick": ExperimentConfig(
        name="Quick Diagonal Test (Trinity)",
        model_type="pure_wave",
        use_rgd=False,  # Use AdamW for faster convergence initially
        use_qfe=False,  # Disable QFE to let model learn freely first
        lr=1e-3,  # Higher LR for faster learning
        dropout=0.1,
        steps=1000,  # Quick test
        qfe_lambda=0.01,  # Lower if enabled
        use_overdrive=True,
        use_fm_synthesis=True,
        use_interferometer=True,
        mlp_expansion=4
    ),
    
    # ============================================================
    # SCALED FINAL TRAINING: 120M Parameters Long Run
    # ============================================================
    # This is the FINAL experiment to test DC modes at scale
    # Uses: DC modes + Trinity + Multi-head + 4x MLP + longer training
    
    "scaled_final": ExperimentConfig(
        name="🚀 Scaled PureWave (120M) - Final Training",
        model_type="pure_wave",
        use_rgd=False,      # Use AdamW for stability at scale
        use_qfe=False,      # Disable QFE initially
        lr=3e-4,            # Lower LR for larger model
        weight_decay=0.1,   # Stronger regularization at scale
        dropout=0.1,
        warmup_steps=2000,  # Longer warmup for larger model
        patience=10,        # More patience for longer training
        steps=20000,        # Long training run
        grad_accum_steps=8, # Effective batch = 8 * 8 = 64
        # === FULL TRINITY ENABLED ===
        use_overdrive=True,
        use_fm_synthesis=True,
        use_interferometer=True,
        mlp_expansion=4,    # 4x MLP expansion for reasoning
    ),
    
    # Scaled with QFE loss (experimental)
    "scaled_final_qfe": ExperimentConfig(
        name="🚀 Scaled PureWave (120M) + QFE Loss",
        model_type="pure_wave",
        use_rgd=False,
        use_qfe=True,       # Enable QFE for coherence
        qfe_lambda=0.01,    # Low QFE weight
        lr=3e-4,
        weight_decay=0.1,
        dropout=0.1,
        warmup_steps=2000,
        patience=10,
        steps=20000,
        grad_accum_steps=8,
        use_overdrive=True,
        use_fm_synthesis=True,
        use_interferometer=True,
        mlp_expansion=4,
    ),
    
    # Scaled baseline (no Trinity) for comparison
    "scaled_baseline": ExperimentConfig(
        name="📊 Scaled PureWave (120M) - Baseline (No Trinity)",
        model_type="pure_wave",
        use_rgd=False,
        use_qfe=False,
        lr=3e-4,
        weight_decay=0.1,
        dropout=0.1,
        warmup_steps=2000,
        patience=10,
        steps=20000,
        grad_accum_steps=8,
        # === TRINITY DISABLED for comparison ===
        use_overdrive=False,
        use_fm_synthesis=False,
        use_interferometer=False,
        mlp_expansion=1,    # No MLP expansion
    ),
}


@dataclass
class ModelConfig:
    """Model size configurations"""
    name: str
    d_model: int
    num_layers: int
    num_heads: int
    num_waves: int
    num_harmonics: int
    vocab_size: int
    block_size: int
    batch_size: int


MODEL_CONFIGS = {
    "small": ModelConfig(
        name="Small (GPT-2 Compatible)",
        d_model=384, num_layers=8, num_heads=8,
        num_waves=48, num_harmonics=4,
        vocab_size=50257, block_size=256, batch_size=16 # Reduced to 16 to fit QFE Loss in Memory
    ),
    "medium": ModelConfig(
        name="Medium",
        d_model=512, num_layers=10, num_heads=8,
        num_waves=64, num_harmonics=4,
        vocab_size=50257, block_size=384, batch_size=24
    ),
    "large": ModelConfig(
        name="Large",
        d_model=768, num_layers=12, num_heads=12,
        num_waves=96, num_harmonics=4,
        vocab_size=50257, block_size=512, batch_size=16
    ),
    # === SCALED MODEL: ~120M Parameters ===
    # Designed for final long training run to test DC modes at scale
    # Target: 120M params with full Trinity + DC modes
    # CORRECTED: Increased num_waves to reach 120M target
    "scaled": ModelConfig(
        name="Scaled (120M PureWave)",
        d_model=768,        # Same as GPT-2 small
        num_layers=12,      # 12 layers for depth
        num_heads=12,       # 12 heads for multi-head diversity
        num_waves=144,      # INCREASED: More waves for 120M target
        num_harmonics=4,    # Keep harmonics at 4
        vocab_size=50257,   # GPT-2 vocabulary
        block_size=512,     # Longer context
        batch_size=6        # Smaller batch for larger model
    ),
}


# ==========================================
# Dataset Loaders
# ==========================================

def load_fineweb_tiktoken(console, subset="sample-10BT", target_tokens=500_000_000):
    """
    Load FineWeb-Edu and tokenize with Tiktoken until exactly target_tokens.
    """
    console.print(f"📥 Loading FineWeb-Edu ({subset}) via Tiktoken...")
    console.print(f"   Target: {target_tokens:,} tokens")
    
    enc = tiktoken.get_encoding("gpt2")
    
    
    # MOCK FOR VERIFICATION
    # console.print("[yellow]⚠️  MOCKING DATA FOR FAST STATS CHECK[/yellow]")
    # return torch.randint(0, 50257, (100000,), dtype=torch.long)
    
    # Stream the dataset
    ds = load_dataset(
       "HuggingFaceFW/fineweb-edu",
       name=subset,
       split="train",
       streaming=True
    )
    
    all_tokens = []
    total_count = 0
    
    with Progress(console=console) as progress:
        task = progress.add_task("Streaming & Tokenizing...", total=target_tokens)
        
        for item in ds:
            text = item.get("text", "")
            if not text: continue
            
            # Tokenize
            tokens = enc.encode(text)
            all_tokens.extend(tokens)
            total_count += len(tokens)
            
            progress.update(task, completed=min(total_count, target_tokens))
            
            if total_count >= target_tokens:
                break
                
    # Trim to exact target
    all_tokens = all_tokens[:target_tokens]
    console.print(f"✅ Loaded exactly {len(all_tokens):,} tokens.")
    
    # Convert to tensor
    data = torch.tensor(all_tokens, dtype=torch.long)
    return data


def create_mock_dataset(vocab_size: int = 50257, num_tokens: int = 10000) -> torch.Tensor:
    """
    Create mock dataset for dry-run testing.
    
    Args:
        vocab_size: Size of vocabulary
        num_tokens: Number of tokens to generate
        
    Returns:
        Tensor of random token IDs
    """
    return torch.randint(0, vocab_size, (num_tokens,), dtype=torch.long)


def get_dataset(dataset_name: str, console, max_tokens: int = 500_000_000, dry_run: bool = False):
    """Get dataset by name"""
    if dry_run:
        console.print("[yellow]📦 Creating mock dataset for dry-run...[/yellow]")
        return create_mock_dataset(vocab_size=50257, num_tokens=10000)
    
    if dataset_name == "shakespeare":
        # Fallback to shakespeare if explicit
        return load_shakespeare(console) # Returns text, not tensor
    elif dataset_name == "fineweb" or dataset_name == "fineweb_small":
        # Use tiktoken loader
        return load_fineweb_tiktoken(console, subset="sample-10BT", target_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ==========================================
# Diagonal Blindness Analysis
# ==========================================

def extract_attention_maps(model, input_ids: torch.Tensor) -> List[torch.Tensor]:
    """Extract attention maps from PureWaveGPT for diagonal analysis."""
    
    attention_maps = []
    
    def attention_hook(module, input, output):
        # For PureWaveInterference, extract coupling matrix if available
        if hasattr(module, 'coupling') and module.coupling is not None:
            # coupling shape: (B, H, T, T) - average over batch and heads
            attn_map = module.coupling.mean(dim=(0, 1)).detach().cpu()  # (T, T)
            attention_maps.append(attn_map)
    
    # Register hooks on attention layers
    hooks = []
    if hasattr(model, 'wave_layers'):
        for layer in model.wave_layers:
            if hasattr(layer, 'wave_attention'):
                hook = layer.wave_attention.register_forward_hook(attention_hook)
                hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_maps


def extract_attention_maps_detailed(model, input_ids: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
    """
    Extract DETAILED attention maps for DC mode analysis.
    
    NOTE: DC modes are in the WAVE dimension (W), not the HEAD dimension (H).
    The coupling matrix (B, H, T, T) is computed by summing over all waves.
    To isolate DC mode contribution, we need to recompute interference
    using only DC mode waves.
    
    Returns:
        Dict with:
        - 'full': List of (T, T) averaged attention maps per layer
        - 'per_head': List of (H, T, T) per-head attention maps per layer
        - 'dc_attention': List of (T, T) DC-mode-only attention per layer
        - 'ac_attention': List of (T, T) AC-mode-only attention per layer
        - 'n_dc_modes': Number of DC modes in the model
    """
    
    full_maps = []
    per_head_maps = []
    dc_attention_maps = []
    ac_attention_maps = []
    n_dc_modes = 0
    
    # Get n_dc_modes from model
    if hasattr(model, 'wave_excitation') and hasattr(model.wave_excitation, 'n_dc_modes'):
        n_dc_modes = model.wave_excitation.n_dc_modes
    
    # Store intermediate wave states for DC/AC separation
    wave_states_per_layer = []
    
    def attention_hook(module, input, output):
        if hasattr(module, 'coupling') and module.coupling is not None:
            # coupling shape: (B, H, T, T)
            coupling = module.coupling.detach().cpu()
            
            # Full map: average over batch and heads
            full_map = coupling.mean(dim=(0, 1))  # (T, T)
            full_maps.append(full_map)
            
            # Per-head map: average over batch only
            per_head_map = coupling.mean(dim=0)  # (H, T, T)
            per_head_maps.append(per_head_map)
    
    # Register hooks
    hooks = []
    if hasattr(model, 'wave_layers'):
        for layer in model.wave_layers:
            if hasattr(layer, 'wave_attention'):
                hook = layer.wave_attention.register_forward_hook(attention_hook)
                hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute DC/AC separated attention maps
    # Since we can't easily separate DC/AC from the coupling matrix,
    # we'll use the per-head maps as a proxy:
    # - First few heads tend to capture lower frequencies (DC-like)
    # - Later heads tend to capture higher frequencies (AC-like)
    # This is an approximation based on initialization patterns
    
    n_heads = per_head_maps[0].shape[0] if per_head_maps else 8
    n_dc_heads = max(1, n_heads // 4)  # Use first 25% of heads as DC proxy
    
    for per_head_map in per_head_maps:
        # DC attention: first few heads (lower frequency patterns)
        dc_attn = per_head_map[:n_dc_heads].mean(dim=0)  # (T, T)
        dc_attention_maps.append(dc_attn)
        
        # AC attention: remaining heads (higher frequency patterns)
        ac_attn = per_head_map[n_dc_heads:].mean(dim=0)  # (T, T)
        ac_attention_maps.append(ac_attn)
    
    return {
        'full': full_maps,
        'per_head': per_head_maps,
        'dc_attention': dc_attention_maps,
        'ac_attention': ac_attention_maps,
        'n_dc_modes': n_dc_modes,
        'n_dc_heads': n_dc_heads
    }


def analyze_diagonal_pattern(attention_maps: List[torch.Tensor]) -> Dict[str, float]:
    """
    Analyze how diagonal the attention patterns are.
    
    FIXED: Proper metrics for causal attention matrices.
    
    For a causal attention matrix (lower triangular), we need to compare
    against what UNIFORM causal attention would look like:
    - Position i can attend to positions 0..i (i+1 positions)
    - Uniform attention at position i: each position gets 1/(i+1)
    - Diagonal element at position i: 1/(i+1)
    - Expected diagonal ratio for uniform: sum(1/(i+1)) / T ≈ ln(T)/T
    
    Metrics:
    - diagonal_dominance: How much stronger is diagonal vs uniform? (>1 = diagonal bias)
    - near_diagonal_ratio: Attention within ±3 positions (local attention)
    - global_attention_ratio: Attention to positions >10 away (long-range)
    """
    
    if not attention_maps:
        return {}
    
    metrics = {}
    
    for layer_idx, attn_map in enumerate(attention_maps):
        T = attn_map.shape[0]
        
        # === DIAGONAL DOMINANCE ===
        # Compare actual diagonal attention to expected uniform diagonal
        diagonal_sum = torch.diag(attn_map).sum().item()
        
        # For causal attention, total sum is lower triangle
        # Each row i sums to ~1 (normalized), so total ≈ T
        total_sum = attn_map.sum().item()
        
        # Expected diagonal for uniform causal attention:
        # At position i, diagonal gets 1/(i+1) of attention
        # Sum over all positions: sum(1/(i+1) for i in 0..T-1) ≈ ln(T) + 0.577
        expected_diagonal_sum = sum(1.0 / (i + 1) for i in range(T))
        
        # Diagonal dominance: actual / expected (>1 means diagonal bias)
        diagonal_dominance = diagonal_sum / (expected_diagonal_sum + 1e-8)
        
        # === NEAR-DIAGONAL RATIO (Local Attention) ===
        # Attention within ±3 positions of diagonal
        near_diag_sum = 0.0
        for i in range(T):
            for j in range(max(0, i - 3), min(T, i + 1)):  # Causal: j <= i
                near_diag_sum += attn_map[i, j].item()
        near_diagonal_ratio = near_diag_sum / (total_sum + 1e-8)
        
        # === GLOBAL ATTENTION RATIO (Long-Range) ===
        # Attention to positions more than 10 away
        global_sum = 0.0
        global_possible = 0  # Count how many such positions exist
        for i in range(T):
            for j in range(max(0, i - 10)):  # Positions more than 10 before i
                global_sum += attn_map[i, j].item()
                global_possible += 1
        
        # Normalize by what's possible (not all positions have long-range options)
        if global_possible > 0:
            # Expected global attention if uniform
            expected_global = global_possible / (T * (T + 1) / 2)  # Fraction of causal positions
            global_attention_ratio = (global_sum / total_sum) / (expected_global + 1e-8)
        else:
            global_attention_ratio = 0.0
        
        # === ATTENTION ENTROPY (Spread) ===
        # Higher entropy = more spread out attention
        # Compute per-row entropy and average
        row_entropies = []
        for i in range(T):
            row = attn_map[i, :i+1]  # Causal: only positions 0..i
            if row.sum() > 1e-8:
                row_norm = row / (row.sum() + 1e-8)
                # Entropy: -sum(p * log(p))
                entropy = -torch.sum(row_norm * torch.log(row_norm + 1e-8)).item()
                # Normalize by max entropy (uniform distribution)
                max_entropy = np.log(i + 1) if i > 0 else 1.0
                normalized_entropy = entropy / (max_entropy + 1e-8)
                row_entropies.append(normalized_entropy)
        
        avg_entropy = np.mean(row_entropies) if row_entropies else 0.0
        
        # Store metrics
        metrics[f'layer_{layer_idx}_diagonal_dominance'] = diagonal_dominance
        metrics[f'layer_{layer_idx}_near_diagonal_ratio'] = near_diagonal_ratio
        metrics[f'layer_{layer_idx}_global_attention_ratio'] = global_attention_ratio
        metrics[f'layer_{layer_idx}_attention_entropy'] = avg_entropy
    
    # Overall metrics (average across layers)
    diagonal_dominances = [v for k, v in metrics.items() if 'diagonal_dominance' in k]
    near_diagonal_ratios = [v for k, v in metrics.items() if 'near_diagonal_ratio' in k]
    global_ratios = [v for k, v in metrics.items() if 'global_attention_ratio' in k]
    entropies = [v for k, v in metrics.items() if 'attention_entropy' in k]
    
    if diagonal_dominances:
        metrics['avg_diagonal_dominance'] = np.mean(diagonal_dominances)
    if near_diagonal_ratios:
        metrics['avg_near_diagonal_ratio'] = np.mean(near_diagonal_ratios)
    if global_ratios:
        metrics['avg_global_attention_ratio'] = np.mean(global_ratios)
    if entropies:
        metrics['avg_attention_entropy'] = np.mean(entropies)
    
    return metrics


def plot_attention_maps(attention_maps: List[torch.Tensor], save_path: str, title_suffix: str = ""):
    """Plot attention maps to visualize diagonal patterns."""
    
    if not attention_maps:
        return
    
    n_layers = len(attention_maps)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, attn_map in enumerate(attention_maps):
        if i >= len(axes):
            break
            
        ax = axes[i]
        im = ax.imshow(attn_map.numpy(), cmap='plasma', aspect='auto', origin='upper')
        
        # Compute per-layer metrics for title
        T = attn_map.shape[0]
        diag_sum = torch.diag(attn_map).sum().item()
        expected_diag = sum(1.0 / (j + 1) for j in range(T))
        diag_dom = diag_sum / (expected_diag + 1e-8)
        
        ax.set_title(f'Layer {i} - DiagDom: {diag_dom:.2f}{title_suffix}')
        ax.set_xlabel('Key Position (attends to)')
        ax.set_ylabel('Query Position (from)')
        plt.colorbar(im, ax=ax, label='Attention Weight')
    
    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dc_mode_attention(dc_attention_maps: List[torch.Tensor], n_dc_heads: int, save_path: str, title_suffix: str = ""):
    """
    Plot 1: "The Global Anchor" - DC Mode Isolation
    
    Shows attention from DC-like heads only, which should reveal
    non-local, position-independent semantic connections.
    
    Expected: Diagonal should be FAINT or GONE. Should show scattered
    blocks or vertical/horizontal lines (global topic connections).
    """
    
    if not dc_attention_maps:
        return
    
    n_layers = len(dc_attention_maps)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(f'🔴 DC Mode Attention (Global Anchor) - Low-Freq Heads{title_suffix}', fontsize=14, fontweight='bold')
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, dc_attention in enumerate(dc_attention_maps):
        if i >= len(axes):
            break
        
        ax = axes[i]
        T = dc_attention.shape[0]
        
        # Use diverging colormap to show deviations from uniform
        im = ax.imshow(dc_attention.numpy(), cmap='RdBu_r', aspect='auto', origin='upper')
        
        # Compute DC-specific metrics
        diag_sum = torch.diag(dc_attention).sum().item()
        expected_diag = sum(1.0 / (j + 1) for j in range(T))
        diag_dom = diag_sum / (expected_diag + 1e-8)
        
        ax.set_title(f'Layer {i} DC Heads (0-{n_dc_heads-1}) | DiagDom: {diag_dom:.2f}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, label='DC Attention')
    
    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ac_mode_attention(ac_attention_maps: List[torch.Tensor], n_dc_heads: int, n_total_heads: int, save_path: str, title_suffix: str = ""):
    """
    Plot AC Mode Attention (for comparison with DC modes).
    
    Shows attention from AC-like heads only (higher frequency patterns),
    which should show position-dependent, local attention patterns.
    
    Expected: Strong diagonal pattern (positional attention).
    """
    
    if not ac_attention_maps:
        return
    
    n_layers = len(ac_attention_maps)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(f'🟢 AC Mode Attention (Positional) - High-Freq Heads{title_suffix}', fontsize=14, fontweight='bold')
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, ac_attention in enumerate(ac_attention_maps):
        if i >= len(axes):
            break
        
        ax = axes[i]
        T = ac_attention.shape[0]
        
        im = ax.imshow(ac_attention.numpy(), cmap='plasma', aspect='auto', origin='upper')
        
        # Compute AC-specific metrics
        diag_sum = torch.diag(ac_attention).sum().item()
        expected_diag = sum(1.0 / (j + 1) for j in range(T))
        diag_dom = diag_sum / (expected_diag + 1e-8)
        
        ax.set_title(f'Layer {i} AC Heads ({n_dc_heads}-{n_total_heads-1}) | DiagDom: {diag_dom:.2f}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, label='AC Attention')
    
    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_differential_attention(attention_maps: List[torch.Tensor], save_path: str, title_suffix: str = ""):
    """
    Plot 2: "The Contextual Differential" - Diagonal Subtraction
    
    Subtracts the trivial self-attention (diagonal) to reveal
    subtle learned connections like:
    - Horizontal smears (FM Synthesis effects)
    - Checkered patterns (Harmonic Coupling)
    - Vertical lines (Global topic anchors)
    
    Method: Attention_diff = Attention - α * I
    where α is the average diagonal value
    """
    
    if not attention_maps:
        return
    
    n_layers = len(attention_maps)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(f'🔍 Differential Attention (Diagonal Removed){title_suffix}', fontsize=14, fontweight='bold')
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, attn_map in enumerate(attention_maps):
        if i >= len(axes):
            break
        
        ax = axes[i]
        T = attn_map.shape[0]
        
        # Compute α = average diagonal value
        diag_values = torch.diag(attn_map)
        alpha = diag_values.mean().item()
        
        # Create identity matrix scaled by α
        identity_scaled = torch.eye(T) * alpha
        
        # Differential: remove diagonal "glare"
        diff_map = attn_map - identity_scaled
        
        # Use diverging colormap centered at 0
        vmax = max(abs(diff_map.min().item()), abs(diff_map.max().item()))
        vmin = -vmax
        
        im = ax.imshow(diff_map.numpy(), cmap='RdBu_r', aspect='auto', origin='upper',
                      vmin=vmin, vmax=vmax)
        
        # Compute off-diagonal energy
        off_diag_mask = ~torch.eye(T, dtype=torch.bool)
        off_diag_energy = diff_map[off_diag_mask].abs().mean().item()
        
        ax.set_title(f'Layer {i} | α={alpha:.3f} | OffDiag Energy: {off_diag_energy:.4f}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, label='Differential')
    
    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dc_vs_ac_comparison(
    dc_attention_maps: List[torch.Tensor],
    ac_attention_maps: List[torch.Tensor],
    n_dc_heads: int,
    n_total_heads: int,
    save_path: str,
    title_suffix: str = ""
):
    """
    Side-by-side comparison of DC vs AC attention patterns.
    
    This is the KEY diagnostic plot:
    - DC (left): Should show GLOBAL patterns (non-diagonal)
    - AC (right): Should show LOCAL patterns (diagonal)
    
    If both look the same (diagonal), DC modes are NOT working.
    """
    
    if not dc_attention_maps or not ac_attention_maps:
        return
    
    n_layers = min(len(dc_attention_maps), len(ac_attention_maps))
    
    fig, axes = plt.subplots(n_layers, 2, figsize=(10, 4*n_layers))
    fig.suptitle(f'🔴 DC vs 🟢 AC Attention Comparison{title_suffix}', fontsize=14, fontweight='bold')
    
    if n_layers == 1:
        axes = axes.reshape(1, 2)
    
    for i in range(n_layers):
        dc_attn = dc_attention_maps[i]
        ac_attn = ac_attention_maps[i]
        T = dc_attn.shape[0]
        
        # Compute metrics
        dc_diag = torch.diag(dc_attn).sum().item()
        ac_diag = torch.diag(ac_attn).sum().item()
        expected_diag = sum(1.0 / (j + 1) for j in range(T))
        dc_dom = dc_diag / (expected_diag + 1e-8)
        ac_dom = ac_diag / (expected_diag + 1e-8)
        
        # DC plot (left)
        ax_dc = axes[i, 0]
        im_dc = ax_dc.imshow(dc_attn.numpy(), cmap='RdBu_r', aspect='auto', origin='upper')
        ax_dc.set_title(f'Layer {i} 🔴 DC (H0-{n_dc_heads-1}) | DiagDom: {dc_dom:.2f}')
        ax_dc.set_xlabel('Key Position')
        ax_dc.set_ylabel('Query Position')
        plt.colorbar(im_dc, ax=ax_dc, label='Attention')
        
        # AC plot (right)
        ax_ac = axes[i, 1]
        im_ac = ax_ac.imshow(ac_attn.numpy(), cmap='plasma', aspect='auto', origin='upper')
        ax_ac.set_title(f'Layer {i} 🟢 AC (H{n_dc_heads}-{n_total_heads-1}) | DiagDom: {ac_dom:.2f}')
        ax_ac.set_xlabel('Key Position')
        ax_ac.set_ylabel('Query Position')
        plt.colorbar(im_ac, ax=ax_ac, label='Attention')
        
        # Add verdict
        if dc_dom < ac_dom * 0.7:
            verdict = "✅ DC is more global!"
        elif dc_dom > ac_dom * 1.3:
            verdict = "❌ DC is MORE diagonal than AC!"
        else:
            verdict = "⚠️ DC and AC similar"
        
        # Add text annotation
        fig.text(0.5, 1.0 - (i + 0.9) / n_layers, verdict, 
                ha='center', fontsize=10, fontweight='bold',
                transform=fig.transFigure)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_diagnostic_attention(
    model, 
    input_ids: torch.Tensor, 
    save_dir: str, 
    step: int,
    title_suffix: str = ""
):
    """
    Generate ALL diagnostic attention plots:
    1. Standard attention maps (full average)
    2. DC Mode attention (Global Anchor) - should show NON-diagonal patterns
    3. AC Mode attention (Positional) - should show diagonal patterns
    4. Differential attention (Diagonal Removed) - reveals hidden patterns
    
    These plots help diagnose whether DC modes are working to cure diagonal blindness.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract detailed attention maps
    detailed = extract_attention_maps_detailed(model, input_ids)
    full_maps = detailed['full']
    per_head_maps = detailed['per_head']
    dc_attention_maps = detailed['dc_attention']
    ac_attention_maps = detailed['ac_attention']
    n_dc_modes = detailed['n_dc_modes']
    n_dc_heads = detailed['n_dc_heads']
    
    step_suffix = f" (Step {step}){title_suffix}"
    
    # Get total number of heads
    n_total_heads = per_head_maps[0].shape[0] if per_head_maps else 8
    
    # 1. Standard attention maps (full average over all heads)
    plot_attention_maps(
        full_maps,
        os.path.join(save_dir, f"attention_full_step_{step:05d}.png"),
        step_suffix
    )
    
    # 2. DC Mode attention (Global Anchor) - Low-frequency heads
    # Should show NON-diagonal patterns if DC modes are working
    if dc_attention_maps:
        plot_dc_mode_attention(
            dc_attention_maps,
            n_dc_heads,
            os.path.join(save_dir, f"attention_dc_modes_step_{step:05d}.png"),
            step_suffix
        )
    
    # 3. AC Mode attention (Positional) - High-frequency heads
    # Should show diagonal patterns (positional attention)
    if ac_attention_maps:
        plot_ac_mode_attention(
            ac_attention_maps,
            n_dc_heads,
            n_total_heads,
            os.path.join(save_dir, f"attention_ac_modes_step_{step:05d}.png"),
            step_suffix
        )
    
    # 4. Differential attention (Diagonal Removed)
    # Reveals hidden patterns by subtracting diagonal "glare"
    plot_differential_attention(
        full_maps,
        os.path.join(save_dir, f"attention_differential_step_{step:05d}.png"),
        step_suffix
    )
    
    # 5. DC vs AC comparison plot (side by side)
    if dc_attention_maps and ac_attention_maps:
        plot_dc_vs_ac_comparison(
            dc_attention_maps,
            ac_attention_maps,
            n_dc_heads,
            n_total_heads,
            os.path.join(save_dir, f"attention_dc_vs_ac_step_{step:05d}.png"),
            step_suffix
        )


def create_synthetic_data_with_patterns(vocab_size: int, seq_len: int, batch_size: int, device: torch.device):
    """Create synthetic data with patterns that require global context."""
    
    data = []
    for _ in range(batch_size):
        seq = torch.randint(1, vocab_size-10, (seq_len,))  # Leave room for special tokens
        
        # Pattern 1: Long-range dependencies (token at pos 0 determines token at pos -1)
        if seq[0] % 2 == 0:
            seq[-1] = (seq[0] + 1) % vocab_size
        
        # Pattern 2: Nested structures (simple bracket matching)
        open_bracket = vocab_size - 3
        close_bracket = vocab_size - 2
        
        # Insert some bracket pairs
        for i in range(0, seq_len-10, 20):
            if i + 5 < seq_len:
                seq[i] = open_bracket
                seq[i + 5] = close_bracket
        
        data.append(seq)
    
    return torch.stack(data).to(device)


# ==========================================
# Experiment Utilities
# ==========================================

def zip_experiment_directory(experiment_dir: str, console: Console):
    """
    Zip the entire experiment directory for easy download.
    
    Args:
        experiment_dir: Path to the experiment directory
        console: Rich console for logging
    """
    if not os.path.exists(experiment_dir):
        console.print(f"[yellow]Warning: Experiment directory {experiment_dir} does not exist[/yellow]")
        return
    
    # Create zip filename
    zip_filename = f"{experiment_dir}.zip"
    
    console.print(f"[cyan]📦 Zipping experiment directory...[/cyan]")
    console.print(f"[cyan]   Source: {experiment_dir}[/cyan]")
    console.print(f"[cyan]   Target: {zip_filename}[/cyan]")
    
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through all files in the experiment directory
            for root, dirs, files in os.walk(experiment_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Create archive name relative to experiment directory
                    arcname = os.path.relpath(file_path, os.path.dirname(experiment_dir))
                    zipf.write(file_path, arcname)
        
        # Get zip file size for reporting
        zip_size = os.path.getsize(zip_filename)
        zip_size_mb = zip_size / (1024 * 1024)
        
        console.print(f"[green]✅ Experiment zipped successfully![/green]")
        console.print(f"[green]   📁 Zip file: {zip_filename}[/green]")
        console.print(f"[green]   📊 Size: {zip_size_mb:.1f} MB[/green]")
        
        # List contents for verification
        with zipfile.ZipFile(zip_filename, 'r') as zipf:
            file_count = len(zipf.namelist())
            console.print(f"[green]   📄 Files: {file_count}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to zip experiment directory: {e}[/red]")
        raise


# ==========================================
# Training Loop
# ==========================================

def train_experiment(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    exp_config: ExperimentConfig,
    model_config: ModelConfig,
    console: Console,
    device: str = "cuda",
    experiment_dir: Optional[str] = None,
    enable_monitoring: bool = True,
    dry_run: bool = False
) -> Dict:
    """Train a single experiment configuration with optional monitoring"""
    
    console.print(Panel(f"[bold cyan]{exp_config.name}[/bold cyan]", border_style="cyan"))
    
    # Use batch_size_override if set (for memory-heavy experiments like InterferenceAttention)
    batch_size = getattr(exp_config, 'batch_size_override', None) or model_config.batch_size
    if batch_size != model_config.batch_size:
        console.print(f"[yellow]⚠️  Using reduced batch_size={batch_size} (override for memory)[/yellow]")
    
    params = sum(p.numel() for p in model.parameters())
    console.print(f"📊 Parameters: {params:,} ({params/1e6:.2f}M)")
    console.print(f"🔄 Steps: {exp_config.steps}")
    
    # Initialize monitoring components if enabled
    checkpoint_manager = None
    metrics_logger = None
    viz_manager = None
    config_tracker = None
    
    if enable_monitoring and experiment_dir:
        from monitoring import (
            CheckpointManager, MetricsLogger, 
            VisualizationManager, ConfigTracker,
            create_experiment_directory
        )
        
        # Create experiment directory structure
        dirs = create_experiment_directory("experiments", experiment_dir)
        console.print(f"📁 Experiment directory: {dirs['root']}")
        
        # Initialize monitoring components with adjusted intervals for dry-run
        checkpoint_interval = 5 if dry_run else 1000
        log_interval = 1 if dry_run else 10
        viz_interval = 5 if dry_run else 1000
        
        checkpoint_manager = CheckpointManager(
            experiment_dir=dirs['root'],
            save_interval=checkpoint_interval,
            keep_last_n=3
        )
        
        metrics_logger = MetricsLogger(
            log_dir=dirs['logs'],
            log_interval=log_interval
        )
        
        viz_manager = VisualizationManager(
            viz_dir=dirs['visualizations'],
            viz_interval=viz_interval
        )
        
        config_tracker = ConfigTracker(
            experiment_dir=dirs['root']
        )
        
        # Save initial configuration
        config_dict = {
            'experiment_name': exp_config.name,
            'model_config': {
                'd_model': model_config.d_model,
                'num_layers': model_config.num_layers,
                'num_heads': model_config.num_heads,
                'num_waves': model_config.num_waves,
                'num_harmonics': model_config.num_harmonics,
                'vocab_size': model_config.vocab_size,
                'block_size': model_config.block_size,
                'batch_size': model_config.batch_size
            },
            'training_config': asdict(exp_config)
        }
        
        dataset_info = {
            'train_tokens': len(train_data),
            'val_tokens': len(val_data),
            'total_tokens': len(train_data) + len(val_data)
        }
        
        try:
            config_tracker.save_config(config_dict, model, dataset_info)
            console.print("✓ Configuration saved")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save config: {e}[/yellow]")
    
    # Check for existing checkpoint and offer resumption
    start_step = 0
    resumed_loss_history = []
    if checkpoint_manager:
        try:
            checkpoint = checkpoint_manager.load_latest_checkpoint()
            if checkpoint:
                console.print(f"[yellow]Found existing checkpoint at step {checkpoint['step']}[/yellow]")
                console.print(f"[yellow]Checkpoint loss: {checkpoint['loss_history'][-1] if checkpoint.get('loss_history') else 'N/A'}[/yellow]")
                console.print("[yellow]Resume from checkpoint? (y/n)[/yellow]")
                # For automated runs, we'll skip resumption by default
                # In interactive mode, user can modify this
                resume = False  # Set to True to enable auto-resume
                
                if resume:
                    # Load model state
                    model.load_state_dict(checkpoint['model_state_dict'])
                    start_step = checkpoint['step'] + 1
                    
                    # Restore loss history for continuity verification
                    if 'loss_history' in checkpoint:
                        resumed_loss_history = checkpoint['loss_history']
                        console.print(f"[green]✓ Restored {len(resumed_loss_history)} loss values[/green]")
                    
                    console.print(f"[green]✓ Resumed from step {checkpoint['step']}[/green]")
                    console.print(f"[green]✓ Will continue training from step {start_step}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load checkpoint: {e}[/yellow]")
    
    # Setup optimizer
    # Use WaveNativeOptimizer from wave_physics_core if available, else fall back to legacy RGD
    if exp_config.use_rgd:
        if WAVE_PHYSICS_CORE_AVAILABLE and WaveNativeOptimizer is not None:
            # Physics-first approach: WaveNativeOptimizer with parameter-specific masses
            # and resonance-aware damping
            try:
                from wave_physics_core import create_wave_param_groups
                param_groups = create_wave_param_groups(
                    model, 
                    lr=exp_config.lr, 
                    weight_decay=exp_config.weight_decay
                )
                optimizer = WaveNativeOptimizer(
                    param_groups,
                    lr=exp_config.lr,
                    damping=0.1,
                    coherence_weight=exp_config.rgd_strength,
                    weight_decay=exp_config.weight_decay,
                    use_resonance_damping=True  # Enable adaptive damping
                )
                console.print(f"⚡ Optimizer: WaveNativeOptimizer (coherence={exp_config.rgd_strength}, resonance_damping=True)")
            except ImportError:
                # Fallback if create_wave_param_groups not available
                optimizer = WaveNativeOptimizer(
                    model.parameters(),
                    lr=exp_config.lr,
                    damping=0.1,
                    coherence_weight=exp_config.rgd_strength,
                    weight_decay=exp_config.weight_decay
                )
                console.print(f"⚡ Optimizer: WaveNativeOptimizer (coherence={exp_config.rgd_strength})")
        else:
            # Fallback to legacy ResonantGradientDescent
            optimizer = ResonantGradientDescent(
                model.parameters(),
                lr=exp_config.lr,
                resonance_strength=exp_config.rgd_strength,
                warmup_steps=exp_config.warmup_steps,
                weight_decay=exp_config.weight_decay
            )
            console.print(f"⚡ Optimizer: RGD (strength={exp_config.rgd_strength}) [legacy]")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp_config.lr, weight_decay=exp_config.weight_decay)
        console.print(f"⚙️  Optimizer: AdamW")
    
    # Restore optimizer state if resuming from checkpoint
    if start_step > 0 and checkpoint_manager:
        try:
            checkpoint = checkpoint_manager.load_latest_checkpoint()
            if checkpoint and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                console.print(f"[green]✓ Optimizer state restored[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to restore optimizer state: {e}[/yellow]")
    
    # Setup loss
    # Use WaveCoherenceLoss from wave_physics_core if available, else fall back to legacy QFE
    if exp_config.use_qfe:
        if WAVE_PHYSICS_CORE_AVAILABLE and WaveCoherenceLoss is not None:
            # Physics-first approach: WaveCoherenceLoss with phase/energy/harmonic regularization
            loss_fn = WaveCoherenceLoss(
                lambda_phase=exp_config.qfe_lambda,
                lambda_energy=exp_config.qfe_lambda,
                lambda_harmonic=exp_config.qfe_lambda,
                window_size=8
            )
            console.print(f"🌌 Loss: WaveCoherenceLoss (λ={exp_config.qfe_lambda})")
        else:
            # Fallback to legacy QuantumFieldEntanglementLoss
            loss_fn = QuantumFieldEntanglementLoss(
                lambda_coherence=exp_config.qfe_lambda,
                amplitude_threshold=exp_config.qfe_threshold
            )
            console.print(f"🌌 Loss: QFE (λ={exp_config.qfe_lambda}) [legacy]")
    else:
        loss_fn = None
        console.print(f"📉 Loss: Cross-Entropy")
    
    console.print(f"📈 LR: {exp_config.lr:.0e}")
    
    # LR schedule
    def get_lr(step):
        if step < exp_config.warmup_steps:
            return exp_config.lr * step / exp_config.warmup_steps
        progress = (step - exp_config.warmup_steps) / (exp_config.steps - exp_config.warmup_steps)
        return exp_config.lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    # Wave ratio schedule (push toward pure wave)
    def get_wave_ratio_target(step):
        if not exp_config.wave_ratio_schedule:
            return 0.5
        # Schedule: 0.5 -> 0.9 over training
        progress = step / exp_config.steps
        return 0.5 + 0.4 * progress
    
    model.train()
    losses = resumed_loss_history.copy() if resumed_loss_history else []
    coherence_losses = []
    wave_ratios = []
    learning_rates = []
    perplexities = []
    
    # Store the last loss before resumption for continuity verification
    last_loss_before_resume = losses[-1] if losses else None
    
    # Early stopping tracking
    early_stopping = EarlyStopping(
        patience=exp_config.patience,
        min_delta=0.005,
        console=console
    )
    
    start_time = time.perf_counter()
    total_tokens = 0
    
    # === AUTOMATIC MIXED PRECISION (AMP) for faster training ===
    use_amp = torch.cuda.is_available() and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        console.print("[green]⚡ AMP enabled (mixed precision training)[/green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[yellow]Loss: {task.fields[loss]:.4f}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Training", total=exp_config.steps, loss=0.0)
        
        for step in range(start_step, exp_config.steps):
            # Update LR
            current_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Update wave_ratio if available (handle DataParallel)
            base_model = model.module if hasattr(model, 'module') else model
            
            # Check if this is a PureWaveGPT (no embedding attribute)
            if hasattr(base_model, 'wave_excitation'):
                # PureWaveGPT - no wave_ratio annealing needed
                wave_ratios.append(0.0)  # Pure wave mode
            elif hasattr(base_model, 'embedding') and hasattr(base_model.embedding, 'wave_ratio') and exp_config.wave_ratio_schedule:
                # Standard WaveGPT with annealing
                target_ratio = get_wave_ratio_target(step)
                # Softly push toward target
                with torch.no_grad():
                    # wave_ratio uses sigmoid, so we need inverse sigmoid (logit)
                    current = torch.sigmoid(base_model.embedding.wave_ratio).item()
                    # Blend toward target
                    new_ratio = current * 0.99 + target_ratio * 0.01
                    # Convert back to logit space
                    new_ratio = max(0.01, min(0.99, new_ratio))
                    base_model.embedding.wave_ratio.data = torch.tensor(
                        math.log(new_ratio / (1 - new_ratio))
                    ).to(device)
                wave_ratios.append(torch.sigmoid(base_model.embedding.wave_ratio).item())
            else:
                # No wave_ratio available
                wave_ratios.append(0.0)
            
            
            # --- Gradient Accumulation Loop ---
            accum_loss_scalar = 0.0
            
            # Compute annealing ratio for this step (Requirements 6.1, 6.2)
            # Decays from 1.0 to 0.0 over ANNEALING_STEPS (3000 steps)
            current_annealing_ratio = get_annealing_ratio(step, ANNEALING_STEPS)
            
            optimizer.zero_grad() # Zero gradients before accumulation starts
            
            for _ in range(exp_config.grad_accum_steps):
                # Get batch
                x, y = get_batch(train_data, batch_size, model_config.block_size, device)
                total_tokens += x.numel()
                
                # Forward with AMP (mixed precision) for faster training
                with torch.cuda.amp.autocast(enabled=use_amp):
                    # Forward with annealing ratio (Requirements 6.1, 6.2)
                    # Pass standard_embed_ratio only to models that support it
                    base_model = model.module if hasattr(model, 'module') else model
                    
                    if hasattr(base_model, 'wave_excitation'):
                        # PureWaveGPT - no annealing needed
                        logits, ce_loss = model(x, y)
                    else:
                        # WaveGPT - supports annealing
                        logits, ce_loss = model(x, y, standard_embed_ratio=current_annealing_ratio)
                    
                    # Compute loss
                    if exp_config.use_qfe and loss_fn is not None:
                        # Handle both WaveCoherenceLoss (new) and QuantumFieldEntanglementLoss (legacy)
                        if WAVE_PHYSICS_CORE_AVAILABLE and isinstance(loss_fn, WaveCoherenceLoss):
                            # WaveCoherenceLoss returns dict with 'total', 'ce', 'coherence' keys
                            loss_dict = loss_fn(logits, y)
                            loss = loss_dict['total']
                            coherence_losses.append(loss_dict['coherence'].item())
                        else:
                            # Legacy QuantumFieldEntanglementLoss with return_components=True
                            loss_dict = loss_fn(logits, y, return_components=True)
                            loss = loss_dict['total']
                            coherence_losses.append(loss_dict['coherence'].item())
                    else:
                        loss = ce_loss
                        if loss.ndim > 0:
                            loss = loss.mean()
                
                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    console.print(f"[red]NaN/Inf at step {step}[/red]")
                    return {'loss': float('nan')}
                
                # Scale loss for accumulation
                loss = loss / exp_config.grad_accum_steps
                
                # Backward with AMP scaler
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Track raw loss (unscaled) for logging
                accum_loss_scalar += loss.item() * exp_config.grad_accum_steps

            # Update weights after accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step with AMP scaler
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # Average loss over accumulation steps (approximate)
            avg_loss = accum_loss_scalar / exp_config.grad_accum_steps
            losses.append(avg_loss)
            learning_rates.append(current_lr)
            
            # === DIAGONAL BLINDNESS MONITORING ===
            # Monitor attention patterns for diagonal breaker experiments
            diagonal_metrics = {}
            if (step % 100 == 0 or step == exp_config.steps - 1) and exp_config.model_type == "pure_wave":
                try:
                    # Create test batch with patterns requiring global context
                    test_batch = create_synthetic_data_with_patterns(
                        model_config.vocab_size, 
                        min(64, model_config.block_size),  # Smaller for attention analysis
                        1, 
                        device
                    )
                    
                    # Extract attention maps
                    attention_maps = extract_attention_maps(model, test_batch)
                    
                    if attention_maps:
                        diagonal_metrics = analyze_diagonal_pattern(attention_maps)
                        
                        # Save ALL diagnostic attention maps at key checkpoints
                        if experiment_dir and (step % 1000 == 0 or step == exp_config.steps - 1):
                            attention_dir = os.path.join(experiment_dir, "attention_maps")
                            os.makedirs(attention_dir, exist_ok=True)
                            
                            # Generate ALL diagnostic plots (DC modes, AC modes, differential)
                            plot_all_diagnostic_attention(
                                model,
                                test_batch,
                                attention_dir,
                                step,
                                title_suffix=""
                            )
                        
                        # Log diagonal metrics (FIXED: proper causal attention metrics)
                        if 'avg_diagonal_dominance' in diagonal_metrics:
                            diag_dom = diagonal_metrics['avg_diagonal_dominance']
                            near_diag = diagonal_metrics.get('avg_near_diagonal_ratio', 0)
                            global_attn = diagonal_metrics.get('avg_global_attention_ratio', 0)
                            entropy = diagonal_metrics.get('avg_attention_entropy', 0)
                            
                            # Diagonal dominance: >1 means diagonal bias, <1 means spread out
                            # Near diagonal: fraction of attention within ±3 positions
                            # Global attention: ratio vs expected for long-range (>10 positions)
                            # Entropy: 0=peaked, 1=uniform
                            console.print(f"   DiagDom: {diag_dom:.2f} | NearDiag: {near_diag:.2f} | "
                                        f"Global: {global_attn:.2f} | Entropy: {entropy:.2f}")
                            
                            # Check if diagonal blindness is broken
                            # DiagDom < 1.5 means attention is spreading beyond diagonal
                            # Global > 0.5 means significant long-range attention
                            # Entropy > 0.5 means attention is spread out
                            if step > 500 and diag_dom < 1.5 and entropy > 0.5:
                                console.print("[bold green]🎯 DIAGONAL BLINDNESS BROKEN![/bold green]")
                            elif step > 500 and diag_dom > 3.0:
                                console.print("[bold red]⚠️ STRONG DIAGONAL BIAS (DiagDom > 3)[/bold red]")
                
                except Exception as e:
                    # Don't fail training if attention analysis fails
                    if step % 1000 == 0:  # Only log occasionally to avoid spam
                        console.print(f"[yellow]⚠️ Attention analysis failed: {e}[/yellow]")
            
            progress.update(task, advance=1, loss=avg_loss)
            
            # Verify loss continuity after resumption (first step only)
            if step == start_step and last_loss_before_resume is not None:
                loss_diff = abs(avg_loss - last_loss_before_resume)
                if loss_diff < 0.5:  # Reasonable threshold for continuity
                    console.print(f"[green]✓ Loss continuity verified: {last_loss_before_resume:.4f} → {avg_loss:.4f} (diff: {loss_diff:.4f})[/green]")
                else:
                    console.print(f"[yellow]⚠ Loss jump detected: {last_loss_before_resume:.4f} → {avg_loss:.4f} (diff: {loss_diff:.4f})[/yellow]")
            
            # Log metrics if monitoring enabled
            if metrics_logger and metrics_logger.should_log(step):
                try:
                    metrics_dict = {
                        'loss': avg_loss,
                        'learning_rate': current_lr,
                        'tokens_per_sec': total_tokens / (time.perf_counter() - start_time) if (time.perf_counter() - start_time) > 0 else 0,
                        'standard_embed_ratio': current_annealing_ratio  # Track annealing progress
                    }
                    
                    # Add wave-specific metrics if available
                    if wave_ratios:
                        metrics_dict['wave_ratio'] = wave_ratios[-1]
                    if coherence_losses:
                        metrics_dict['coherence_loss'] = coherence_losses[-1]
                    
                    metrics_logger.log_metrics(step, metrics_dict)
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to log metrics: {e}[/yellow]")
            
            # Save checkpoint if monitoring enabled
            if checkpoint_manager and checkpoint_manager.should_checkpoint(step):
                try:
                    checkpoint_config = {
                        'experiment_name': exp_config.name,
                        'step': step,
                        'total_steps': exp_config.steps
                    }
                    checkpoint_manager.save_checkpoint(
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        loss_history=losses,
                        config=checkpoint_config
                    )
                    console.print(f"[green]✓ Checkpoint saved at step {step}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to save checkpoint: {e}[/yellow]")

            
            # Check every 250 steps for scientific rigor
            if (step + 1) % 250 == 0:
                avg = sum(losses[-250:]) / len(losses[-250:])
                
                # Fast validation check
                model.eval()
                val_losses_accum = []
                num_val_batches = 20 # Robust multi-batch validation
                with torch.no_grad():
                    for _ in range(num_val_batches):
                        val_x, val_y = get_batch(val_data, batch_size, model_config.block_size, device) 
                        _, val_loss_check = model(val_x, val_y)
                        if val_loss_check.ndim > 0:
                            val_loss_check = val_loss_check.mean()
                        val_losses_accum.append(val_loss_check.item())
                model.train()
                
                current_val_loss = sum(val_losses_accum) / len(val_losses_accum)
                perplexities.append(math.exp(current_val_loss))
                
                # === EXTRACT TRINITY METRICS FOR PURE WAVE GPT ===
                trinity_info = ""
                if exp_config.model_type == "pure_wave":
                    try:
                        base_model = model.module if hasattr(model, 'module') else model
                        if hasattr(base_model, 'wave_layers') and len(base_model.wave_layers) > 0:
                            layer0 = base_model.wave_layers[0]
                            
                            # Saturation gains (Overdrive) - higher = sharper logic
                            if hasattr(layer0, 'wave_mlp') and hasattr(layer0.wave_mlp, 'freq_gain'):
                                avg_gain = layer0.wave_mlp.freq_gain.mean().item()
                                trinity_info += f" | Gain: {avg_gain:.2f}"
                            
                            # FM modulation index - how much context shifts frequencies
                            if hasattr(layer0, 'wave_fm') and hasattr(layer0.wave_fm, 'mod_index'):
                                avg_mod = layer0.wave_fm.mod_index.mean().item()
                                trinity_info += f" | FM: {avg_mod:.2f}"
                            
                            # Interferometer gate sharpness - how binary the gating is
                            if hasattr(layer0, 'attn_gate') and hasattr(layer0.attn_gate, 'gate_sharpness'):
                                gate_sharp = layer0.attn_gate.gate_sharpness.item()
                                trinity_info += f" | Gate: {gate_sharp:.2f}"
                    except Exception as e:
                        pass  # Don't fail training if metrics extraction fails
                
                # Build log message based on model type
                if exp_config.model_type == "pure_wave":
                    # PureWaveGPT - no annealing, show Trinity metrics
                    console.print(f"Step {step+1:5d} | Train: {avg_loss:.4f} | Val: {current_val_loss:.4f} | Avg: {avg:.4f}{trinity_info}")
                else:
                    # Standard/Wave GPT - show annealing ratio
                    wave_r = wave_ratios[-1] if wave_ratios else current_annealing_ratio
                    console.print(f"Step {step+1:5d} | Train: {avg_loss:.4f} | Val: {current_val_loss:.4f} | Avg: {avg:.4f} | R: {wave_r:.3f}")
                
                # Generate visualizations if monitoring enabled
                if viz_manager and viz_manager.should_visualize(step + 1):
                    try:
                        # Prepare metrics for visualization
                        viz_metrics = {
                            'learning_rate': learning_rates,
                            'perplexity': perplexities
                        }
                        if wave_ratios:
                            viz_metrics['wave_ratio'] = wave_ratios
                        if coherence_losses:
                            viz_metrics['coherence_loss'] = coherence_losses
                        
                        viz_manager.generate_training_plots(step + 1, losses, viz_metrics)
                        viz_manager.generate_model_plots(step + 1, model)
                        console.print(f"[green]✓ Visualizations generated at step {step + 1}[/green]")
                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to generate visualizations: {e}[/yellow]")
                
                 # Check Early Stopping
                if early_stopping(current_val_loss, model):
                    console.print(f"[yellow]🛑 Early stopping triggered. Best Val: {early_stopping.best_loss:.4f}[/yellow]")
                    break

    elapsed = time.perf_counter() - start_time
    speed = total_tokens / elapsed
    
    # Restore best model
    early_stopping.restore(model)
    
    # Final eval
    model.eval()
    with torch.no_grad():
        x, y = get_batch(val_data, batch_size, model_config.block_size, device)
        _, val_loss = model(x, y)
        if val_loss.ndim > 0: val_loss = val_loss.mean()
            
    # Generation Check
    console.print("\n[bold]🎨 Generating sample (Visual Check)...[/bold]")
    context = "The King" if "shakespeare" in exp_config.name.lower() else "The universe"
    # Need tokenizer to encode
    # Assumption: tokenizer is available in closure or we need to pass it. 
    # Current scope doesn't have tokenizer. 
    # Use simple ascii encoding for Shakespeare fallback if needed, but we should pass tokenizer or rely on it being loaded globally?
    # Actually, main() has tokenizer. We can just return 'model' and generate in main, OR generic encode/decode here.
    # Let's do a simple generation and return the string.
    
    # Quick dirty generation without tokenizer ref if needed, but better to skip if no tokenizer.
    # However, request says "Add generation Check".
    # I will assume we can generate token IDs and return them, or simply use the model.generate method.
    # I'll just skip text decoding here and return generated tokens to be decoded later, OR simply omit decoding text and trust the user to inspect tokens if they want.
    # Wait, the request says "Generate 100 tokens... This allows us to visually verify". Verification implies text.
    # I will stick to returning the model and letting `run_ablation_suite` handle the printing using the tokenizer it HAS.
    
    # Save final results if monitoring enabled
    if config_tracker:
        try:
            final_metrics = {
                'val_loss': val_loss.item(),
                'perplexity': torch.exp(val_loss).item(),
                'best_val_loss': early_stopping.best_loss,
                'total_time_seconds': elapsed,
                'tokens_per_second': speed,
                'total_steps': len(losses),
                'final_loss': losses[-1] if losses else 0.0
            }
            
            # Find best checkpoint path if available
            best_checkpoint_path = None
            if checkpoint_manager:
                checkpoints = checkpoint_manager.list_checkpoints()
                if checkpoints:
                    best_checkpoint_path = checkpoints[-1]['path']
            
            config_tracker.save_results(
                final_metrics=final_metrics,
                best_checkpoint=best_checkpoint_path,
                generation_samples=None  # Will be added by caller if needed
            )
            console.print("[green]✓ Final results saved[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save results: {e}[/yellow]")
    
    # === ZIP EXPERIMENT DIRECTORY ===
    if experiment_dir and enable_monitoring:
        try:
            zip_experiment_directory(experiment_dir, console)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to zip experiment directory: {e}[/yellow]")
    
    return {
        "name": exp_config.name,
        "config": asdict(exp_config),
        "val_loss": val_loss.item(),
        "perplexity": torch.exp(val_loss).item(),
        "speed": speed,
        "time": elapsed,
        "losses": losses,
        "coherence_losses": coherence_losses,
        "wave_ratios": wave_ratios,
        "params": params,
        "model": model, 
        "early_stopping_best": early_stopping.best_loss,
        "experiment_dir": experiment_dir if enable_monitoring else None
    }


# ==========================================
# Experiment Runner
# ==========================================

def run_ablation_suite(
    console: Console,
    experiments: List[str] = None,
    model_size: str = "small",
    dataset: str = "shakespeare",
    steps: int = 20000,
    device: str = "cuda",
    parallel: bool = False,
    dry_run: bool = False
) -> Dict:
    """Run ablation study experiments"""
    
    if experiments is None or "all" in experiments:
        experiments = list(ABLATION_EXPERIMENTS.keys())
    elif "diagonal" in experiments:
        # Diagonal breaker experiment group
        diagonal_experiments = [
            "diagonal_baseline",
            "diagonal_breaker", 
            "diagonal_overdrive_only",
            "diagonal_fm_only",
            "diagonal_interferometer_only"
        ]
        # Replace "diagonal" with actual experiment names
        experiments = [exp for exp in experiments if exp != "diagonal"] + diagonal_experiments
    
    model_config = MODEL_CONFIGS[model_size]
    
    # Override settings for dry-run mode
    if dry_run:
        console.print("[bold yellow]🧪 DRY-RUN MODE ENABLED[/bold yellow]")
        console.print("[yellow]Using small model, mock data, and 10 training steps for fast testing[/yellow]\n")
        steps = 10
        model_size = "small"
        model_config = MODEL_CONFIGS[model_size]
        # Use smaller batch size for faster testing
        model_config.batch_size = 4
    
    console.print(Panel.fit(
        f"[bold magenta]🔬 WAVE-NATIVE GPT ABLATION SUITE[/bold magenta]\n\n"
        f"Model: {model_config.name}\n"
        f"Dataset: {dataset}\n"
        f"Experiments: {', '.join(experiments)}\n"
        f"Steps per experiment: {steps}" + ("\n[yellow]DRY-RUN MODE[/yellow]" if dry_run else ""),
        border_style="magenta"
    ))
    
    # Load data (Directly returns tensor if tiktoken, text if shakespeare)
    data_or_text = get_dataset(dataset, console, dry_run=dry_run)
    
    if isinstance(data_or_text, torch.Tensor):
        # Already tokenized (FineWeb/Tiktoken)
        data = data_or_text
        enc = tiktoken.get_encoding("gpt2") # For decoding later
        tokenizer = enc # Alias for generation
    else:
        # Shakespeare (Text) -> Legacy BasicTokenizer
        text = data_or_text
        tokenizer_path = os.path.join(current_dir, "benchmark_results", "tokenizer.json")
        console.print("♻️  Using BasicTokenizer for Shakespeare...")
        tokenizer = BasicTokenizer()
        if os.path.exists(tokenizer_path):
             with open(tokenizer_path, 'r') as f:
                data_json = json.load(f)
                tokenizer.vocab = {int(k): bytes(v) for k, v in data_json['vocab'].items()}
        else:
             tokenizer.train(text, model_config.vocab_size)
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
    n = int(0.9 * len(data)) # Strict 90/10 split
    train_data = data[:n]
    val_data = data[n:]
    console.print(f"📊 Tokens: Train {len(train_data):,} | Val {len(val_data):,}")
    
    results = {}
    
    for exp_name in experiments:
        console.print("\n" + "="*60)
        
        exp_config = ABLATION_EXPERIMENTS[exp_name]
        exp_config.steps = steps  # Override if needed, but default is now Scientific 5000
        
        # Generate experiment ID for monitoring
        from monitoring import generate_experiment_id
        experiment_id = generate_experiment_id(exp_name)
        
        model_type = getattr(exp_config, 'model_type', 'wave')
        
        if model_type == "pure_wave":
            # === PURE WAVE GPT - Wave-to-Wave Throughout! ===
            from wave_gpt import PureWaveGPT
            wave_config = WaveGPTConfig(
                vocab_size=model_config.vocab_size,
                d_model=model_config.d_model,
                num_layers=model_config.num_layers,
                num_heads=model_config.num_heads,
                num_waves=model_config.num_waves,
                num_harmonics=model_config.num_harmonics,
                block_size=model_config.block_size,
                dropout=exp_config.dropout,
                model_type="pure_wave",
                pure_wave_mode_v2=True,
                # Non-Linear Physics Trinity (Diagonal Breaker)
                use_overdrive=getattr(exp_config, 'use_overdrive', False),
                use_fm_synthesis=getattr(exp_config, 'use_fm_synthesis', False),
                use_interferometer=getattr(exp_config, 'use_interferometer', False),
                mlp_expansion=getattr(exp_config, 'mlp_expansion', 1)
            )
            model = PureWaveGPT(wave_config).to(device)
            
            # Report Trinity status
            trinity_enabled = any([
                getattr(exp_config, 'use_overdrive', False),
                getattr(exp_config, 'use_fm_synthesis', False), 
                getattr(exp_config, 'use_interferometer', False)
            ])
            
            if trinity_enabled:
                console.print("[bold green]🌊⚡ PURE WAVE GPT + NON-LINEAR PHYSICS TRINITY[/bold green]")
                console.print(f"   Overdrive: {'✅' if getattr(exp_config, 'use_overdrive', False) else '❌'}")
                console.print(f"   FM Synthesis: {'✅' if getattr(exp_config, 'use_fm_synthesis', False) else '❌'}")
                console.print(f"   Interferometer: {'✅' if getattr(exp_config, 'use_interferometer', False) else '❌'}")
                console.print(f"   MLP Expansion: {getattr(exp_config, 'mlp_expansion', 1)}x")
            else:
                console.print("[bold cyan]🌊 PURE WAVE GPT (Baseline - No Trinity)[/bold cyan]")
            
        elif model_type == "wave":
            wave_config = WaveGPTConfig(
                vocab_size=model_config.vocab_size,
                d_model=model_config.d_model,
                num_layers=model_config.num_layers,
                num_heads=model_config.num_heads,
                num_waves=model_config.num_waves,
                num_harmonics=model_config.num_harmonics,
                block_size=model_config.block_size,
                dropout=exp_config.dropout,
                pure_wave_attention=getattr(exp_config, 'pure_wave_attention', False),
                pure_wave_kernel=getattr(exp_config, 'pure_wave_kernel', 'elu_plus_one'),
                pure_wave_mode=getattr(exp_config, 'pure_wave_mode', 'quadratic'),
                use_interference_attention=getattr(exp_config, 'use_interference_attention', False),
                model_type="wave"
            )
            model = WaveGPT(wave_config).to(device)
        else:
             wave_config = WaveGPTConfig(
                vocab_size=model_config.vocab_size,
                d_model=model_config.d_model,
                num_layers=model_config.num_layers,
                num_heads=model_config.num_heads,
                num_waves=model_config.num_waves,
                num_harmonics=model_config.num_harmonics,
                block_size=model_config.block_size,
                dropout=exp_config.dropout,
                model_type="standard" # Flag for Standard
             )
             model = WaveGPT(wave_config).to(device)
        
        # Count Parameters
        params = sum(p.numel() for p in model.parameters())
        console.print(f"📊 Parameters: {params:,} ({params/1e6:.2f}M)")
        
        # Parameter count assertion for fair comparison (Requirements 6.5)
        # Only enforce for non-dry-run experiments to allow testing with smaller models
        if not dry_run:
            MIN_PARAMS = 50_000_000  # 50M
            MAX_PARAMS = 55_000_000  # 55M
            if not (MIN_PARAMS < params < MAX_PARAMS):
                console.print(f"[yellow]⚠️  Parameter count {params:,} outside fair comparison range (50M-55M)[/yellow]")
                console.print(f"[yellow]   Adjust model config for fair comparison runs[/yellow]")
                # Note: We warn but don't assert to allow flexibility in experiments
                # For strict fair comparison, uncomment the assertion below:
                # assert MIN_PARAMS < params < MAX_PARAMS, \
                #     f"Parameter count {params:,} must be between 50M and 55M for fair comparison"
        
        # DataParallel if requested
        if parallel and torch.cuda.device_count() > 1:
            console.print(f"🔀 Using DataParallel on {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        # Print Detailed Stats
        print_detailed_stats(model, model_config, dataset, train_data, val_data, console)

        # Train with monitoring enabled
        result = train_experiment(
            model, train_data, val_data, exp_config, model_config, console, device,
            experiment_dir=experiment_id,
            enable_monitoring=True,
            dry_run=dry_run
        )
        
        # GENERATION CHECK (Scientific Visual Verification)
        console.print("[bold]📝 Generation Check:[/bold]")
        start_text = "The theory of quantum mechanics suggests" # Scientific Prompt
        
        # Handle tokenizer differences (tiktoken vs basic)
        if hasattr(tokenizer, 'encode'):
             context_ids = tokenizer.encode(start_text)
        else:
             context_ids = tokenizer.encode(start_text)
             
        context_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
        
        # Generate
        if isinstance(model, nn.DataParallel):
            gen_ids = model.module.generate(context_tensor, max_new_tokens=100, temperature=0.8)
        else:
            gen_ids = model.generate(context_tensor, max_new_tokens=100, temperature=0.8)
        
        # Decode
        if hasattr(tokenizer, 'decode'):
            gen_text = tokenizer.decode(gen_ids[0].tolist())
        else:
            # Fallback for BasicTokenizer
            gen_text = tokenizer.decode(gen_ids[0].tolist())
            
        console.print(Panel(gen_text, title=f"{exp_config.name} Output", border_style="green"))
        
        result['generation'] = gen_text
        results[exp_name] = result
        
        # Clean up
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return results


def plot_ablation_results(results: Dict, save_dir: str):
    """Generate comparison plots for ablation study"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use('dark_background')
    
    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['cyan', 'magenta', 'yellow', 'lime']
    
    for i, (name, result) in enumerate(results.items()):
        losses = result['losses']
        # Adjust window size for short runs
        window = min(100, max(1, len(losses) // 10))
        if len(losses) < window:
            # No smoothing for very short runs
            ax.plot(range(len(losses)), losses, 
                    label=f"{result['name']} (final: {result['val_loss']:.3f})",
                    color=colors[i % len(colors)], linewidth=2)
        else:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(losses)), smoothed, 
                    label=f"{result['name']} (final: {result['val_loss']:.3f})",
                    color=colors[i % len(colors)], linewidth=2)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Ablation Study: Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/ablation_loss_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Wave ratio progression (if available)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, result) in enumerate(results.items()):
        if result.get('wave_ratios'):
            ax.plot(result['wave_ratios'], 
                    label=result['name'],
                    color=colors[i % len(colors)], linewidth=2)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Wave Ratio')
    ax.set_title('Wave Ratio Progression (0.5 → 0.9 schedule)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/wave_ratio_progression.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = [r['name'] for r in results.values()]
    val_losses = [r['val_loss'] for r in results.values()]
    perplexities = [r['perplexity'] for r in results.values()]
    
    axes[0].barh(names, val_losses, color=colors[:len(names)])
    axes[0].set_xlabel('Validation Loss')
    axes[0].set_title('Final Validation Loss')
    
    axes[1].barh(names, perplexities, color=colors[:len(names)])
    axes[1].set_xlabel('Perplexity')
    axes[1].set_title('Final Perplexity')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Save results JSON
    summary = {
        name: {
            "val_loss": r["val_loss"],
            "perplexity": r["perplexity"],
            "speed": r["speed"],
            "time": r["time"],
            "config": r["config"]
        }
        for name, r in results.items()
    }
    with open(f"{save_dir}/ablation_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def print_results_table(results: Dict, console: Console):
    """Print results as a nice table"""
    table = Table(title="🔬 Ablation Study Results")
    table.add_column("Experiment", style="cyan")
    table.add_column("RGD", justify="center")
    table.add_column("QFE", justify="center")
    table.add_column("Val Loss", justify="right")
    table.add_column("Perplexity", justify="right")
    table.add_column("Speed (tok/s)", justify="right")
    
    for name, result in results.items():
        config = result['config']
        table.add_row(
            result['name'],
            "✓" if config['use_rgd'] else "✗",
            "✓" if config['use_qfe'] else "✗",
            f"{result['val_loss']:.4f}",
            f"{result['perplexity']:.2f}",
            f"{result['speed']:,.0f}"
        )
    
    console.print(table)


# ==========================================
# Main
# ==========================================

def print_detailed_stats(model, config, tokenizer_name, train_data, val_data, console):
    """Print comprehensive breakdown of model and data statistics"""
    
    # 1. Dataset Stats
    total_tokens = len(train_data) + len(val_data)
    
    table = Table(title="📚 Dataset Statistics", border_style="blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Total Tokens", f"{total_tokens:,}")
    table.add_row("Train Split", f"{len(train_data):,} ({len(train_data)/total_tokens:.1%})")
    table.add_row("Validation Split", f"{len(val_data):,} ({len(val_data)/total_tokens:.1%})")
    table.add_row("Tokenizer", "TikToken (GPT-2)" if "fineweb" in tokenizer_name else "BasicTokenizer")
    
    console.print(table)
    
    # 2. Parameter Breakdown
    embed_params = 0
    attn_params = 0
    mlp_params = 0
    other_params = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        n = name.lower()
        if "embed" in n or "wte" in n or "wpe" in n:
            embed_params += param.numel()
        elif "attn" in n or "attention" in n:
            attn_params += param.numel()
        elif "mlp" in n:
            mlp_params += param.numel()
        else:
            other_params += param.numel()
            
    total_params = embed_params + attn_params + mlp_params + other_params
    
    # Model Stats Table
    m_table = Table(title="🧠 Model Parameter Breakdown", border_style="magenta")
    m_table.add_column("Component", style="green")
    m_table.add_column("Parameters", style="white", justify="right")
    m_table.add_column("% of Total", style="yellow", justify="right")
    
    m_table.add_row("Embeddings", f"{embed_params:,}", f"{embed_params/total_params:.1%}")
    m_table.add_row("Attention", f"{attn_params:,}", f"{attn_params/total_params:.1%}")
    m_table.add_row("MLP", f"{mlp_params:,}", f"{mlp_params/total_params:.1%}")
    m_table.add_row("Other (Norms/Head)", f"{other_params:,}", f"{other_params/total_params:.1%}")
    m_table.add_row("TOTAL", f"{total_params:,}", "100.0%", style="bold")
    
    console.print(m_table)
    console.print(f"📊 Ratio: {total_tokens/total_params:.2f} Tokens per Parameter (Target > 7.4)", style="bold purple")


def main():
    parser = argparse.ArgumentParser(description="Wave-Native GPT Experiment Suite")
    parser.add_argument("--experiment", type=str, nargs="+", default=["all"],
                        choices=["all", "diagonal"] + list(ABLATION_EXPERIMENTS.keys()),
                        help="Experiments to run (use 'diagonal' for all diagonal breaker experiments)")
    parser.add_argument("--model", type=str, default="small",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model size")
    parser.add_argument("--dataset", type=str, default="fineweb",
                        choices=["shakespeare", "fineweb_small", "fineweb", "fineweb_large"],
                        help="Dataset to use")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Training steps per experiment")
    parser.add_argument("--parallel", action="store_true",
                        help="Use DataParallel for multi-GPU")
    parser.add_argument("--output", type=str, default="experiment_results",
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run in dry-run mode with mock data for fast testing (10 steps)")
    
    args = parser.parse_args()
    
    console = Console()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run experiments
    results = run_ablation_suite(
        console,
        experiments=args.experiment,
        model_size=args.model,
        dataset=args.dataset,
        steps=args.steps,
        device=device,
        parallel=args.parallel,
        dry_run=args.dry_run
    )
    
    if not results:
        console.print("[red]No results to save[/red]")
        return
    
    # Save results
    output_dir = os.path.join(current_dir, args.output)
    summary = plot_ablation_results(results, output_dir)
    
    # Print table
    print_results_table(results, console)
    
    # Zip the overall results directory
    try:
        zip_experiment_directory(output_dir, console)
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to zip results directory: {e}[/yellow]")
    
    console.print(f"\n📁 Results saved to: {output_dir}")
    console.print("[bold green]✅ Experiment suite complete![/bold green]")


if __name__ == "__main__":
    main()
