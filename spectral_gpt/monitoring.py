"""
Experiment Monitoring Infrastructure for Spectral GPT

Provides checkpointing, metrics logging, visualization, and configuration tracking
for robust experiment management with interruption recovery.
"""

import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Optimizer


# ==========================================
# Experiment ID Generation
# ==========================================

def get_git_hash() -> str:
    """Get current git commit hash (short form)"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def generate_experiment_id(experiment_name: str) -> str:
    """
    Generate unique experiment ID with timestamp and git hash.
    
    Format: {experiment_name}_{timestamp}_{git_hash}
    Example: wave_rgd_qfe_20241209_143022_a3f2b1c4
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Unique experiment ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_hash = get_git_hash()
    return f"{experiment_name}_{timestamp}_{git_hash}"


def create_experiment_directory(base_dir: str, experiment_id: str) -> Dict[str, str]:
    """
    Create directory structure for experiment.
    
    Structure:
        experiments/{experiment_id}/
        ├── checkpoints/
        ├── logs/
        └── visualizations/
    
    Args:
        base_dir: Base directory for all experiments
        experiment_id: Unique experiment identifier
        
    Returns:
        Dictionary mapping directory types to paths
    """
    exp_dir = Path(base_dir) / experiment_id
    
    dirs = {
        'root': str(exp_dir),
        'checkpoints': str(exp_dir / 'checkpoints'),
        'logs': str(exp_dir / 'logs'),
        'visualizations': str(exp_dir / 'visualizations')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


# ==========================================
# CheckpointManager
# ==========================================

class CheckpointManager:
    """
    Manages model checkpointing during training with automatic cleanup and resumption support.
    
    Features:
    - Atomic writes using temporary files to prevent corruption
    - Automatic cleanup of old checkpoints based on retention policy
    - Symlink to latest checkpoint for easy resumption
    - Saves model state, optimizer state, step, loss history, and config
    
    Example:
        >>> manager = CheckpointManager(experiment_dir="experiments/exp_001", 
        ...                             save_interval=1000, keep_last_n=3)
        >>> 
        >>> # During training
        >>> if manager.should_checkpoint(step):
        ...     manager.save_checkpoint(step, model, optimizer, loss_history, config)
        >>> 
        >>> # Resume from interruption
        >>> checkpoint = manager.load_latest_checkpoint()
        >>> if checkpoint:
        ...     model.load_state_dict(checkpoint['model_state_dict'])
        ...     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ...     start_step = checkpoint['step'] + 1
    """
    
    def __init__(self, 
                 experiment_dir: str,
                 save_interval: int = 1000,
                 keep_last_n: int = 3):
        """
        Initialize CheckpointManager.
        
        Args:
            experiment_dir: Root directory for experiment
            save_interval: Steps between checkpoints
            keep_last_n: Number of recent checkpoints to retain
        """
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.latest_symlink = self.checkpoint_dir / 'checkpoint_latest.pt'
    
    def should_checkpoint(self, step: int) -> bool:
        """
        Check if current step requires checkpointing.
        
        Args:
            step: Current training step
            
        Returns:
            True if checkpoint should be saved
        """
        return step > 0 and step % self.save_interval == 0
    
    def save_checkpoint(self,
                       step: int,
                       model: nn.Module,
                       optimizer: Optimizer,
                       loss_history: List[float],
                       config: Dict) -> str:
        """
        Save checkpoint to disk with atomic write.
        
        Uses temporary file and atomic rename to prevent corruption from
        interrupted writes.
        
        Args:
            step: Current training step
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            loss_history: List of training losses
            config: Experiment configuration dictionary
            
        Returns:
            Path to saved checkpoint file
        """
        # Handle DataParallel models
        if isinstance(model, nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint_data = {
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'git_hash': get_git_hash()
        }
        
        # Checkpoint filename
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
        temp_path = self.checkpoint_dir / f'checkpoint_step_{step}.pt.tmp'
        
        # Atomic write: save to temp file first
        torch.save(checkpoint_data, temp_path)
        
        # Atomic rename (POSIX systems guarantee atomicity)
        temp_path.replace(checkpoint_path)
        
        # Update symlink to latest checkpoint
        if self.latest_symlink.exists() or self.latest_symlink.is_symlink():
            self.latest_symlink.unlink()
        
        # Create relative symlink
        try:
            self.latest_symlink.symlink_to(checkpoint_path.name)
        except OSError:
            # Fallback for systems that don't support symlinks (Windows)
            # Just copy the file
            import shutil
            shutil.copy2(checkpoint_path, self.latest_symlink)
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """
        Load most recent checkpoint if exists.
        
        Returns:
            Checkpoint dictionary or None if no checkpoint exists
        """
        if not self.latest_symlink.exists():
            return None
        
        try:
            checkpoint = torch.load(self.latest_symlink, map_location='cpu', weights_only=True)
            return checkpoint
        except Exception as e:
            print(f"Warning: Failed to load checkpoint from {self.latest_symlink}: {e}")
            
            # Try to find the most recent checkpoint by step number
            checkpoints = sorted(
                self.checkpoint_dir.glob('checkpoint_step_*.pt'),
                key=lambda p: int(p.stem.split('_')[-1]),
                reverse=True
            )
            
            for checkpoint_path in checkpoints:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                    print(f"Loaded checkpoint from {checkpoint_path}")
                    return checkpoint
                except Exception:
                    continue
            
            return None
    
    def load_checkpoint(self, step: int) -> Optional[Dict]:
        """
        Load specific checkpoint by step number.
        
        Args:
            step: Step number of checkpoint to load
            
        Returns:
            Checkpoint dictionary or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
        
        if not checkpoint_path.exists():
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            return checkpoint
        except Exception as e:
            print(f"Warning: Failed to load checkpoint from {checkpoint_path}: {e}")
            return None
    
    def cleanup_old_checkpoints(self):
        """
        Remove checkpoints beyond retention limit.
        
        Keeps only the N most recent checkpoints based on step number.
        """
        # Get all checkpoint files (excluding symlink and temp files)
        checkpoints = sorted(
            [p for p in self.checkpoint_dir.glob('checkpoint_step_*.pt') 
             if not p.name.endswith('.tmp')],
            key=lambda p: int(p.stem.split('_')[-1]),
            reverse=True
        )
        
        # Remove old checkpoints beyond retention limit
        for checkpoint_path in checkpoints[self.keep_last_n:]:
            try:
                checkpoint_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete old checkpoint {checkpoint_path}: {e}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.
        
        Returns:
            List of dictionaries containing checkpoint info
        """
        checkpoints = []
        
        for checkpoint_path in sorted(self.checkpoint_dir.glob('checkpoint_step_*.pt')):
            if checkpoint_path.name.endswith('.tmp'):
                continue
            
            try:
                step = int(checkpoint_path.stem.split('_')[-1])
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
                
                checkpoints.append({
                    'step': step,
                    'path': str(checkpoint_path),
                    'size_mb': size_mb,
                    'modified': mtime.isoformat()
                })
            except Exception:
                continue
        
        return sorted(checkpoints, key=lambda x: x['step'])


# ==========================================
# MetricsLogger
# ==========================================

class MetricsLogger:
    """
    Incrementally logs training metrics with immediate persistence.
    
    Features:
    - JSONL format (one JSON object per line) for easy streaming and parsing
    - Immediate buffer flush after each write to ensure persistence
    - Separate human-readable log file for debugging
    - Tracks step, loss, learning_rate, and model-specific metrics
    
    Example:
        >>> logger = MetricsLogger(log_dir="experiments/exp_001/logs", log_interval=10)
        >>> 
        >>> # During training
        >>> if logger.should_log(step):
        ...     metrics = {
        ...         'loss': 0.5,
        ...         'learning_rate': 0.001,
        ...         'wave_ratio': 0.8,
        ...         'perplexity': 1.65
        ...     }
        ...     logger.log_metrics(step, metrics)
        >>> 
        >>> # Load metrics for analysis
        >>> all_metrics = logger.load_metrics()
        >>> latest_step = logger.get_latest_step()
    """
    
    def __init__(self, log_dir: str, log_interval: int = 10):
        """
        Initialize MetricsLogger.
        
        Args:
            log_dir: Directory for log files
            log_interval: Steps between log entries
        """
        self.log_dir = Path(log_dir)
        self.log_interval = log_interval
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file paths
        self.metrics_file = self.log_dir / 'metrics.jsonl'
        self.training_log = self.log_dir / 'training.log'
        
        # Initialize log files if they don't exist
        if not self.metrics_file.exists():
            self.metrics_file.touch()
        if not self.training_log.exists():
            self.training_log.touch()
    
    def should_log(self, step: int) -> bool:
        """
        Check if current step requires logging.
        
        Args:
            step: Current training step
            
        Returns:
            True if metrics should be logged
        """
        return step % self.log_interval == 0
    
    def log_metrics(self,
                   step: int,
                   metrics: Dict[str, float],
                   flush: bool = True):
        """
        Append metrics to log file.
        
        Writes to both JSONL format (for programmatic analysis) and
        human-readable format (for debugging).
        
        Args:
            step: Current training step
            metrics: Dictionary of metric names to values
            flush: Whether to flush buffer immediately (default: True)
        """
        # Add timestamp and step to metrics
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # Write to JSONL file with line buffering (buffering=1)
        with open(self.metrics_file, 'a', buffering=1) as f:
            f.write(json.dumps(log_entry) + '\n')
            if flush:
                f.flush()
        
        # Write to human-readable log
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics_str = ', '.join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" 
                                 for k, v in metrics.items()])
        log_line = f"[{timestamp_str}] Step {step}: {metrics_str}\n"
        
        with open(self.training_log, 'a', buffering=1) as f:
            f.write(log_line)
            if flush:
                f.flush()
    
    def load_metrics(self) -> List[Dict]:
        """
        Load all logged metrics from file.
        
        Returns:
            List of metric dictionaries, one per logged step
        """
        if not self.metrics_file.exists():
            return []
        
        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse metrics line: {line[:50]}... Error: {e}")
                        continue
        
        return metrics
    
    def get_latest_step(self) -> int:
        """
        Get the last logged step number.
        
        Returns:
            Last logged step, or 0 if no metrics logged yet
        """
        metrics = self.load_metrics()
        if not metrics:
            return 0
        
        return metrics[-1]['step']
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of logged metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        metrics = self.load_metrics()
        if not metrics:
            return {}
        
        # Extract numeric metrics
        numeric_keys = set()
        for entry in metrics:
            for key, value in entry.items():
                if isinstance(value, (int, float)) and key not in ['step']:
                    numeric_keys.add(key)
        
        summary = {
            'total_steps': len(metrics),
            'first_step': metrics[0]['step'],
            'last_step': metrics[-1]['step']
        }
        
        # Compute statistics for each numeric metric
        for key in numeric_keys:
            values = [entry[key] for entry in metrics if key in entry and isinstance(entry[key], (int, float))]
            if values:
                summary[f'{key}_min'] = min(values)
                summary[f'{key}_max'] = max(values)
                summary[f'{key}_mean'] = sum(values) / len(values)
                summary[f'{key}_final'] = values[-1]
        
        return summary


# ==========================================
# VisualizationManager
# ==========================================

class VisualizationManager:
    """
    Generates and saves visualizations during training.
    
    Features:
    - Timestamped filenames for tracking evolution over training
    - Separate methods for training dynamics vs model internals
    - Matplotlib with dark theme for consistency
    - Error handling for missing model attributes
    - Plots include: loss curves, frequency distributions, phase distributions, 
      harmonics, wave packets, interference patterns
    
    Example:
        >>> viz_manager = VisualizationManager(
        ...     viz_dir="experiments/exp_001/visualizations",
        ...     viz_interval=1000
        ... )
        >>> 
        >>> # During training
        >>> if viz_manager.should_visualize(step):
        ...     viz_manager.generate_training_plots(step, loss_history, metrics)
        ...     viz_manager.generate_model_plots(step, model)
    """
    
    def __init__(self,
                 viz_dir: str,
                 viz_interval: int = 1000):
        """
        Initialize VisualizationManager.
        
        Args:
            viz_dir: Directory for visualization files
            viz_interval: Steps between visualization generation
        """
        self.viz_dir = Path(viz_dir)
        self.viz_interval = viz_interval
        
        # Create visualization directory if it doesn't exist
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        self._setup_matplotlib_style()
    
    def _setup_matplotlib_style(self):
        """Configure matplotlib with dark theme and consistent styling"""
        try:
            import matplotlib.pyplot as plt
            
            # Use dark background style
            plt.style.use('dark_background')
            
            # Set default figure parameters
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['figure.dpi'] = 100
            plt.rcParams['savefig.dpi'] = 150
            plt.rcParams['savefig.bbox'] = 'tight'
            
            # Font settings
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['xtick.labelsize'] = 9
            plt.rcParams['ytick.labelsize'] = 9
            plt.rcParams['legend.fontsize'] = 9
            
            # Grid settings
            plt.rcParams['grid.alpha'] = 0.3
            plt.rcParams['grid.linestyle'] = '--'
            
        except Exception as e:
            print(f"Warning: Failed to setup matplotlib style: {e}")
    
    def should_visualize(self, step: int) -> bool:
        """
        Check if current step requires visualization.
        
        Args:
            step: Current training step
            
        Returns:
            True if visualizations should be generated
        """
        return step > 0 and step % self.viz_interval == 0
    
    def generate_training_plots(self,
                                step: int,
                                loss_history: List[float],
                                metrics: Dict[str, List[float]]):
        """
        Generate plots of training dynamics.
        
        Creates visualizations for:
        - Loss curves (training and validation if available)
        - Learning rate schedule
        - Other tracked metrics (perplexity, wave_ratio, etc.)
        
        Args:
            step: Current training step
            loss_history: List of loss values over training
            metrics: Dictionary mapping metric names to lists of values
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create figure with subplots
            num_metrics = len(metrics) + 1  # +1 for loss
            ncols = 2
            nrows = (num_metrics + 1) // 2
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
            if nrows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            # Plot 1: Loss curve
            ax = axes[0]
            steps = list(range(len(loss_history)))
            ax.plot(steps, loss_history, linewidth=2, color='cyan', label='Training Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title(f'Training Loss (Step {step})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Plot other metrics
            plot_idx = 1
            for metric_name, metric_values in metrics.items():
                if plot_idx >= len(axes):
                    break
                
                ax = axes[plot_idx]
                steps = list(range(len(metric_values)))
                
                # Choose color based on metric type
                color = 'yellow' if 'lr' in metric_name.lower() else 'magenta'
                if 'perplexity' in metric_name.lower():
                    color = 'green'
                elif 'wave' in metric_name.lower():
                    color = 'orange'
                
                ax.plot(steps, metric_values, linewidth=2, color=color, label=metric_name)
                ax.set_xlabel('Step')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} (Step {step})')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                plot_idx += 1
            
            # Hide unused subplots
            for idx in range(plot_idx, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Training Dynamics - Step {step}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save with timestamped filename
            filename = f'training_dynamics_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved training plots: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to generate training plots: {e}")
    
    def generate_model_plots(self,
                            step: int,
                            model: nn.Module):
        """
        Generate plots of model internals (wave properties).
        
        Creates visualizations for:
        - Frequency distributions
        - Phase distributions
        - Harmonic amplitudes
        - Wave packets
        - Interference patterns
        
        Args:
            step: Current training step
            model: Model to visualize
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Check if model has wave properties
            # Handle both WaveGPT (has embedding) and PureWaveGPT (has wave_excitation)
            if hasattr(model, 'module'):
                model = model.module
            
            if hasattr(model, 'embedding'):
                # Standard WaveGPT
                embedding = model.embedding
            elif hasattr(model, 'wave_excitation'):
                # PureWaveGPT - use wave_excitation as embedding
                embedding = model.wave_excitation
            else:
                print(f"Warning: Model does not have 'embedding' or 'wave_excitation' attribute, skipping model plots")
                return
            
            # Check for required wave attributes
            has_freqs = hasattr(embedding, 'base_freqs')
            has_phases = hasattr(embedding, 'phases')
            # Handle both WavePacketEmbedding (harmonic_amps) and PureWaveExcitation (amplitudes)
            has_harmonics = hasattr(embedding, 'harmonic_amps') or hasattr(embedding, 'amplitudes')
            
            # Debug: Print what attributes are available
            if not has_harmonics:
                available_attrs = [attr for attr in dir(embedding) if not attr.startswith('_')]
                print(f"Debug: Available embedding attributes: {available_attrs}")
            
            if not (has_freqs or has_phases or has_harmonics):
                print(f"Warning: Model embedding does not have wave attributes, skipping model plots")
                return
            
            # Generate frequency distribution plot
            if has_freqs:
                self._plot_frequencies(step, embedding)
            
            # Generate phase distribution plot
            if has_phases:
                self._plot_phases(step, embedding)
            
            # Generate harmonics plot
            if has_harmonics:
                self._plot_harmonics(step, embedding)
            
            # Generate wave packets plot (requires all attributes)
            if has_freqs and has_phases and has_harmonics:
                self._plot_wave_packets(step, embedding)
            
            # Generate new diagnostic plots
            self._plot_attention_patterns(step, model)
            self._plot_frequency_specific_attention(step, model)
            self._plot_phase_coherence(step, model)
            self._plot_gradient_diagnostics(step, model)
            self._plot_wave_space_similarity(step, model)
            self._plot_interference_strength(step, model)
            
        except Exception as e:
            print(f"Warning: Failed to generate model plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_frequencies(self, step: int, embedding):
        """Plot frequency distribution across tokens and waves"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            freqs = embedding.base_freqs.detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Heatmap of frequencies (first 100 tokens)
            n_tokens = min(100, freqs.shape[0])
            im = axes[0].imshow(freqs[:n_tokens].T, aspect='auto', cmap='viridis')
            axes[0].set_xlabel('Token ID')
            axes[0].set_ylabel('Wave Component')
            axes[0].set_title(f'Base Frequencies per Token (Step {step})')
            plt.colorbar(im, ax=axes[0], label='Frequency (Hz)')
            
            # Plot 2: Histogram of all frequencies
            axes[1].hist(freqs.flatten(), bins=50, color='cyan', alpha=0.7, edgecolor='white')
            axes[1].set_xlabel('Frequency (Hz)')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'Frequency Distribution (Step {step})')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f'frequencies_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved frequency plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot frequencies: {e}")
    
    def _plot_phases(self, step: int, embedding):
        """Plot phase distribution across tokens and waves"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            phases = embedding.phases.detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Heatmap of phases (first 100 tokens)
            n_tokens = min(100, phases.shape[0])
            im = axes[0].imshow(phases[:n_tokens].T, aspect='auto', cmap='twilight')
            axes[0].set_xlabel('Token ID')
            axes[0].set_ylabel('Wave Component')
            axes[0].set_title(f'Token Phases (Step {step})')
            plt.colorbar(im, ax=axes[0], label='Phase (radians)')
            
            # Plot 2: Polar plot of phases (first wave component)
            ax_polar = plt.subplot(122, projection='polar')
            phase0 = phases[:n_tokens, 0]  # First wave component
            colors = plt.cm.viridis(np.linspace(0, 1, n_tokens))
            ax_polar.scatter(phase0, np.ones_like(phase0), c=colors, s=30, alpha=0.6)
            ax_polar.set_title(f'Phase Distribution on Unit Circle\n(First Wave, Step {step})')
            
            plt.tight_layout()
            
            filename = f'phases_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved phase plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot phases: {e}")
    
    def _plot_harmonics(self, step: int, embedding):
        """Plot harmonic amplitude distribution"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Handle both WavePacketEmbedding (harmonic_amps) and PureWaveExcitation (amplitudes)
            if hasattr(embedding, 'harmonic_amps'):
                harm_amps = embedding.harmonic_amps.detach().cpu().numpy()
            elif hasattr(embedding, 'amplitudes'):
                harm_amps = embedding.amplitudes.detach().cpu().numpy()
            else:
                print("Warning: No harmonic amplitudes found, skipping harmonics plot")
                return
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Average harmonic profile across tokens
            mean_harmonics = harm_amps.mean(axis=0)  # (waves, harmonics)
            im = axes[0].imshow(mean_harmonics, aspect='auto', cmap='plasma')
            axes[0].set_xlabel('Harmonic (1f, 2f, 3f, ...)')
            axes[0].set_ylabel('Wave Component')
            axes[0].set_title(f'Mean Harmonic Amplitudes (Step {step})')
            plt.colorbar(im, ax=axes[0], label='Amplitude')
            
            # Plot 2: Harmonic distribution histogram
            for h in range(harm_amps.shape[2]):
                axes[1].hist(harm_amps[:, :, h].flatten(), bins=30, 
                           alpha=0.5, label=f'Harmonic {h+1}')
            axes[1].set_xlabel('Amplitude')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'Harmonic Amplitude Distribution (Step {step})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f'harmonics_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved harmonics plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot harmonics: {e}")
    
    def _plot_wave_packets(self, step: int, embedding):
        """Plot wave packets for sample tokens"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            freqs = embedding.base_freqs.detach().cpu().numpy()
            phases = embedding.phases.detach().cpu().numpy()
            
            # Handle both WavePacketEmbedding (harmonic_amps) and PureWaveExcitation (amplitudes)
            if hasattr(embedding, 'harmonic_amps'):
                harm_amps = embedding.harmonic_amps.detach().cpu().numpy()
            elif hasattr(embedding, 'amplitudes'):
                harm_amps = embedding.amplitudes.detach().cpu().numpy()
            else:
                print("Warning: No harmonic amplitudes found, skipping wave packets plot")
                return
            
            # Sample tokens to visualize
            n_tokens = freqs.shape[0]
            sample_tokens = [0, 1, min(10, n_tokens-1), min(50, n_tokens-1), 
                           min(100, n_tokens-1), min(500, n_tokens-1)]
            sample_tokens = [t for t in sample_tokens if t < n_tokens]
            
            fig, axes = plt.subplots(2, 3, figsize=(14, 7))
            axes = axes.flatten()
            
            # Time points for wave visualization (show position-based waves)
            positions = np.linspace(0, 50, 200)  # Show 50 positions (like sequence positions)
            
            for idx, tok_id in enumerate(sample_tokens):
                if idx >= len(axes):
                    break
                
                ax = axes[idx]
                wave_sum = np.zeros_like(positions)
                
                # Create wave packet like in physics image
                # Use only first 2 waves to avoid chaos
                for w in range(min(2, freqs.shape[1])):
                    base_f = freqs[tok_id, w]
                    phase = phases[tok_id, w]
                    
                    # Create multi-peak wave packet with revivals (like panels b,c,d in physics image)
                    
                    # Primary peak (main attention)
                    primary_center = 15
                    primary_width = 3.0 / (base_f + 0.1)
                    primary_envelope = np.exp(-0.5 * ((positions - primary_center) / primary_width) ** 2)
                    
                    # Secondary peak (revival/long-range attention)
                    secondary_center = 35
                    secondary_width = 5.0 / (base_f + 0.1)
                    secondary_envelope = 0.4 * np.exp(-0.5 * ((positions - secondary_center) / secondary_width) ** 2)
                    
                    # Beating pattern (creates oscillatory modulation)
                    beat_freq = base_f * 0.05
                    beat_modulation = 0.5 * (1.0 + np.cos(2 * np.pi * beat_freq * positions / 5.0))
                    
                    # Combined envelope with multiple peaks and beating
                    envelope = (primary_envelope + secondary_envelope) * beat_modulation
                    
                    # Add harmonics with envelope modulation
                    for h in range(min(2, harm_amps.shape[2])):  # Only first 2 harmonics
                        amp = harm_amps[tok_id, w, h]
                        freq = base_f * (h + 1)
                        
                        # Create wave packet: envelope * oscillation
                        wave_component = amp * envelope * np.cos(2 * np.pi * freq * positions / 10.0 + phase)
                        wave_sum += wave_component
                
                ax.plot(positions, wave_sum, linewidth=2, color='cyan')
                ax.set_title(f'Token {tok_id} Wave Packet')
                ax.set_xlabel('Position')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color='white', alpha=0.3, linewidth=0.5)
                
                # Add frequency info to title
                if freqs.shape[1] > 0:
                    main_freq = freqs[tok_id, 0]
                    ax.set_title(f'Token {tok_id} (f={main_freq:.2f}Hz)')
                
                ax.plot(t, wave_sum, linewidth=2, color='cyan')
                ax.set_title(f'Token {tok_id}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color='white', alpha=0.3, linewidth=0.5)
            
            # Hide unused subplots
            for idx in range(len(sample_tokens), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Wave Packets per Token (Step {step})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = f'wave_packets_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved wave packets plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot wave packets: {e}")
    
    def _plot_attention_patterns(self, step: int, model):
        """Plot attention pattern heatmaps comparing standard vs interference attention"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Check if model has attention layers
            if hasattr(model, 'module'):
                model = model.module
            
            # Handle both WaveGPT (layers) and PureWaveGPT (wave_layers)
            if hasattr(model, 'layers'):
                layers = model.layers
            elif hasattr(model, 'wave_layers'):
                layers = model.wave_layers
            else:
                print("Warning: Model does not have 'layers' or 'wave_layers' attribute, skipping attention patterns")
                return
            
            # Find attention layers
            attention_layers = []
            for layer in layers:
                if hasattr(layer, 'attn'):
                    attention_layers.append(layer.attn)
                elif hasattr(layer, 'wave_attention'):  # PureWaveGPT uses wave_attention
                    attention_layers.append(layer.wave_attention)
            
            if not attention_layers:
                print("Warning: No attention layers found, skipping attention patterns")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            
            # Sample a few layers to visualize
            sample_layers = min(6, len(attention_layers))
            
            for i in range(sample_layers):
                ax = axes[i]
                attn_layer = attention_layers[i]
                
                # Check if this is InterferenceAttention
                is_interference = hasattr(attn_layer, 'interference_strength')
                
                if is_interference and hasattr(attn_layer, 'last_attention_weights'):
                    # Plot interference attention pattern
                    attn_weights = attn_layer.last_attention_weights.detach().cpu().numpy()
                    if len(attn_weights.shape) == 4:  # [batch, heads, seq, seq]
                        attn_weights = attn_weights[0, 0]  # First batch, first head
                    elif len(attn_weights.shape) == 3:  # [heads, seq, seq]
                        attn_weights = attn_weights[0]  # First head
                    
                    im = ax.imshow(attn_weights[:50, :50], cmap='viridis', aspect='auto')
                    ax.set_title(f'Layer {i} - Interference Attention')
                    plt.colorbar(im, ax=ax, shrink=0.6)
                else:
                    # Create dummy standard attention pattern for comparison
                    seq_len = 50
                    # Standard attention tends to be more diagonal/local
                    standard_attn = np.exp(-0.1 * np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)))
                    standard_attn = standard_attn / standard_attn.sum(axis=1, keepdims=True)
                    
                    im = ax.imshow(standard_attn, cmap='plasma', aspect='auto')
                    ax.set_title(f'Layer {i} - Standard Attention')
                    plt.colorbar(im, ax=ax, shrink=0.6)
                
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
            
            # Hide unused subplots
            for i in range(sample_layers, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Attention Pattern Comparison (Step {step})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = f'attention_patterns_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved attention patterns plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot attention patterns: {e}")
    
    def _plot_frequency_specific_attention(self, step: int, model):
        """Plot attention maps split by frequency bands (low/mid/high)"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Check if model has wave embedding
            if hasattr(model, 'module'):
                model = model.module
            
            # Handle both WaveGPT and PureWaveGPT
            if hasattr(model, 'embedding') and hasattr(model.embedding, 'base_freqs'):
                freqs = model.embedding.base_freqs.detach().cpu().numpy()
            elif hasattr(model, 'wave_excitation') and hasattr(model.wave_excitation, 'base_freqs'):
                freqs = model.wave_excitation.base_freqs.detach().cpu().numpy()
            else:
                print("Warning: Model does not have wave frequencies, skipping frequency-specific attention")
                return
            
            # Define frequency bands
            freq_flat = freqs.flatten()
            low_thresh = np.percentile(freq_flat, 33)
            high_thresh = np.percentile(freq_flat, 67)
            
            # Create frequency masks
            low_freq_mask = freqs <= low_thresh
            mid_freq_mask = (freqs > low_thresh) & (freqs <= high_thresh)
            high_freq_mask = freqs > high_thresh
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            # Plot frequency distributions for each band
            axes[0, 0].hist(freqs[low_freq_mask].flatten(), bins=30, alpha=0.7, color='blue', label='Low Freq')
            axes[0, 0].set_title('Low Frequency Distribution')
            axes[0, 0].set_xlabel('Frequency (Hz)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].hist(freqs[mid_freq_mask].flatten(), bins=30, alpha=0.7, color='green', label='Mid Freq')
            axes[0, 1].set_title('Mid Frequency Distribution')
            axes[0, 1].set_xlabel('Frequency (Hz)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[0, 2].hist(freqs[high_freq_mask].flatten(), bins=30, alpha=0.7, color='red', label='High Freq')
            axes[0, 2].set_title('High Frequency Distribution')
            axes[0, 2].set_xlabel('Frequency (Hz)')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Plot token frequency profiles
            n_tokens = min(100, freqs.shape[0])
            token_ids = np.arange(n_tokens)
            
            # Average frequency per token across waves
            avg_freqs = freqs[:n_tokens].mean(axis=1)
            
            axes[1, 0].scatter(token_ids[avg_freqs <= low_thresh], avg_freqs[avg_freqs <= low_thresh], 
                              c='blue', alpha=0.6, s=20, label='Low Freq Tokens')
            axes[1, 0].set_title('Low Frequency Tokens')
            axes[1, 0].set_xlabel('Token ID')
            axes[1, 0].set_ylabel('Average Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            mid_mask = (avg_freqs > low_thresh) & (avg_freqs <= high_thresh)
            axes[1, 1].scatter(token_ids[mid_mask], avg_freqs[mid_mask], 
                              c='green', alpha=0.6, s=20, label='Mid Freq Tokens')
            axes[1, 1].set_title('Mid Frequency Tokens')
            axes[1, 1].set_xlabel('Token ID')
            axes[1, 1].set_ylabel('Average Frequency')
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].scatter(token_ids[avg_freqs > high_thresh], avg_freqs[avg_freqs > high_thresh], 
                              c='red', alpha=0.6, s=20, label='High Freq Tokens')
            axes[1, 2].set_title('High Frequency Tokens')
            axes[1, 2].set_xlabel('Token ID')
            axes[1, 2].set_ylabel('Average Frequency')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.suptitle(f'Frequency-Specific Analysis (Step {step})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = f'frequency_specific_attention_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved frequency-specific attention plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot frequency-specific attention: {e}")
    
    def _plot_phase_coherence(self, step: int, model):
        """Plot phase coherence analysis between attending tokens"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Check if model has wave embedding
            if hasattr(model, 'module'):
                model = model.module
            
            # Handle both WaveGPT and PureWaveGPT
            if hasattr(model, 'embedding') and hasattr(model.embedding, 'phases'):
                phases = model.embedding.phases.detach().cpu().numpy()
            elif hasattr(model, 'wave_excitation') and hasattr(model.wave_excitation, 'phases'):
                phases = model.wave_excitation.phases.detach().cpu().numpy()
            else:
                print("Warning: Model does not have wave phases, skipping phase coherence")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Plot 1: Phase difference matrix (first 50 tokens)
            n_tokens = min(50, phases.shape[0])
            phase_diffs = np.zeros((n_tokens, n_tokens))
            
            for i in range(n_tokens):
                for j in range(n_tokens):
                    # Compute phase difference for first wave component
                    diff = np.abs(phases[i, 0] - phases[j, 0])
                    # Wrap to [0, π] (minimum phase difference)
                    diff = min(diff, 2*np.pi - diff)
                    phase_diffs[i, j] = diff
            
            im = axes[0, 0].imshow(phase_diffs, cmap='viridis', aspect='auto')
            axes[0, 0].set_title('Phase Difference Matrix')
            axes[0, 0].set_xlabel('Token J')
            axes[0, 0].set_ylabel('Token I')
            plt.colorbar(im, ax=axes[0, 0], label='Phase Diff (rad)')
            
            # Plot 2: Phase coherence histogram
            coherence_scores = np.cos(phase_diffs)  # High coherence = phases aligned
            axes[0, 1].hist(coherence_scores.flatten(), bins=50, alpha=0.7, color='cyan', edgecolor='white')
            axes[0, 1].set_title('Phase Coherence Distribution')
            axes[0, 1].set_xlabel('Coherence Score (cos(Δφ))')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axvline(coherence_scores.mean(), color='red', linestyle='--', 
                              label=f'Mean: {coherence_scores.mean():.3f}')
            axes[0, 1].legend()
            
            # Plot 3: Phase evolution over tokens (first few wave components)
            n_waves = min(4, phases.shape[1])
            for w in range(n_waves):
                axes[1, 0].plot(phases[:n_tokens, w], alpha=0.7, label=f'Wave {w}')
            axes[1, 0].set_title('Phase Evolution Across Tokens')
            axes[1, 0].set_xlabel('Token ID')
            axes[1, 0].set_ylabel('Phase (radians)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Phase synchronization strength
            # Compute order parameter (measure of phase synchronization)
            sync_strength = []
            for w in range(min(8, phases.shape[1])):
                # Complex order parameter
                complex_phases = np.exp(1j * phases[:n_tokens, w])
                order_param = np.abs(complex_phases.mean())
                sync_strength.append(order_param)
            
            axes[1, 1].bar(range(len(sync_strength)), sync_strength, alpha=0.7, color='orange')
            axes[1, 1].set_title('Phase Synchronization Strength')
            axes[1, 1].set_xlabel('Wave Component')
            axes[1, 1].set_ylabel('Order Parameter')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Phase Coherence Analysis (Step {step})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = f'phase_coherence_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved phase coherence plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot phase coherence: {e}")
    
    def _plot_gradient_diagnostics(self, step: int, model):
        """Plot gradient flow diagnostics for wave parameters"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Check if model has wave embedding
            if hasattr(model, 'module'):
                model = model.module
            
            # Handle both WaveGPT and PureWaveGPT
            if hasattr(model, 'embedding'):
                embedding = model.embedding
            elif hasattr(model, 'wave_excitation'):
                embedding = model.wave_excitation
            else:
                print("Warning: Model does not have embedding or wave_excitation, skipping gradient diagnostics")
                return
            
            # Collect gradient norms for wave parameters
            grad_info = {}
            
            if hasattr(embedding, 'phases') and embedding.phases.grad is not None:
                grad_info['phases'] = embedding.phases.grad.norm().item()
            
            if hasattr(embedding, 'base_freqs') and embedding.base_freqs.grad is not None:
                grad_info['base_freqs'] = embedding.base_freqs.grad.norm().item()
            
            # Handle both WavePacketEmbedding (harmonic_amps) and PureWaveExcitation (amplitudes)
            if hasattr(embedding, 'harmonic_amps') and embedding.harmonic_amps.grad is not None:
                grad_info['harmonic_amps'] = embedding.harmonic_amps.grad.norm().item()
            elif hasattr(embedding, 'amplitudes') and embedding.amplitudes.grad is not None:
                grad_info['amplitudes'] = embedding.amplitudes.grad.norm().item()
            
            # Also check other model parameters
            total_grad_norm = 0
            param_count = 0
            layer_grad_norms = []
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    param_count += 1
                    
                    if 'layers' in name:
                        layer_idx = int(name.split('.')[1]) if name.split('.')[1].isdigit() else 0
                        if len(layer_grad_norms) <= layer_idx:
                            layer_grad_norms.extend([0] * (layer_idx + 1 - len(layer_grad_norms)))
                        layer_grad_norms[layer_idx] += grad_norm
            
            total_grad_norm = np.sqrt(total_grad_norm)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Plot 1: Wave parameter gradient norms
            if grad_info:
                names = list(grad_info.keys())
                values = list(grad_info.values())
                
                axes[0, 0].bar(names, values, alpha=0.7, color=['cyan', 'magenta', 'yellow'][:len(names)])
                axes[0, 0].set_title('Wave Parameter Gradient Norms')
                axes[0, 0].set_ylabel('Gradient Norm')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_yscale('log')
            else:
                axes[0, 0].text(0.5, 0.5, 'No wave gradients available', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Wave Parameter Gradient Norms')
            
            # Plot 2: Layer-wise gradient norms
            if layer_grad_norms:
                axes[0, 1].bar(range(len(layer_grad_norms)), layer_grad_norms, alpha=0.7, color='orange')
                axes[0, 1].set_title('Layer-wise Gradient Norms')
                axes[0, 1].set_xlabel('Layer Index')
                axes[0, 1].set_ylabel('Gradient Norm')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'No layer gradients available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Layer-wise Gradient Norms')
            
            # Plot 3: Gradient norm distribution (if we have wave parameters)
            if hasattr(embedding, 'phases') and embedding.phases.grad is not None:
                phase_grads = embedding.phases.grad.detach().cpu().numpy().flatten()
                axes[1, 0].hist(np.log10(np.abs(phase_grads) + 1e-10), bins=50, alpha=0.7, color='cyan')
                axes[1, 0].set_title('Phase Gradient Distribution')
                axes[1, 0].set_xlabel('log10(|gradient|)')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No phase gradients available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Phase Gradient Distribution')
            
            # Plot 4: Total gradient norm over time (if we track it)
            # For now, just show current values
            current_metrics = {
                'Total Grad Norm': total_grad_norm,
                'Param Count': param_count,
                'Avg Grad Norm': total_grad_norm / max(param_count, 1)
            }
            
            metric_names = list(current_metrics.keys())
            metric_values = list(current_metrics.values())
            
            axes[1, 1].bar(metric_names, metric_values, alpha=0.7, color='green')
            axes[1, 1].set_title('Gradient Summary Statistics')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
            
            plt.suptitle(f'Gradient Flow Diagnostics (Step {step})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = f'gradient_diagnostics_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved gradient diagnostics plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot gradient diagnostics: {e}")
    
    def _plot_wave_space_similarity(self, step: int, model):
        """Plot token clustering in amplitude/phase/frequency space"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Check if model has wave embedding
            if hasattr(model, 'module'):
                model = model.module
            
            # Handle both WaveGPT and PureWaveGPT
            if hasattr(model, 'embedding'):
                embedding = model.embedding
            elif hasattr(model, 'wave_excitation'):
                embedding = model.wave_excitation
            else:
                print("Warning: Model does not have embedding or wave_excitation, skipping wave space similarity")
                return
            
            # Collect wave parameters
            has_freqs = hasattr(embedding, 'base_freqs')
            has_phases = hasattr(embedding, 'phases')
            # Handle both WavePacketEmbedding (harmonic_amps) and PureWaveExcitation (amplitudes)
            has_harmonics = hasattr(embedding, 'harmonic_amps') or hasattr(embedding, 'amplitudes')
            
            if not (has_freqs or has_phases or has_harmonics):
                print("Warning: No wave parameters found, skipping wave space similarity")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            n_tokens = min(100, embedding.base_freqs.shape[0] if has_freqs else 50)
            
            # Plot 1: Frequency vs Phase scatter (first wave component)
            if has_freqs and has_phases:
                freqs = embedding.base_freqs.detach().cpu().numpy()[:n_tokens, 0]
                phases = embedding.phases.detach().cpu().numpy()[:n_tokens, 0]
                
                scatter = axes[0, 0].scatter(freqs, phases, c=np.arange(n_tokens), 
                                           cmap='viridis', alpha=0.6, s=30)
                axes[0, 0].set_xlabel('Frequency (Hz)')
                axes[0, 0].set_ylabel('Phase (radians)')
                axes[0, 0].set_title('Frequency vs Phase Space')
                axes[0, 0].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[0, 0], label='Token ID')
            else:
                axes[0, 0].text(0.5, 0.5, 'Freq/Phase not available', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Frequency vs Phase Space')
            
            # Plot 2: Amplitude distribution (if harmonics available)
            if has_harmonics:
                # Handle both WavePacketEmbedding (harmonic_amps) and PureWaveExcitation (amplitudes)
                if hasattr(embedding, 'harmonic_amps'):
                    harm_amps = embedding.harmonic_amps.detach().cpu().numpy()[:n_tokens]
                elif hasattr(embedding, 'amplitudes'):
                    harm_amps = embedding.amplitudes.detach().cpu().numpy()[:n_tokens]
                else:
                    harm_amps = np.zeros((n_tokens, 1, 1))
                
                # Use first harmonic of first wave
                amps = harm_amps[:, 0, 0] if harm_amps.shape[2] > 0 else np.zeros(n_tokens)
                
                axes[0, 1].hist(amps, bins=30, alpha=0.7, color='orange', edgecolor='white')
                axes[0, 1].set_xlabel('Amplitude')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].set_title('Amplitude Distribution')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'Harmonics not available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Amplitude Distribution')
            
            # Plot 3: Token similarity matrix in wave space
            if has_freqs and has_phases:
                # Compute pairwise similarities in wave space
                n_sample = min(50, n_tokens)
                similarity_matrix = np.zeros((n_sample, n_sample))
                
                for i in range(n_sample):
                    for j in range(n_sample):
                        # Simple similarity based on frequency and phase differences
                        freq_diff = np.abs(freqs[i] - freqs[j])
                        phase_diff = np.abs(phases[i] - phases[j])
                        phase_diff = min(phase_diff, 2*np.pi - phase_diff)  # Wrap phase
                        
                        # Normalize and combine
                        freq_sim = np.exp(-freq_diff / (freqs.std() + 1e-8))
                        phase_sim = np.exp(-phase_diff / (phases.std() + 1e-8))
                        similarity_matrix[i, j] = (freq_sim + phase_sim) / 2
                
                im = axes[0, 2].imshow(similarity_matrix, cmap='viridis', aspect='auto')
                axes[0, 2].set_xlabel('Token J')
                axes[0, 2].set_ylabel('Token I')
                axes[0, 2].set_title('Wave Space Similarity Matrix')
                plt.colorbar(im, ax=axes[0, 2], label='Similarity')
            else:
                axes[0, 2].text(0.5, 0.5, 'Cannot compute similarity', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Wave Space Similarity Matrix')
            
            # Plot 4: Multi-wave frequency spectrum per token
            if has_freqs:
                freqs_all = embedding.base_freqs.detach().cpu().numpy()
                n_waves = min(8, freqs_all.shape[1])
                
                # Show spectrum for a few sample tokens
                sample_tokens = [0, n_tokens//4, n_tokens//2, 3*n_tokens//4, n_tokens-1]
                sample_tokens = [t for t in sample_tokens if t < n_tokens]
                
                for idx, tok_id in enumerate(sample_tokens):
                    axes[1, 0].plot(freqs_all[tok_id, :n_waves], alpha=0.7, 
                                   marker='o', label=f'Token {tok_id}')
                
                axes[1, 0].set_xlabel('Wave Component')
                axes[1, 0].set_ylabel('Frequency (Hz)')
                axes[1, 0].set_title('Multi-Wave Frequency Spectrum')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Frequencies not available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Multi-Wave Frequency Spectrum')
            
            # Plot 5: Phase coherence across waves
            if has_phases:
                phases_all = embedding.phases.detach().cpu().numpy()
                n_waves = min(8, phases_all.shape[1])
                
                # Compute phase coherence between wave components
                coherence_matrix = np.zeros((n_waves, n_waves))
                for i in range(n_waves):
                    for j in range(n_waves):
                        phase_diff = phases_all[:n_tokens, i] - phases_all[:n_tokens, j]
                        coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
                        coherence_matrix[i, j] = coherence
                
                im = axes[1, 1].imshow(coherence_matrix, cmap='plasma', aspect='auto')
                axes[1, 1].set_xlabel('Wave Component J')
                axes[1, 1].set_ylabel('Wave Component I')
                axes[1, 1].set_title('Inter-Wave Phase Coherence')
                plt.colorbar(im, ax=axes[1, 1], label='Coherence')
            else:
                axes[1, 1].text(0.5, 0.5, 'Phases not available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Inter-Wave Phase Coherence')
            
            # Plot 6: Wave parameter evolution (show ranges)
            param_ranges = {}
            if has_freqs:
                freqs_all = embedding.base_freqs.detach().cpu().numpy()
                param_ranges['Freq Min'] = freqs_all.min()
                param_ranges['Freq Max'] = freqs_all.max()
                param_ranges['Freq Mean'] = freqs_all.mean()
                param_ranges['Freq Std'] = freqs_all.std()
            
            if has_phases:
                phases_all = embedding.phases.detach().cpu().numpy()
                param_ranges['Phase Min'] = phases_all.min()
                param_ranges['Phase Max'] = phases_all.max()
                param_ranges['Phase Mean'] = phases_all.mean()
                param_ranges['Phase Std'] = phases_all.std()
            
            if param_ranges:
                names = list(param_ranges.keys())
                values = list(param_ranges.values())
                
                axes[1, 2].bar(names, values, alpha=0.7, color='green')
                axes[1, 2].set_title('Wave Parameter Statistics')
                axes[1, 2].set_ylabel('Value')
                axes[1, 2].tick_params(axis='x', rotation=45)
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'No parameters available', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Wave Parameter Statistics')
            
            plt.suptitle(f'Wave Space Similarity Analysis (Step {step})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = f'wave_space_similarity_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved wave space similarity plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot wave space similarity: {e}")
    
    def _plot_interference_strength(self, step: int, model):
        """Plot distribution of interference scores before softmax"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Check if model has interference attention layers
            if hasattr(model, 'module'):
                model = model.module
            
            # Handle both WaveGPT (layers) and PureWaveGPT (wave_layers)
            if hasattr(model, 'layers'):
                layers = model.layers
            elif hasattr(model, 'wave_layers'):
                layers = model.wave_layers
            else:
                print("Warning: Model does not have 'layers' or 'wave_layers' attribute, skipping interference strength")
                return
            
            # Find interference attention layers
            interference_layers = []
            for layer in layers:
                if hasattr(layer, 'attn') and hasattr(layer.attn, 'interference_strength'):
                    interference_layers.append(layer.attn)
                elif hasattr(layer, 'wave_attention') and hasattr(layer.wave_attention, 'interference_strength'):  # PureWaveGPT
                    interference_layers.append(layer.wave_attention)
            
            if not interference_layers:
                print("Warning: No interference attention layers found, skipping interference strength")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            
            # Analyze each interference layer
            for i, attn_layer in enumerate(interference_layers[:6]):
                ax = axes[i]
                
                # Get interference strength parameter
                interference_strength = attn_layer.interference_strength.item()
                
                # Check if we have stored interference scores
                if hasattr(attn_layer, 'last_interference_scores'):
                    scores = attn_layer.last_interference_scores.detach().cpu().numpy()
                    
                    # Flatten scores for histogram
                    scores_flat = scores.flatten()
                    
                    # Plot histogram of interference scores
                    ax.hist(scores_flat, bins=50, alpha=0.7, color='cyan', edgecolor='white')
                    ax.set_xlabel('Interference Score')
                    ax.set_ylabel('Count')
                    ax.set_title(f'Layer {i} - Interference Scores\n(Strength: {interference_strength:.3f})')
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    mean_score = scores_flat.mean()
                    std_score = scores_flat.std()
                    ax.axvline(mean_score, color='red', linestyle='--', 
                              label=f'Mean: {mean_score:.3f}')
                    ax.axvline(mean_score + std_score, color='orange', linestyle=':', alpha=0.7,
                              label=f'±1σ: {std_score:.3f}')
                    ax.axvline(mean_score - std_score, color='orange', linestyle=':', alpha=0.7)
                    ax.legend()
                    
                else:
                    # If no stored scores, show interference strength parameter
                    ax.bar(['Interference\nStrength'], [interference_strength], 
                          alpha=0.7, color='magenta')
                    ax.set_title(f'Layer {i} - Interference Strength')
                    ax.set_ylabel('Parameter Value')
                    ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(interference_layers), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Interference Strength Analysis (Step {step})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = f'interference_strength_step_{step}.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved interference strength plot: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to plot interference strength: {e}")
    
    def generate_comparison_plots(self,
                                 experiments: List[Dict]):
        """
        Generate comparison plots across multiple experiments.
        
        Args:
            experiments: List of experiment dictionaries containing:
                - name: Experiment name
                - loss_history: List of loss values
                - metrics: Dictionary of metric histories
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not experiments:
                print("Warning: No experiments provided for comparison")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            # Plot 1: Loss comparison
            ax = axes[0]
            for exp in experiments:
                name = exp.get('name', 'Unknown')
                loss_history = exp.get('loss_history', [])
                steps = list(range(len(loss_history)))
                ax.plot(steps, loss_history, linewidth=2, label=name, alpha=0.8)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Learning rate comparison (if available)
            ax = axes[1]
            for exp in experiments:
                name = exp.get('name', 'Unknown')
                metrics = exp.get('metrics', {})
                if 'learning_rate' in metrics:
                    lr_history = metrics['learning_rate']
                    steps = list(range(len(lr_history)))
                    ax.plot(steps, lr_history, linewidth=2, label=name, alpha=0.8)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # Plot 3: Perplexity comparison (if available)
            ax = axes[2]
            for exp in experiments:
                name = exp.get('name', 'Unknown')
                metrics = exp.get('metrics', {})
                if 'perplexity' in metrics:
                    ppl_history = metrics['perplexity']
                    steps = list(range(len(ppl_history)))
                    ax.plot(steps, ppl_history, linewidth=2, label=name, alpha=0.8)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Perplexity')
            ax.set_title('Perplexity Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Final metrics bar chart
            ax = axes[3]
            exp_names = [exp.get('name', 'Unknown') for exp in experiments]
            final_losses = [exp.get('loss_history', [0])[-1] if exp.get('loss_history') else 0 
                          for exp in experiments]
            
            x = np.arange(len(exp_names))
            ax.bar(x, final_losses, color='cyan', alpha=0.7, edgecolor='white')
            ax.set_xticks(x)
            ax.set_xticklabels(exp_names, rotation=45, ha='right')
            ax.set_ylabel('Final Loss')
            ax.set_title('Final Loss Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Experiment Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = 'experiment_comparison.png'
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved comparison plots: {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to generate comparison plots: {e}")
            import traceback
            traceback.print_exc()


# ==========================================
# ConfigTracker
# ==========================================

class ConfigTracker:
    """
    Tracks and saves experiment configuration and metadata.
    
    Features:
    - JSON format for easy parsing and comparison
    - Automatic git hash capture for reproducibility
    - Hardware info includes GPU model, CUDA version, CPU count
    - Results include best validation loss, final perplexity, training time
    
    Example:
        >>> tracker = ConfigTracker(experiment_dir="experiments/exp_001")
        >>> 
        >>> # At experiment start
        >>> tracker.save_config(config, model, dataset_info)
        >>> 
        >>> # At experiment end
        >>> tracker.save_results(final_metrics, best_checkpoint, generation_samples)
        >>> 
        >>> # Load for analysis
        >>> config = tracker.load_config()
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize ConfigTracker.
        
        Args:
            experiment_dir: Root directory for experiment
        """
        self.experiment_dir = Path(experiment_dir)
        self.config_file = self.experiment_dir / 'config.json'
        self.results_file = self.experiment_dir / 'results.json'
        
        # Create experiment directory if it doesn't exist
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self,
                   config: Dict,
                   model: Optional[nn.Module] = None,
                   dataset_info: Optional[Dict] = None):
        """
        Save experiment configuration.
        
        Includes:
        - All hyperparameters
        - Model architecture details
        - Dataset information
        - Git commit hash
        - Timestamp
        - Hardware info (GPU, CPU, memory)
        
        Args:
            config: Configuration dictionary
            model: Model to extract architecture info from (optional)
            dataset_info: Dataset information dictionary (optional)
        """
        full_config = {
            'experiment_id': self.experiment_dir.name,
            'timestamp': datetime.now().isoformat(),
            'git_hash': get_git_hash(),
            'config': config,
            'hardware': self._get_hardware_info()
        }
        
        # Add model info if provided
        if model is not None:
            full_config['model'] = self._get_model_info(model)
        
        # Add dataset info if provided
        if dataset_info is not None:
            full_config['dataset'] = dataset_info
        
        # Save to file
        with open(self.config_file, 'w') as f:
            json.dump(full_config, f, indent=2)
        
        print(f"✓ Saved config: {self.config_file}")
    
    def load_config(self) -> Dict:
        """
        Load experiment configuration.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_file.exists():
            return {}
        
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def save_results(self,
                    final_metrics: Dict,
                    best_checkpoint: Optional[str] = None,
                    generation_samples: Optional[List[str]] = None):
        """
        Save final experiment results.
        
        Args:
            final_metrics: Dictionary of final metrics
            best_checkpoint: Path to best checkpoint (optional)
            generation_samples: List of generated text samples (optional)
        """
        results = {
            'experiment_id': self.experiment_dir.name,
            'timestamp': datetime.now().isoformat(),
            'final_metrics': final_metrics
        }
        
        if best_checkpoint is not None:
            results['best_checkpoint'] = best_checkpoint
        
        if generation_samples is not None:
            results['generation_samples'] = generation_samples
        
        # Save to file
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Saved results: {self.results_file}")
    
    def _get_hardware_info(self) -> Dict:
        """Collect hardware information"""
        import platform
        
        hardware_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count()
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            hardware_info['gpu_available'] = True
            hardware_info['gpu_count'] = torch.cuda.device_count()
            hardware_info['gpu_model'] = torch.cuda.get_device_name(0)
            hardware_info['cuda_version'] = torch.version.cuda
        else:
            hardware_info['gpu_available'] = False
        
        return hardware_info
    
    def _get_model_info(self, model: nn.Module) -> Dict:
        """Extract model architecture information"""
        # Handle DataParallel
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        model_info = {
            'type': model.__class__.__name__,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Try to extract common attributes
        for attr in ['d_model', 'num_layers', 'num_heads', 'num_waves', 'num_harmonics', 
                     'vocab_size', 'block_size', 'dropout']:
            if hasattr(model, attr):
                model_info[attr] = getattr(model, attr)
        
        return model_info


# ==========================================
# ResultsParser
# ==========================================

class ResultsParser:
    """
    Parses experiment results from text files into structured JSON format.
    
    Extracts:
    - Experiment configurations (model, optimizer, loss, hyperparameters)
    - Training metrics (step, loss, validation loss, wave ratio)
    - Final results (best validation loss, perplexity, speed)
    - Generation samples
    """
    
    def __init__(self, results_file: str):
        """
        Initialize ResultsParser.
        
        Args:
            results_file: Path to results.txt file
        """
        self.results_file = results_file
        self.experiments = []
        
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse the results file and extract all experiments.
        
        Returns:
            List of experiment dictionaries with structured data
        """
        with open(self.results_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by experiment sections
        # Look for experiment headers like "Standard Transformer (GPT-2)" or "Full Physics (RGD + QFE)"
        experiments = []
        current_experiment = None
        current_metrics = []
        
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Detect experiment header (boxed section)
            if line.startswith('╭─') and i + 1 < len(lines):
                # Next line contains experiment name
                exp_name_line = lines[i + 1]
                if '│' in exp_name_line:
                    exp_name = exp_name_line.strip('│ ').strip()
                    
                    # Skip if this is a generation output box (contains "Output")
                    if 'Output' in exp_name or 'The theory' in exp_name:
                        i += 1
                        continue
                    
                    # Skip if this is just a header box without actual experiment name
                    if '🔬' in exp_name and 'ABLATION' in exp_name:
                        i += 1
                        continue
                    
                    # Save previous experiment if exists
                    if current_experiment is not None:
                        current_experiment['training_metrics'] = current_metrics
                        experiments.append(current_experiment)
                    
                    # Start new experiment
                    current_experiment = {
                        'name': exp_name,
                        'config': {},
                        'training_metrics': [],
                        'final_results': {},
                        'generation_sample': ''
                    }
                    current_metrics = []
            
            # Extract configuration parameters
            elif current_experiment is not None:
                # Parameters line (only capture if we don't already have it)
                if line.startswith('📊 Parameters:'):
                    if 'parameters' not in current_experiment['config']:
                        params_str = line.split(':')[1].strip()
                        # Extract number like "52,892,160 (52.89M)"
                        params_num = params_str.split('(')[0].strip().replace(',', '')
                        current_experiment['config']['parameters'] = int(params_num)
                
                # Steps line
                elif line.startswith('🔄 Steps:'):
                    steps = int(line.split(':')[1].strip())
                    current_experiment['config']['steps'] = steps
                
                # Optimizer line
                elif line.startswith('⚙️  Optimizer:') or line.startswith('⚡ Optimizer:'):
                    optimizer = line.split(':')[1].strip()
                    current_experiment['config']['optimizer'] = optimizer
                
                # Loss line
                elif line.startswith('📉 Loss:') or line.startswith('🌌 Loss:'):
                    loss = line.split(':')[1].strip()
                    current_experiment['config']['loss'] = loss
                
                # Learning rate line
                elif line.startswith('📈 LR:'):
                    lr = line.split(':')[1].strip()
                    current_experiment['config']['learning_rate'] = lr
                
                # Training step metrics
                elif line.startswith('Step ') and '|' in line:
                    # Parse line like: "Step   250 | Train(CE): 7.0259 | Val: 7.0833 | AvgTrain: 8.3495 | R: 0.500"
                    parts = line.split('|')
                    if len(parts) >= 4:
                        try:
                            step_str = parts[0].strip().split()[1]
                            step = int(step_str)
                            
                            train_loss_str = parts[1].strip().split(':')[1].strip()
                            train_loss = float(train_loss_str)
                            
                            val_loss_str = parts[2].strip().split(':')[1].strip()
                            val_loss = float(val_loss_str)
                            
                            avg_train_str = parts[3].strip().split(':')[1].strip()
                            avg_train = float(avg_train_str)
                            
                            wave_ratio = None
                            if len(parts) >= 5 and 'R:' in parts[4]:
                                wave_ratio_str = parts[4].strip().split(':')[1].strip()
                                wave_ratio = float(wave_ratio_str)
                            
                            metric = {
                                'step': step,
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                                'avg_train_loss': avg_train,
                            }
                            if wave_ratio is not None:
                                metric['wave_ratio'] = wave_ratio
                            
                            current_metrics.append(metric)
                        except (ValueError, IndexError):
                            pass
                
                # Best model restoration line
                elif line.startswith('♻️  Restoring best model'):
                    # Extract best validation loss like "(Val: 4.4382)"
                    if '(' in line and 'Val:' in line:
                        val_str = line.split('Val:')[1].split(')')[0].strip()
                        try:
                            current_experiment['final_results']['best_val_loss'] = float(val_str)
                        except ValueError:
                            pass
                
                # Generation sample section
                elif 'Output' in line and '─' in line:
                    # Start of generation sample box
                    sample_lines = []
                    i += 1
                    while i < len(lines) and not lines[i].startswith('╰'):
                        if lines[i].startswith('│'):
                            sample_lines.append(lines[i].strip('│ ').strip())
                        i += 1
                    current_experiment['generation_sample'] = ' '.join(sample_lines)
            
            i += 1
        
        # Save last experiment
        if current_experiment is not None:
            current_experiment['training_metrics'] = current_metrics
            experiments.append(current_experiment)
        
        # Parse ablation study table if present
        ablation_results = self._parse_ablation_table(content)
        if ablation_results:
            # Match ablation results to experiments by name
            for exp in experiments:
                for ablation in ablation_results:
                    if ablation['name'] in exp['name']:
                        exp['final_results'].update({
                            'val_loss': ablation['val_loss'],
                            'perplexity': ablation['perplexity'],
                            'speed_tokens_per_sec': ablation['speed']
                        })
        
        self.experiments = experiments
        return experiments
    
    def _parse_ablation_table(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse the ablation study results table.
        
        Returns:
            List of ablation results
        """
        results = []
        lines = content.split('\n')
        
        # Find the ablation table
        in_table = False
        for line in lines:
            if '🔬 Ablation Study Results' in line:
                in_table = True
                continue
            
            if in_table:
                # Skip table borders and headers
                if line.startswith('┏') or line.startswith('┃') or line.startswith('┡') or line.startswith('└'):
                    continue
                
                # Parse data rows starting with │
                if line.startswith('│'):
                    parts = [p.strip() for p in line.split('│') if p.strip()]
                    if len(parts) >= 5:
                        # Skip header row
                        if 'Experiment' in parts[0] or 'RGD' in parts[1]:
                            continue
                        
                        try:
                            name = parts[0]
                            val_loss = float(parts[3])
                            perplexity = float(parts[4])
                            speed = float(parts[5].replace(',', ''))
                            
                            results.append({
                                'name': name,
                                'val_loss': val_loss,
                                'perplexity': perplexity,
                                'speed': speed
                            })
                        except (ValueError, IndexError):
                            pass
                
                # End of table
                if line.startswith('└'):
                    break
        
        return results
    
    def save_to_json(self, output_file: str):
        """
        Save parsed results to JSON file.
        
        Args:
            output_file: Path to output JSON file
        """
        if not self.experiments:
            self.parse()
        
        output_data = {
            'experiments': self.experiments,
            'parsed_at': datetime.now().isoformat(),
            'source_file': self.results_file
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def get_experiment_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment data by name.
        
        Args:
            name: Experiment name (can be partial match)
            
        Returns:
            Experiment dictionary or None if not found
        """
        if not self.experiments:
            self.parse()
        
        for exp in self.experiments:
            if name.lower() in exp['name'].lower():
                return exp
        return None
    
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """
        Get all parsed experiments.
        
        Returns:
            List of experiment dictionaries
        """
        if not self.experiments:
            self.parse()
        return self.experiments


# ==========================================
# ResultsAggregator
# ==========================================

class ResultsAggregator:
    """
    Aggregates and compares results from multiple experiments.
    
    Features:
    - Load multiple experiment results
    - Generate comparison tables
    - Statistical significance testing
    - Ablation study analysis
    - Summary visualizations
    """
    
    def __init__(self):
        """Initialize ResultsAggregator."""
        self.experiments = []
        
    def load_from_json(self, json_file: str):
        """
        Load experiments from JSON file.
        
        Args:
            json_file: Path to JSON file with parsed experiments
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
            self.experiments.extend(data.get('experiments', []))
    
    def load_from_parser(self, parser: ResultsParser):
        """
        Load experiments from ResultsParser.
        
        Args:
            parser: ResultsParser instance with parsed experiments
        """
        self.experiments.extend(parser.get_all_experiments())
    
    def add_experiment(self, experiment: Dict[str, Any]):
        """
        Add a single experiment.
        
        Args:
            experiment: Experiment dictionary
        """
        self.experiments.append(experiment)
    
    def generate_comparison_table(self) -> str:
        """
        Generate a comparison table of all experiments.
        
        Returns:
            Formatted table string
        """
        if not self.experiments:
            return "No experiments loaded."
        
        # Build table
        rows = []
        headers = ['Experiment', 'Parameters', 'Val Loss', 'Perplexity', 'Speed (tok/s)']
        
        for exp in self.experiments:
            name = exp['name']
            params = exp['config'].get('parameters', 'N/A')
            if isinstance(params, int):
                params = f"{params:,}"
            
            val_loss = exp['final_results'].get('val_loss', 
                       exp['final_results'].get('best_val_loss', 'N/A'))
            perplexity = exp['final_results'].get('perplexity', 'N/A')
            speed = exp['final_results'].get('speed_tokens_per_sec', 'N/A')
            
            if isinstance(speed, (int, float)):
                speed = f"{speed:,.0f}"
            
            rows.append([name, params, val_loss, perplexity, speed])
        
        # Format table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        def format_row(row, widths):
            return '| ' + ' | '.join(str(item).ljust(width) for item, width in zip(row, widths)) + ' |'
        
        separator = '|-' + '-|-'.join('-' * width for width in col_widths) + '-|'
        
        table = []
        table.append(format_row(headers, col_widths))
        table.append(separator)
        for row in rows:
            table.append(format_row(row, col_widths))
        
        return '\n'.join(table)
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics across all experiments.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.experiments:
            return {}
        
        val_losses = []
        perplexities = []
        speeds = []
        
        for exp in self.experiments:
            val_loss = exp['final_results'].get('val_loss',
                       exp['final_results'].get('best_val_loss'))
            if val_loss is not None and isinstance(val_loss, (int, float)):
                val_losses.append(val_loss)
            
            perplexity = exp['final_results'].get('perplexity')
            if perplexity is not None and isinstance(perplexity, (int, float)):
                perplexities.append(perplexity)
            
            speed = exp['final_results'].get('speed_tokens_per_sec')
            if speed is not None and isinstance(speed, (int, float)):
                speeds.append(speed)
        
        import statistics
        
        summary = {
            'num_experiments': len(self.experiments),
            'val_loss': {
                'mean': statistics.mean(val_losses) if val_losses else None,
                'median': statistics.median(val_losses) if val_losses else None,
                'min': min(val_losses) if val_losses else None,
                'max': max(val_losses) if val_losses else None,
                'stdev': statistics.stdev(val_losses) if len(val_losses) > 1 else None
            },
            'perplexity': {
                'mean': statistics.mean(perplexities) if perplexities else None,
                'median': statistics.median(perplexities) if perplexities else None,
                'min': min(perplexities) if perplexities else None,
                'max': max(perplexities) if perplexities else None,
                'stdev': statistics.stdev(perplexities) if len(perplexities) > 1 else None
            },
            'speed': {
                'mean': statistics.mean(speeds) if speeds else None,
                'median': statistics.median(speeds) if speeds else None,
                'min': min(speeds) if speeds else None,
                'max': max(speeds) if speeds else None,
                'stdev': statistics.stdev(speeds) if len(speeds) > 1 else None
            }
        }
        
        return summary
    
    def compare_experiments(self, exp1_name: str, exp2_name: str) -> Dict[str, Any]:
        """
        Compare two experiments.
        
        Args:
            exp1_name: Name of first experiment (partial match)
            exp2_name: Name of second experiment (partial match)
            
        Returns:
            Comparison dictionary
        """
        exp1 = None
        exp2 = None
        
        for exp in self.experiments:
            if exp1_name.lower() in exp['name'].lower():
                exp1 = exp
            if exp2_name.lower() in exp['name'].lower():
                exp2 = exp
        
        if exp1 is None or exp2 is None:
            return {'error': 'One or both experiments not found'}
        
        comparison = {
            'experiment_1': exp1['name'],
            'experiment_2': exp2['name'],
            'val_loss_diff': None,
            'val_loss_improvement_pct': None,
            'perplexity_diff': None,
            'speed_diff': None
        }
        
        val1 = exp1['final_results'].get('val_loss', exp1['final_results'].get('best_val_loss'))
        val2 = exp2['final_results'].get('val_loss', exp2['final_results'].get('best_val_loss'))
        
        if val1 is not None and val2 is not None:
            comparison['val_loss_diff'] = val2 - val1
            comparison['val_loss_improvement_pct'] = ((val1 - val2) / val1) * 100
        
        perp1 = exp1['final_results'].get('perplexity')
        perp2 = exp2['final_results'].get('perplexity')
        
        if perp1 is not None and perp2 is not None:
            comparison['perplexity_diff'] = perp2 - perp1
        
        speed1 = exp1['final_results'].get('speed_tokens_per_sec')
        speed2 = exp2['final_results'].get('speed_tokens_per_sec')
        
        if speed1 is not None and speed2 is not None:
            comparison['speed_diff'] = speed2 - speed1
        
        return comparison
    
    def generate_ablation_analysis(self) -> Dict[str, Any]:
        """
        Generate ablation study analysis.
        
        Identifies baseline and component contributions.
        
        Returns:
            Ablation analysis dictionary
        """
        if not self.experiments:
            return {}
        
        # Try to identify baseline (standard transformer)
        baseline = None
        for exp in self.experiments:
            if 'standard' in exp['name'].lower() or 'transformer' in exp['name'].lower():
                if 'wave' not in exp['name'].lower() and 'physics' not in exp['name'].lower():
                    baseline = exp
                    break
        
        # Identify component experiments
        rgd_only = None
        qfe_only = None
        full_physics = None
        
        for exp in self.experiments:
            name_lower = exp['name'].lower()
            if 'rgd only' in name_lower:
                rgd_only = exp
            elif 'qfe only' in name_lower:
                qfe_only = exp
            elif 'full physics' in name_lower or ('rgd' in name_lower and 'qfe' in name_lower):
                full_physics = exp
        
        analysis = {
            'baseline': baseline['name'] if baseline else None,
            'components': {}
        }
        
        if baseline:
            baseline_val = baseline['final_results'].get('val_loss', 
                          baseline['final_results'].get('best_val_loss'))
            
            if rgd_only:
                rgd_val = rgd_only['final_results'].get('val_loss',
                         rgd_only['final_results'].get('best_val_loss'))
                if baseline_val and rgd_val:
                    analysis['components']['RGD'] = {
                        'val_loss': rgd_val,
                        'improvement_over_baseline': baseline_val - rgd_val,
                        'improvement_pct': ((baseline_val - rgd_val) / baseline_val) * 100
                    }
            
            if qfe_only:
                qfe_val = qfe_only['final_results'].get('val_loss',
                         qfe_only['final_results'].get('best_val_loss'))
                if baseline_val and qfe_val:
                    analysis['components']['QFE'] = {
                        'val_loss': qfe_val,
                        'improvement_over_baseline': baseline_val - qfe_val,
                        'improvement_pct': ((baseline_val - qfe_val) / baseline_val) * 100
                    }
            
            if full_physics:
                full_val = full_physics['final_results'].get('val_loss',
                          full_physics['final_results'].get('best_val_loss'))
                if baseline_val and full_val:
                    analysis['components']['Full (RGD + QFE)'] = {
                        'val_loss': full_val,
                        'improvement_over_baseline': baseline_val - full_val,
                        'improvement_pct': ((baseline_val - full_val) / baseline_val) * 100
                    }
        
        # Identify top-performing component
        if analysis['components']:
            best_component = max(analysis['components'].items(),
                               key=lambda x: x[1].get('improvement_over_baseline', 0))
            analysis['best_component'] = best_component[0]
            analysis['best_improvement'] = best_component[1].get('improvement_over_baseline')
        
        return analysis
    
    def save_summary(self, output_file: str):
        """
        Save comprehensive summary to JSON file.
        
        Args:
            output_file: Path to output JSON file
        """
        summary = {
            'comparison_table': self.generate_comparison_table(),
            'statistics': self.generate_summary_statistics(),
            'ablation_analysis': self.generate_ablation_analysis(),
            'experiments': self.experiments,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
