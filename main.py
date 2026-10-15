import os
import torch
from datetime import datetime
from src.training.trainer import train_fit
from src.visualization.plotter import plot_results, plot_layer_waves, plot_wave_decomposition
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Configuration
NUM_EPOCHS = 10
NUM_WAVES = 12
MODES = ["Standard", "UserWave", "Poly", "Wavelet", "Factor", "Siren", "GatedWave"]

# Fourier Configuration (for UserWave and GatedWave)
NUM_HARMONICS = 5          # Number of Fourier components per wave (5 = 1×, 2×, 4×, 8×, 16× base freq)
ADAPTIVE_FREQS = True      # If True, harmonic frequencies [1, 2, 4, ...] become learnable
PER_NEURON_COEFFS = False  # If True, each output neuron has its own Fourier coefficients
L1_PENALTY = 0.0           # L1 regularization strength on Fourier coefficients (0.0 = disabled)
WAVE_MODE = "outer_product"  # "outer_product" (2D), "fourier_series" (1D), "gabor" (wavelets)

# Experiment Configurations
EXPERIMENTS = {
    "spectral_baseline": {
        "description": "Spectral 2D Baseline (12 waves, 5 harmonics, adaptive freqs)",
        "num_harmonics": 5,
        "adaptive_freqs": True,
        "per_neuron_coeffs": False,
        "l1_penalty": 0.0,
        "num_waves": 12,
        "wave_mode": "outer_product"
    },
    "spectral_1d": {
        "description": "1D Sine Wave Mode (High Parameter Efficiency)",
        "num_harmonics": 5,
        "adaptive_freqs": True,
        "per_neuron_coeffs": False,
        "l1_penalty": 0.0,
        "num_waves": 12,
        "wave_mode": "fourier_series"
    },
    "gabor_wavelets": {
        "description": "Gabor Wavelets (Spatially Localized Sine Waves)",
        "num_harmonics": 7,
        "adaptive_freqs": True,
        "per_neuron_coeffs": False,
        "l1_penalty": 0.0,
        "num_waves": 32,
        "wave_mode": "gabor"
    },
    "high_freq_rich": {
        "description": "Rich Harmonics (9 octaves for fine details)",
        "num_harmonics": 9,
        "adaptive_freqs": True,
        "per_neuron_coeffs": False,
        "l1_penalty": 0.0,
        "num_waves": 12,
        "wave_mode": "outer_product"
    },
    "compressed_2d": {
        "description": "Ultra Compression (4 waves only)",
        "num_harmonics": 5,
        "adaptive_freqs": True,
        "per_neuron_coeffs": False,
        "l1_penalty": 0.0,
        "num_waves": 4,
        "wave_mode": "outer_product"
    },
    "high_capacity_1d": {
        "description": "High Capacity 1D (64 waves, 7 harmonics)",
        "num_harmonics": 7,
        "adaptive_freqs": True,
        "per_neuron_coeffs": False,
        "l1_penalty": 0.0,
        "num_waves": 64,
        "wave_mode": "fourier_series"
    },
    "per_neuron_coeffs": {
        "description": "Per-Neuron Coefficients (Max Flexibility)",
        "num_harmonics": 5,
        "adaptive_freqs": True,
        "per_neuron_coeffs": True,
        "l1_penalty": 0.001,
        "num_waves": 12,
        "wave_mode": "outer_product"
    },
    "spectral_cnn": {
        "description": "Full Spectral CNN (SpectralConv2d + Linear)",
        "num_harmonics": 5,
        "adaptive_freqs": True,
        "per_neuron_coeffs": False,
        "l1_penalty": 0.0,
        "num_waves": 16,
        "wave_mode": "outer_product"
    }
}

# Select which experiments to run
ACTIVE_EXPERIMENTS = [
    #"spectral_baseline",
    "spectral_1d",
    "gabor_wavelets",
    "high_freq_rich",
    "compressed_2d",
    "high_capacity_1d",
    "per_neuron_coeffs",
    "spectral_cnn"
]

def run_experiment(exp_name, exp_config, device):
    """Run a single experiment configuration."""
    # Get num_waves from config or use default
    n_waves = exp_config.get('num_waves', NUM_WAVES)
    
    console.print()
    console.print(Panel(
        f"[bold yellow]EXPERIMENT: {exp_name.upper()}[/]\n"
        f"[dim]{exp_config['description']}[/]\n\n"
        f"[cyan]Wave Mode:[/] {exp_config.get('wave_mode', WAVE_MODE)}\n"
        f"[cyan]Config:[/] harmonics={exp_config['num_harmonics']}, waves={n_waves}, "
        f"adaptive={exp_config['adaptive_freqs']}, per_neuron={exp_config['per_neuron_coeffs']}, "
        f"L1={exp_config['l1_penalty']}",
        title=f"📊 Experiment {ACTIVE_EXPERIMENTS.index(exp_name) + 1}/{len(ACTIVE_EXPERIMENTS)}",
        border_style="yellow"
    ))
    
    results = {}
    
    # Run all models for complete comparison
    for m in MODES:
        if m in ["UserWave", "GatedWave"]:
            # Spectral layers with Fourier configuration
            results[m] = train_fit(
                m, n_waves, NUM_EPOCHS, device,
                num_harmonics=exp_config['num_harmonics'],
                adaptive_freqs=exp_config['adaptive_freqs'],
                per_neuron_coeffs=exp_config['per_neuron_coeffs'],
                l1_penalty=exp_config['l1_penalty'],
                wave_mode=exp_config.get('wave_mode', WAVE_MODE) # Allow overriding wave_mode per experiment
            )
        else:
            # Other models use default settings
            results[m] = train_fit(m, NUM_WAVES, NUM_EPOCHS, device)
    
    return results

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"./results/Experiments_{timestamp}/"
    os.makedirs(base_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}
    
    console.print(Panel(
        f"[bold green]🌊 SPECTRAL CNN: HOLOGRAPHIC GENERAL INTELLIGENCE[/]\n\n"
        f"[cyan]Running:[/] {len(ACTIVE_EXPERIMENTS)} experiment(s)\n"
        f"[cyan]Epochs:[/] {NUM_EPOCHS}\n"
        f"[cyan]Device:[/] {device}\n"
        f"[cyan]Results:[/] {base_dir}\n\n"
        f"[dim]Wave-based neural networks learning continuous field representations[/]",
        title="🔬 Benchmark Suite",
        border_style="green"
    ))
    
    for exp_name in ACTIVE_EXPERIMENTS:
        if exp_name not in EXPERIMENTS:
            console.print(f"[yellow]⚠️  Warning: Experiment '{exp_name}' not defined, skipping...[/]")
            continue
            
        exp_config = EXPERIMENTS[exp_name]
        exp_results = run_experiment(exp_name, exp_config, device)
        all_results[exp_name] = (exp_config, exp_results)
        
        # Save experiment results
        exp_dir = f"{base_dir}/{exp_name}/"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Generate visualizations for this experiment
        console.print("[dim]Generating comprehensive dashboard...[/]")
        modes = list(exp_results.keys())
        plot_results(exp_results, modes, exp_dir, NUM_EPOCHS)
        
        for m in modes:
            if m in ["UserWave", "GatedWave"]:
                plot_layer_waves(exp_results[m]['model'], m, exp_dir)
                # Fourier decomposition for first 2 waves
                for wave_idx in range(min(2, exp_config.get('num_waves', NUM_WAVES))):
                    plot_wave_decomposition(exp_results[m]['model'], m, exp_dir, wave_idx=wave_idx)
    
    # Final comparison report
    console.print()
    console.print(Panel("[bold magenta]🎯 FINAL EXPERIMENT COMPARISON[/]", border_style="magenta"))
    
    for exp_name in ACTIVE_EXPERIMENTS:
        if exp_name not in all_results:
            continue
        exp_config, results = all_results[exp_name]
        
        # Create rich table
        table = Table(
            title=f"{exp_name.upper()}: {exp_config['description']}", 
            show_header=True, 
            header_style="bold cyan"
        )
        table.add_column("Model", style="cyan")
        table.add_column("Params", justify="right", style="yellow")
        table.add_column("Comp", justify="right", style="green")
        table.add_column("Train %", justify="right", style="blue")
        table.add_column("Test %", justify="right", style="magenta")
        table.add_column("Gap", justify="right", style="red")
        table.add_column("Time", justify="right", style="white")
        table.add_column("Speed", justify="right", style="dim")
        
        baseline_params = results["Standard"]["params"]
        for m in MODES:
            if m not in results:
                continue
            r = results[m]
            comp_ratio = baseline_params / r['params'] if r['params'] > 0 else 0
            gen_gap = r['train_acc'] - r['test_acc']
            
            # Highlight best test accuracy
            test_style = "bold magenta" if r['test_acc'] == max(results[mode]['test_acc'] for mode in results) else "magenta"
            
            table.add_row(
                m,
                f"{r['params']:,}",
                f"{comp_ratio:.1f}×",
                f"{r['train_acc']:.1f}",
                f"[{test_style}]{r['test_acc']:.1f}[/]",
                f"{gen_gap:.1f}",
                f"{r['total_time']:.0f}s",
                f"{r['inference_speed']:.0f}"
            )
        
        console.print(table)
        console.print()
    
    console.print(Panel(
        f"[bold green]✅ All experiments completed![/]\n\n"
        f"[cyan]Comprehensive dashboards saved to:[/]\n{base_dir}\n\n"
        f"[dim]Each experiment folder contains:[/]\n"
        f"  • [yellow]comprehensive_dashboard.png[/] - Full analysis\n"
        f"  • [yellow]{{Model}}_waves.png[/] - Wave decomposition\n"
        f"  • [yellow]{{Model}}_fourier_decomposition_wave{{N}}.png[/] - Harmonic analysis",
        title="🌊 Spectral Results",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
