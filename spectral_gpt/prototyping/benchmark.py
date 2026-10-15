"""
Spectral GPT Benchmark Suite 🏎️📊

Compares 4 architectures on Tiny Shakespeare:
1. Classic Transformer (Attention + Standard Weights)
2. Spectral Transformer (Attention + Wave Weights)
3. FFT Mixer (FFT + Standard Weights) - "Transformer Killer"
4. Full Spectral (FFT + Wave Weights)

Features:
- Metrics: Loss, Perplexity, Speed, Memory, Params
- Artifacts: Saves models, tokenizer, and config
- Visualization: Loss curves, comparison plots
- Packaging: Zips everything into benchmark_results.zip
"""

import torch
import torch.nn as nn
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import shutil
import math
import ast
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

import sys
import os
# Add project root to sys.path to find 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from model import SpectralGPT
from train import BasicTokenizer, get_batch, GPTConfig

# Configuration
STEPS = 1000       # Increased for wave convergence
BATCH_SIZE = 8     # Smaller batch for wave models memory
BLOCK_SIZE = 128
D_MODEL = 128      # ~1M params for ALL models (fair comparison)
LAYERS = 4         # Fewer layers for ~1M params
HEADS = 4          # Fewer heads for ~1M params
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_benchmark(config_name, layer_type, weight_type, train_data, val_data, vocab_size, results_dir, 
                  init_mode="standard", use_hamiltonian=False, use_collapse=False, activation_type="gelu", 
                  hybrid_mode=False, complex_attention=False, wave_mode="outer_product", tokenizer=None, console=None):
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    
    console.print(Panel(f"[bold cyan]{config_name}[/bold cyan]\n"
                       f"Layer: {layer_type} | Weights: {weight_type} | Activation: {activation_type}\n"
                       f"Init: {init_mode} | WaveMode: {wave_mode} | Hamiltonian: {use_hamiltonian}",
                       title="🧪 Model Config", border_style="cyan"))
    
    # Reset
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # ALL models use same size for fair comparison
    d_model = D_MODEL
    n_layers = LAYERS
    n_heads = HEADS
    dropout = 0.1
    
    # Wave capacity: Use CNN's proven optimal settings
    # 12 waves × 5 harmonics = stable and effective (matches CNN benchmark)
    num_waves = 12
    num_harmonics = 5
    
    # Config
    config = GPTConfig(
        vocab_size=vocab_size, d_model=d_model, num_layers=n_layers,
        num_heads=n_heads, d_ff=d_model*4, block_size=BLOCK_SIZE,
        dropout=dropout, layer_type=layer_type, weight_type=weight_type,
        num_waves=num_waves, num_harmonics=num_harmonics,
        init_mode=init_mode, activation_type=activation_type,
        hybrid_mode=hybrid_mode, complex_attention=complex_attention,
        wave_mode=wave_mode
    )
    
    # Model
    try:
        model = SpectralGPT(config).to(DEVICE)
        
        # Normalize wave weights for stability
        if weight_type == "wave":
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'fourier_coeffs' in name or 'amplitudes' in name:
                        param.data *= 0.1  # Scale down for stability
                    if 'freqs' in name and 'harmonic' not in name:
                        param.data.clamp_(0.1, 5.0)  # Limit frequency range
    except Exception as e:
        console.print(f"[red]Failed to init model: {e}[/red]")
        return None

    params = sum(p.numel() for p in model.parameters())
    console.print(f"📊 Parameters: [bold]{params:,}[/bold] ({params/1e6:.2f}M)")
    
    # Wave models need lower LR to prevent NaN collapse
    if weight_type == "wave":
        lr = 1e-4  # Much lower for wave stability
    else:
        lr = 1e-3  # Standard for classic transformer
    
    console.print(f"⚙️  Learning Rate: [bold]{lr:.0e}[/bold] | Dropout: [bold]{dropout}[/bold]")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Training Loop
    model.train()
    losses = []
    
    # Warmup (with try/except to catch early NaN)
    console.print("🔥 Warming up...")
    try:
        for _ in range(5):
            x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            with torch.no_grad():
                _, _ = model(x, y)
    except Exception as e:
        console.print(f"[red]Warmup failed: {e}[/red]")
        return None
    
    from train import get_collapse_penalty
    
    # Timed Run with Progress Bar and Live Metrics
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[yellow]Loss: {task.fields[loss]:.4f}"),
        TextColumn("[cyan]Avg: {task.fields[avg_loss]:.4f}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task(f"Training {config_name}", total=STEPS, loss=0.0, avg_loss=0.0)
        run_start = time.time()
        
        for step in range(STEPS):
            x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            
            # Forward
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
                
            # Physics Penalties
            if use_collapse:
                loss += 1e-4 * get_collapse_penalty(model)
                
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (more aggressive for wave models)
            clip_value = 0.5 if weight_type == "wave" else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            
            # Wave Stabilization: Built into model
            if weight_type == "wave":
                model.stabilize_waves()
            
            # Physics Constraints: Energy conservation
            if use_hamiltonian or (weight_type == "wave" and step % 10 == 0):
                try:
                    model.constrain_energy()
                except:
                    pass
            
            losses.append(loss.item())
            avg_loss = sum(losses[-50:]) / len(losses[-50:])  # Last 50 steps average
            
            # Check for NaN/Inf
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                progress.update(task, advance=1, loss=float('nan'), avg_loss=float('nan'))
                console.print(f"[red]⚠️  Training collapsed at step {step} (NaN/Inf detected). Stopping early.[/red]")
                break
            
            progress.update(task, advance=1, loss=loss.item(), avg_loss=avg_loss)
            
            # Validation check every 50 steps
            if step % 50 == 0 and step > 0:
                model.eval()
                with torch.no_grad():
                    # Quick val check
                    val_x, val_y = get_batch(val_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
                    _, val_loss = model(val_x, val_y)
                    val_ppl = math.exp(min(val_loss.item(), 20))  # Cap for display
                    
                    # Only generate if model is stable
                    if tokenizer and not math.isnan(val_loss.item()):
                        try:
                            ctx = torch.tensor([[tokenizer.encode("\n")[0]]], dtype=torch.long, device=DEVICE)
                            sample = model.generate(ctx, 29, temperature=0.8)
                            text = tokenizer.decode(sample[0].tolist())
                            console.print(f"[dim]Step {step:3d} | Val Loss: {val_loss.item():.4f} | PPL: {val_ppl:.2f} | Sample: {text[:40]}...[/dim]")
                        except:
                            console.print(f"[dim]Step {step:3d} | Val Loss: {val_loss.item():.4f} | PPL: {val_ppl:.2f} | [red]Generation failed[/red][/dim]")
                model.train()
        
        total_time = time.time() - run_start
    
    tokens_per_sec = (STEPS * BATCH_SIZE * BLOCK_SIZE) / total_time
    
    # Check if model collapsed
    if len(losses) == 0 or math.isnan(losses[-1]) or math.isinf(losses[-1]):
        console.print("[red]⚠️  Model training collapsed. Skipping evaluation.[/red]")
        return {
            "Model": config_name,
            "Layer": layer_type,
            "Weights": weight_type,
            "Params": params,
            "Speed (tok/s)": 0,
            "Memory (MB)": 0,
            "Train Loss": float('nan'),
            "Val Loss": float('nan'),
            "Perplexity": float('nan'),
            "Time (s)": round(total_time, 2),
            "loss_history": losses
        }
    
    # Evaluation
    console.print("📈 Evaluating...")
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(20):
            x, y = get_batch(val_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            _, loss = model(x, y)
            val_losses.append(loss.item())
            
    final_val_loss = sum(val_losses) / len(val_losses)
    perplexity = math.exp(final_val_loss)
    
    memory_mb = 0
    if DEVICE == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Final Generation Sample
    if tokenizer:
        console.print("\n[bold green]📝 Final Generation Sample:[/bold green]")
        model.eval()
        with torch.no_grad():
            ctx = torch.tensor([[tokenizer.encode("\n")[0]]], dtype=torch.long, device=DEVICE)
            sample = model.generate(ctx, 100, temperature=0.8)
            text = tokenizer.decode(sample[0].tolist())
            console.print(Panel(text, title="Generated Text", border_style="green"))
        
    # Save Model
    model_name = config_name.replace(" ", "_").lower()
    torch.save(model.state_dict(), results_dir / "models" / f"{model_name}.pt")
    
    # Print Results
    console.print(Panel(
        f"[bold]Speed:[/bold] {tokens_per_sec:,.0f} tok/s\n"
        f"[bold]Perplexity:[/bold] {perplexity:.2f}\n"
        f"[bold]Val Loss:[/bold] {final_val_loss:.4f}\n"
        f"[bold]Memory:[/bold] {memory_mb:.0f} MB\n"
        f"[bold]Time:[/bold] {total_time:.1f}s",
        title=f"✅ {config_name} Results",
        border_style="green"
    ))
    
    return {
        "Model": config_name,
        "Layer": layer_type,
        "Weights": weight_type,
        "Params": params,
        "Speed (tok/s)": int(tokens_per_sec),
        "Memory (MB)": int(memory_mb),
        "Train Loss": round(losses[-1], 4),
        "Val Loss": round(final_val_loss, 4),
        "Perplexity": round(perplexity, 2),
        "Time (s)": round(total_time, 2),
        "loss_history": losses
    }

def main():
    console = Console()
    console.print("[bold green]Starting Spectral GPT Benchmark Suite...[/bold green]")
    
    # Setup Results Directory (preserve tokenizer if it exists)
    results_dir = Path("benchmark_results")
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    # Data
    if not os.path.exists("input.txt"):
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "input.txt")
    
    with open("input.txt", 'r') as f: text = f.read()
    
    # Check if tokenizer already exists
    tokenizer_path = results_dir / "tokenizer.json"
    if tokenizer_path.exists():
        console.print("[cyan]♻️  Loading existing tokenizer...[/cyan]")
        tokenizer = BasicTokenizer()
        with open(tokenizer_path, 'r') as f:
            data = json.load(f)
            tokenizer.merges = {ast.literal_eval(k): v for k, v in data['merges'].items()}
            tokenizer.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
    else:
        console.print("[yellow]🔧 Training new tokenizer...[/yellow]")
        tokenizer = BasicTokenizer()
        tokenizer.train(text, 1024)
        tokenizer.save(tokenizer_path)
        console.print(f"[green]✅ Tokenizer saved to {tokenizer_path}[/green]")
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    
    # Configurations to Test
    # Format: (Name, Layer, Weight, InitMode, UseHamiltonian, UseCollapse, Activation, Hybrid, Complex, WaveMode)
    # Note: All models must use 'attention' to be causal (no peeking at future tokens)
    configs = [
        ("Classic Transformer", "attention", "standard", "standard", False, False, "gelu", False, False, "outer_product"),
        ("Spectral Transformer", "attention", "wave", "standard", False, False, "gelu", False, False, "outer_product"),
        ("Gabor Spectral 🌊", "attention", "wave", "standing_wave", False, False, "gelu", False, False, "gabor"),  # CNN winner!
        ("Physics Spectral ⚛️", "attention", "wave", "holographic", True, True, "modulate", False, False, "outer_product"),
    ]
    
    results = []
    loss_histories = {}
    for name, layer, weight, init, ham, coll, act, hybrid, complex_att, wave_mode in configs:
        res = run_benchmark(name, layer, weight, train_data, val_data, 1024, results_dir, 
                          init_mode=init, use_hamiltonian=ham, use_collapse=coll, activation_type=act,
                          hybrid_mode=hybrid, complex_attention=complex_att, wave_mode=wave_mode,
                          tokenizer=tokenizer, console=console)
        if res:
            loss_histories[name] = res.pop("loss_history")
            results.append(res)
        
    # Display Results
    table = Table(title="Spectral GPT Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Speed (tok/s)", justify="right", style="green")
    table.add_column("Perplexity", justify="right", style="yellow")
    table.add_column("Val Loss", justify="right", style="red")
    table.add_column("Memory (MB)", justify="right", style="magenta")
    
    for r in results:
        table.add_row(
            r["Model"], 
            f"{r['Params']:,}", 
            f"{r['Speed (tok/s)']:,}", 
            f"{r['Perplexity']}",
            f"{r['Val Loss']}",
            f"{r['Memory (MB)']}"
        )
        
    console.print(table)
    
    # Save Metrics
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "metrics.csv", index=False)
    
    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    
    # 1. Training Loss Curves
    plt.figure(figsize=(12, 6))
    for name, history in loss_histories.items():
        plt.plot(history, label=name, alpha=0.8)
    plt.title("Training Loss Convergence")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(results_dir / "plots" / "loss_curves.png")
    plt.close()
    
    # 2. Speed Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="Speed (tok/s)", hue="Model", palette="viridis")
    plt.title("Training Speed (Tokens/Sec)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(results_dir / "plots" / "speed_comparison.png")
    plt.close()
    
    # 3. Perplexity Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="Perplexity", hue="Model", palette="magma")
    plt.title("Validation Perplexity (Lower is Better)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(results_dir / "plots" / "perplexity_comparison.png")
    plt.close()
    
    # 4. Memory Usage
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="Memory (MB)", hue="Model", palette="rocket")
    plt.title("Peak Memory Usage (MB)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(results_dir / "plots" / "memory_usage.png")
    plt.close()
    
    # --- STRESS TEST (Long Context) ---
    console.print("\n[bold red]🔥 Starting Long-Context Stress Test (N=2048)...[/bold red]")
    stress_results = []
    
    # Only compare architecture scaling (Standard Weights) to isolate O(N^2) vs O(NlogN)
    stress_configs = [
        ("Classic Transformer", "attention", "standard"),
        ("FFT Mixer (GFNet)", "fft", "standard"),
    ]
    
    # Stress Config
    STRESS_BLOCK_SIZE = 2048
    STRESS_BATCH_SIZE = 2  # Low batch size to fit in memory
    
    for name, layer, weight in stress_configs:
        print(f"\n--- Stress Testing: {name} ---")
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        config = GPTConfig(
            vocab_size=1024, d_model=D_MODEL, num_layers=LAYERS,
            num_heads=HEADS, d_ff=D_MODEL*4, block_size=STRESS_BLOCK_SIZE,
            dropout=0.0, layer_type=layer, weight_type=weight,
            num_waves=12, num_harmonics=5
        )
        
        try:
            model = SpectralGPT(config).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            # Warmup
            x = torch.randint(0, 1024, (STRESS_BATCH_SIZE, STRESS_BLOCK_SIZE)).to(DEVICE)
            y = torch.randint(0, 1024, (STRESS_BATCH_SIZE, STRESS_BLOCK_SIZE)).to(DEVICE)
            model(x, y)
            
            # Timed Run (Forward + Backward)
            start = time.time()
            for _ in range(20): # 20 steps
                if DEVICE == 'cuda':
                    with torch.amp.autocast('cuda'): _, loss = model(x, y)
                else:
                    _, loss = model(x, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            dt = time.time() - start
            tok_per_sec = (20 * STRESS_BATCH_SIZE * STRESS_BLOCK_SIZE) / dt
            
            mem = 0
            if DEVICE == 'cuda':
                mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            stress_results.append({
                "Model": name,
                "Speed (tok/s)": int(tok_per_sec),
                "Memory (MB)": int(mem),
                "Time (s)": round(dt, 2)
            })
            
        except RuntimeError as e:
            print(f"OOM or Error: {e}")
            stress_results.append({
                "Model": name,
                "Speed (tok/s)": 0,
                "Memory (MB)": -1,
                "Time (s)": -1
            })

    # Display Stress Results
    stress_table = Table(title="Long-Context Stress Test (N=2048)")
    stress_table.add_column("Model", style="cyan")
    stress_table.add_column("Speed (tok/s)", justify="right", style="green")
    stress_table.add_column("Memory (MB)", justify="right", style="magenta")
    
    for r in stress_results:
        stress_table.add_row(r["Model"], f"{r['Speed (tok/s)']:,}", f"{r['Memory (MB)']}")
    console.print(stress_table)
    
    # Save Stress Metrics
    pd.DataFrame(stress_results).to_csv(results_dir / "stress_metrics.csv", index=False)

    # Zip Results (Update with stress test)
    shutil.make_archive("benchmark_results", 'zip', results_dir)
    console.print(f"\n[bold green]✅ Benchmark & Stress Test Complete![/bold green]")
    console.print(f"Results saved to [underline]benchmark_results.zip[/underline]")

if __name__ == "__main__":
    main()
