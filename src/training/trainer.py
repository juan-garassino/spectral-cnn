"""Training loop and optimizer construction for the MNIST benchmark."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
from rich.panel import Panel
from src.models.networks import UniversalMLP, SpectralCNN

console = Console()

# Configuration
BATCH_SIZE = 512
LR_PHASE, LR_AMP, LR_BIAS = 1e-3, 1e-3, 1e-2
LR_LOW_FREQ, LR_MID_FREQ, LR_HIGH_FREQ = 1e-3, 5e-4, 2e-4  # Different LRs per freq band

def get_optimizer(model, mode, frequency_aware=False):
    """
    Create optimizer with optional frequency-aware learning rates.
    
    Args:
        model: The neural network model
        mode: Model type (Standard, UserWave, etc.)
        frequency_aware: If True, use different learning rates for different frequency bands
    """
    if mode == "Standard": 
        return torch.optim.Adam(model.parameters(), lr=LR_BIAS)
    
    if not frequency_aware:
        # Original approach: group by parameter type
        phase, amp, bias = [], [], []
        for name, p in model.named_parameters():
            if 'bias' in name: bias.append(p)
            elif any(k in name for k in ['u', 'v', 'U', 'V', 'net']): phase.append(p)
            elif any(k in name for k in ['amp', 'coeff', 'fourier_coeffs']): amp.append(p)
            else: phase.append(p)
        return torch.optim.Adam([
            {'params': phase, 'lr': LR_PHASE},
            {'params': amp, 'lr': LR_AMP},
            {'params': bias, 'lr': LR_BIAS}
        ])
    
    else:
        # Frequency-aware approach: separate learning rates per frequency band
        phase, bias = [], []
        low_freq_coeffs = []
        amplitudes = []
        
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias.append(p)
            elif 'fourier_coeffs' in name:
                # Split Fourier coefficients by frequency band
                # We'll need to access the layer to know num_harmonics
                # For now, we'll add them to separate groups based on the parameter structure
                # This requires the layer to expose the harmonics separately
                # For simplicity, we'll just use different LRs for all coeffs for now
                # TODO: Implement per-harmonic grouping
                low_freq_coeffs.append(p)  # Placeholder - will improve
            elif any(k in name for k in ['u', 'v', 'U', 'V', 'net']):
                phase.append(p)
            elif 'amplitudes' in name:
                amplitudes.append(p)
            else:
                phase.append(p)
        
        param_groups = [
            {'params': phase, 'lr': LR_PHASE, 'name': 'phase'},
            {'params': low_freq_coeffs, 'lr': LR_LOW_FREQ, 'name': 'low_freq'},
            {'params': amplitudes, 'lr': LR_MID_FREQ, 'name': 'amplitudes'},
            {'params': bias, 'lr': LR_BIAS, 'name': 'bias'}
        ]
        
        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        
        return torch.optim.Adam(param_groups)

def train_fit(mode_name, num_waves, num_epochs, device, num_harmonics=3, 
             adaptive_freqs=False, per_neuron_coeffs=False, l1_penalty=0.0, wave_mode="outer_product"):
    
    """Train one model on MNIST and report accuracy and timing.

    Downloads MNIST into ``./data`` if absent, trains with Adam
    (batch size 512, gradient clipping at 1.0, ``ReduceLROnPlateau`` on test accuracy),
    and times a full pass over the test set for inference speed.

    Args:
        mode_name: ``"SpectralCNN"`` or a ``UniversalMLP`` layer type.
        num_waves: Waves per spectral layer.
        num_epochs: Number of training epochs.
        device: ``torch.device`` to train on.
        num_harmonics: Fourier harmonics per wave.
        adaptive_freqs: Make harmonic frequencies learnable.
        per_neuron_coeffs: One coefficient set per output neuron.
        l1_penalty: Weight of the L1 penalty on Fourier coefficients (0 disables).
        wave_mode: ``"outer_product"``, ``"fourier_series"`` or ``"gabor"``.

    Returns:
        Dict with ``test_acc`` and ``train_acc`` (percent, last epoch), ``params``
        (trainable parameter count), ``model``, ``history`` (per-epoch
        ``train_acc``/``test_acc`` lists), ``total_time`` (seconds) and
        ``inference_speed`` (test samples per second).
    """
    console.print(Panel(f"[bold cyan]Training: {mode_name}[/]", expand=False))
    
    # Data Loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST("./data", train=False, download=True, transform=transform), batch_size=256, shuffle=False)

    if mode_name == "SpectralCNN":
        model = SpectralCNN(num_waves=num_waves, num_harmonics=num_harmonics,
                            adaptive_freqs=adaptive_freqs, per_neuron_coeffs=per_neuron_coeffs,
                            wave_mode=wave_mode).to(device)
    else:
        model = UniversalMLP(mode_name, num_waves, num_harmonics=num_harmonics,
                            adaptive_freqs=adaptive_freqs, per_neuron_coeffs=per_neuron_coeffs,
                            wave_mode=wave_mode).to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    opt = get_optimizer(model, mode_name)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)

    start = time.time()
    history = {'train_acc': [], 'test_acc': []}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        # Calculate total batches
        total_batches = len(train_loader) * num_epochs
        batch_task = progress.add_task(f"[cyan]{mode_name}", total=total_batches)
        
        for epoch in range(num_epochs):
            model.train()
            correct_train = 0
            total_train = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y)
                
                # Add L1 penalty on Fourier coefficients if enabled
                if l1_penalty > 0.0:
                    calc_model = model.module if isinstance(model, nn.DataParallel) else model
                    loss = loss + l1_penalty * calc_model.get_l1_loss()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if isinstance(model, nn.DataParallel): model.module.constrain_all()
                else: model.constrain_all()

                correct_train += out.argmax(1).eq(y).sum().item()
                total_train += y.size(0)
                
                # Update progress bar every batch
                current_acc = 100.0 * correct_train / total_train if total_train > 0 else 0
                progress.update(
                    batch_task, 
                    advance=1, 
                    description=f"[cyan]{mode_name} | Epoch {epoch+1}/{num_epochs} | Acc: {current_acc:.1f}%"
                )

            train_acc = 100.0 * correct_train / total_train
            history['train_acc'].append(train_acc)

            model.eval()
            correct = 0
            with torch.no_grad():
                for x, y in test_loader:
                    correct += model(x.to(device)).argmax(1).eq(y.to(device)).sum().item()
            acc = 100.0 * correct / len(test_loader.dataset)
            history['test_acc'].append(acc)

            sched.step(acc)

    total_time = time.time() - start
    
    # Inference Speed Test
    model.eval()
    start_infer = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            _ = model(x.to(device))
    infer_time = time.time() - start_infer
    samples_per_sec = len(test_loader.dataset) / infer_time

    calc_model = model.module if isinstance(model, nn.DataParallel) else model
    return {
        'test_acc': acc,
        'train_acc': train_acc,
        'params': sum(p.numel() for p in calc_model.parameters() if p.requires_grad),
        'model': calc_model,
        'history': history,
        'total_time': total_time,
        'inference_speed': samples_per_sec
    }
