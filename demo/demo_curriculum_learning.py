"""
Demo: Frequency Curriculum Learning for Spectral Networks

This script demonstrates two strategies for combating spectral bias:
1. Different learning rates for different frequency components
2. Progressive unfreezing of high frequencies during training
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models.networks import UniversalMLP
from src.training.trainer import get_optimizer
import matplotlib.pyplot as plt
import os

def train_with_curriculum(model, train_loader, test_loader, epochs, device, use_progressive=True):
    """Train with curriculum learning (progressive unfreezing)"""
    opt = get_optimizer(model, "UserWave", frequency_aware=False)
    
    history = {'train_acc': [], 'test_acc': [], 'unfrozen_harmonics': []}
    
    for epoch in range(epochs):
        # Progressive unfreezing
        if use_progressive and hasattr(model.fc1, 'progressive_unfreeze_schedule'):
            num_unfrozen = model.fc1.progressive_unfreeze_schedule(epoch, epochs, strategy='linear')
            history['unfrozen_harmonics'].append(num_unfrozen)
            print(f"Epoch {epoch+1}: Unfrozen {num_unfrozen}/{model.fc1.num_harmonics} harmonics")
        
        # Training
        model.train()
        correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            model.constrain_all()
            correct += out.argmax(1).eq(y).sum().item()
        
        train_acc = 100.0 * correct / len(train_loader.dataset)
        history['train_acc'].append(train_acc)
        
        # Testing
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += model(x.to(device)).argmax(1).eq(y.to(device)).sum().item()
        test_acc = 100.0 * correct / len(test_loader.dataset)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: Train={train_acc:.1f}%, Test={test_acc:.1f}%")
    
    return history

def main():
    print("🎓 Frequency Curriculum Learning Demo\n")
    
    device = torch.device("cpu")
    epochs = 5
    
    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 1. Baseline: Standard training
    print("=" * 60)
    print("1. BASELINE: Standard Training (No Curriculum)")
    print("=" * 60)
    model_baseline = UniversalMLP("UserWave", num_waves=12, num_harmonics=5,
                                 adaptive_freqs=True, wave_mode="outer_product").to(device)
    history_baseline = train_with_curriculum(model_baseline, train_loader, test_loader, 
                                            epochs, device, use_progressive=False)
    
    # 2. Curriculum: Progressive unfreezing
    print("\n" + "=" * 60)
    print("2. CURRICULUM: Progressive Unfreezing")
    print("=" * 60)
    model_curriculum = UniversalMLP("UserWave", num_waves=12, num_harmonics=5,
                                   adaptive_freqs=True, wave_mode="outer_product").to(device)
    
    # Start with only low frequencies
    model_curriculum.fc1.freeze_high_frequencies(threshold=0.2)  # Start with 20% unfrozen
    model_curriculum.fc2.freeze_high_frequencies(threshold=0.2)
    model_curriculum.fc3.freeze_high_frequencies(threshold=0.2)
    
    history_curriculum = train_with_curriculum(model_curriculum, train_loader, test_loader, 
                                              epochs, device, use_progressive=True)
    
    # Visualization
    os.makedirs("results/curriculum_demo", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Test accuracy comparison
    ax = axes[0]
    ax.plot(history_baseline['test_acc'], 'o-', label='Baseline', linewidth=2, markersize=6)
    ax.plot(history_curriculum['test_acc'], 's-', label='Progressive Unfreezing', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Curriculum Learning Effect', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Unfrozen harmonics over time
    ax = axes[1]
    if history_curriculum['unfrozen_harmonics']:
        ax.plot(history_curriculum['unfrozen_harmonics'], 'g-', linewidth=3)
        ax.axhline(y=5, color='r', linestyle='--', label='Total Harmonics')
        ax.fill_between(range(len(history_curriculum['unfrozen_harmonics'])), 
                        0, history_curriculum['unfrozen_harmonics'], alpha=0.3, color='green')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Unfrozen Harmonics', fontsize=12)
        ax.set_title('Progressive Frequency Unfreezing', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/curriculum_demo/curriculum_comparison.png', dpi=150)
    print("\n✅ Results saved: results/curriculum_demo/curriculum_comparison.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline Final Acc:    {history_baseline['test_acc'][-1]:.2f}%")
    print(f"Curriculum Final Acc:  {history_curriculum['test_acc'][-1]:.2f}%")
    print(f"Improvement:           {history_curriculum['test_acc'][-1] - history_baseline['test_acc'][-1]:+.2f}%")

if __name__ == "__main__":
    main()
