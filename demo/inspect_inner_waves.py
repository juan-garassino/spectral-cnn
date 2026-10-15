"""Visualize the individual waves inside a ``UserWaveLinear`` layer."""
import torch
import os
from src.models.layers import UserWaveLinear
from src.visualization import plotter
from torchvision import datasets, transforms

def inspect_waves():
    """Build a ``UserWaveLinear`` layer and save wave and Fourier-decomposition plots to ``results/inspection/``."""
    print("🌊 Inspecting Inner Waves...")
    
    # 1. Initialize a model with 2D Outer Product waves
    # We use a single layer to isolate the wave behavior
    layer = UserWaveLinear(28*28, 10, num_waves=8, num_harmonics=5, wave_mode="outer_product", force_linear_coords=True)
    
    # 2. Train briefly on one batch to get "meaningful" waves (not just random)
    print("Training on 1 batch to learn some structure...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    opt = torch.optim.Adam(layer.parameters(), lr=0.01)
    data, target = next(iter(loader))
    data = data.view(data.size(0), -1)
    
    for _ in range(50): # 50 steps on the same batch
        opt.zero_grad()
        out = layer(data)
        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()
        opt.step()
        layer.constrain() # Ensure waves stay well-behaved
        
    print(f"Training complete. Final Loss: {loss.item():.4f}")

    # 3. Create visualization directory
    os.makedirs("results/inspection", exist_ok=True)
    
    # 4. Plot the separate channels
    # We wrap the layer in a dummy class to match plotter's expected interface
    class DummyModel:
        def __init__(self, layer): self.fc1 = layer
    model = DummyModel(layer)
    
    print("Generating plots...")
    
    # A. Plot the individual channels (The "True Inner")
    plotter.plot_layer_waves(model, "Inner_Channels", "results/inspection")
    
    # B. Plot the harmonic decomposition of the first channel
    plotter.plot_wave_decomposition(model, "Channel_0_Decomposition", "results/inspection", wave_idx=0)
    
    print("\n✅ Done! Check 'results/inspection' for:")
    print("1. Inner_Channels_waves.png (The separate channels)")
    print("2. Channel_0_Decomposition...png (The harmonics inside a channel)")

if __name__ == "__main__":
    inspect_waves()
