import torch
from src.models.networks import UniversalMLP
from src.training.trainer import train_fit
import os

def test_dashboard(tmp_path):
    """Quick test to generate the new dashboard"""
    print("🎨 Testing new dashboard visualization...")
    
    device = torch.device("cpu")
    out_dir = str(tmp_path)  # never overwrite tracked results/ files
    
    # Train two quick models (1 epoch each)
    print("\n1. Training Standard MLP (1 epoch)...")
    results_std = train_fit("Standard", num_waves=12, num_epochs=1, device=device)
    
    print("\n2. Training UserWave MLP (1 epoch)...")
    results_wave = train_fit("UserWave", num_waves=12, num_epochs=1, device=device, 
                            wave_mode="outer_product")
    
    # Create results dict
    results = {
        "Standard": results_std,
        "UserWave": results_wave
    }
    
    # Generate dashboard
    from src.visualization import plotter
    plotter.plot_results(results, ["Standard", "UserWave"], 
                        out_dir, num_epochs=1)
    assert os.path.exists(os.path.join(out_dir, "comprehensive_dashboard.png"))
    
    print("\n✅ Dashboard test complete!")

if __name__ == "__main__":
    import tempfile
    test_dashboard(tempfile.mkdtemp())
