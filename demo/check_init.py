from src.models.layers import UserWaveLinear

# Test the initialization
layer = UserWaveLinear(784, 10, num_waves=12, num_harmonics=5, wave_mode="outer_product")

print("🔍 Checking initialization stats...")
print(f"\nFourier coeffs mean: {layer.fourier_coeffs.mean().item():.4f}")
print(f"Fourier coeffs std: {layer.fourier_coeffs.std().item():.4f}")
print(f"Fourier coeffs min: {layer.fourier_coeffs.min().item():.4f}")
print(f"Fourier coeffs max: {layer.fourier_coeffs.max().item():.4f}")

print(f"\nAmplitudes mean: {layer.amplitudes.mean().item():.4f}")
print(f"Amplitudes std: {layer.amplitudes.std().item():.4f}")

# Generate a weight and check its scale
W = layer.get_weight()
print(f"\nGenerated weights mean: {W.mean().item():.4f}")
print(f"Generated weights std: {W.std().item():.4f}")
print(f"Generated weights min: {W.min().item():.4f}")
print(f"Generated weights max: {W.max().item():.4f}")

# Check if weights are exploding
if W.std() > 10.0:
    print("\n⚠️  WARNING: Weights are too large! This will cause training instability.")
elif W.std() < 0.01:
    print("\n⚠️  WARNING: Weights are too small! This will slow learning.")
else:
    print("\n✅ Weight scale looks reasonable.")
