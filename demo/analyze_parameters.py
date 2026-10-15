"""
Show exactly which parameters are learnable in both 1D and 2D wave modes.
"""

from src.models.layers import UserWaveLinear

def analyze_learnable_parameters():
    print("🔍 ANALYZING LEARNABLE WAVE PARAMETERS\n")
    print("=" * 80)
    
    # 1. 2D Outer Product Mode
    print("\n1️⃣  2D MODE (outer_product)")
    print("=" * 80)
    layer_2d = UserWaveLinear(784, 10, num_waves=12, num_harmonics=5, 
                              wave_mode="outer_product", adaptive_freqs=True)
    
    print("\n📊 Learnable Parameters:")
    total_params = 0
    for name, param in layer_2d.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name:25s} | Shape: {str(tuple(param.shape)):20s} | Count: {param.numel():8,}")
            total_params += param.numel()
    
    print(f"\n  TOTAL: {total_params:,} parameters")
    
    print("\n🌊 What Each Parameter Does:")
    print("  • u, v:             Phase/Coordinate vectors (define wave directions)")
    print("  • freqs:            Base frequency per wave (how fast it oscillates)")
    print("  • harmonic_freqs:   Frequency multipliers (1x, 2x, 4x, ...) [if adaptive_freqs=True]")
    print("  • fourier_coeffs:   Fourier coefficients (amplitude of each harmonic)")
    print("  • amplitudes:       Overall amplitude per wave")
    print("  • bias:             Output bias term")
    
    # 2. 1D Fourier Series Mode
    print("\n" + "=" * 80)
    print("\n2️⃣  1D MODE (fourier_series)")
    print("=" * 80)
    layer_1d = UserWaveLinear(784, 10, num_waves=12, num_harmonics=5, 
                              wave_mode="fourier_series", adaptive_freqs=True)
    
    print("\n📊 Learnable Parameters:")
    total_params = 0
    for name, param in layer_1d.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name:25s} | Shape: {str(tuple(param.shape)):20s} | Count: {param.numel():8,}")
            total_params += param.numel()
    
    print(f"\n  TOTAL: {total_params:,} parameters")
    
    print("\n🌊 What Each Parameter Does:")
    print("  • freqs:            Base frequency per wave")
    print("  • phases:           Explicit phase offset per wave (NEW in 1D mode!)")
    print("  • harmonic_freqs:   Frequency multipliers [if adaptive_freqs=True]")
    print("  • fourier_coeffs:   Fourier coefficients (amplitude of each harmonic)")
    print("  • amplitudes:       Per-neuron, per-wave amplitude (more expressive!)")
    print("  • bias:             Output bias term")
    
    # 3. Gabor Mode
    print("\n" + "=" * 80)
    print("\n3️⃣  GABOR MODE (gabor)")
    print("=" * 80)
    layer_gabor = UserWaveLinear(784, 10, num_waves=12, num_harmonics=5, 
                                 wave_mode="gabor", adaptive_freqs=True)
    
    print("\n📊 Learnable Parameters:")
    total_params = 0
    for name, param in layer_gabor.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name:25s} | Shape: {str(tuple(param.shape)):20s} | Count: {param.numel():8,}")
            total_params += param.numel()
    
    print(f"\n  TOTAL: {total_params:,} parameters")
    
    print("\n🌊 What Each Parameter Does:")
    print("  • centers:          Center position of Gaussian window (spatial localization)")
    print("  • sigmas:           Width of Gaussian window (scale/resolution)")
    print("  • freqs:            Base frequency per wavelet")
    print("  • phases:           Phase offset per wavelet")
    print("  • harmonic_freqs:   Frequency multipliers [if adaptive_freqs=True]")
    print("  • fourier_coeffs:   Fourier coefficients")
    print("  • amplitudes:       Per-neuron, per-wave amplitude")
    print("  • bias:             Output bias term")
    
    # Summary
    print("\n" + "=" * 80)
    print("\n✅ SUMMARY")
    print("=" * 80)
    print("\nYES! We optimize ALL wave-defining parameters:")
    print("  ✓ Phase      (u, v in 2D; explicit 'phases' in 1D/Gabor)")
    print("  ✓ Amplitude  (fourier_coeffs + amplitudes)")
    print("  ✓ Frequency  (freqs + harmonic_freqs if adaptive)")
    print("  ✓ Position   (centers in Gabor mode)")
    print("  ✓ Scale      (sigmas in Gabor mode)")
    print("  ✓ Bias       (bias)")
    print("\nThe network learns the ENTIRE wave representation from data! 🌊🎓")

if __name__ == "__main__":
    analyze_learnable_parameters()
