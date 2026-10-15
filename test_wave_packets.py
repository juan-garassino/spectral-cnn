#!/usr/bin/env python3
"""
Test script to verify wave packet generation creates physics-like waves
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from spectral_gpt.wave_gpt import WavePacketEmbedding

def test_wave_packets():
    """Plot wave-packet examples to ``wave_packet_test.png``."""
    print("🌊 Testing Wave Packet Generation...")
    
    # Create embedding
    embedding = WavePacketEmbedding(vocab_size=100, d_model=64, num_waves=8, num_harmonics=3)
    
    # Test a few tokens
    token_ids = torch.tensor([[0, 1, 10, 50]])  # Shape: (1, 4)
    
    # Get wave parameters
    freqs = embedding.base_freqs[token_ids[0]].detach().numpy()  # (4, 8)
    phases = embedding.phases[token_ids[0]].detach().numpy()     # (4, 8)
    harm_amps = embedding.harmonic_amps[token_ids[0]].detach().numpy()  # (4, 8, 3)
    
    print(f"Frequency ranges:")
    print(f"  Min: {freqs.min():.3f} Hz")
    print(f"  Max: {freqs.max():.3f} Hz")
    print(f"  Mean: {freqs.mean():.3f} Hz")
    
    # Create wave packets manually (like the visualization does)
    positions = np.linspace(0, 50, 200)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for tok_idx in range(4):
        ax = axes[tok_idx]
        wave_sum = np.zeros_like(positions)
        
        # Use first 2 wave components
        for w in range(min(2, freqs.shape[1])):
            base_f = freqs[tok_idx, w]
            phase = phases[tok_idx, w]
            
            # Create multi-peak envelope with revivals
            # Primary peak
            primary_center = 15
            primary_width = 3.0 / (base_f + 0.1)
            primary_envelope = np.exp(-0.5 * ((positions - primary_center) / primary_width) ** 2)
            
            # Secondary peak (revival)
            secondary_center = 35
            secondary_width = 5.0 / (base_f + 0.1)
            secondary_envelope = 0.4 * np.exp(-0.5 * ((positions - secondary_center) / secondary_width) ** 2)
            
            # Beating modulation
            beat_freq = base_f * 0.05
            beat_modulation = 0.5 * (1.0 + np.cos(2 * np.pi * beat_freq * positions / 5.0))
            
            # Combined envelope
            envelope = (primary_envelope + secondary_envelope) * beat_modulation
            
            # Add harmonics
            for h in range(min(2, harm_amps.shape[2])):
                amp = harm_amps[tok_idx, w, h]
                freq = base_f * (h + 1)
                
                # Wave packet = envelope * oscillation
                wave_component = amp * envelope * np.cos(2 * np.pi * freq * positions / 10.0 + phase)
                wave_sum += wave_component
        
        ax.plot(positions, wave_sum, linewidth=2, color='cyan')
        ax.set_title(f'Token {token_ids[0][tok_idx].item()} (f={freqs[tok_idx, 0]:.2f}Hz)')
        ax.set_xlabel('Position')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', alpha=0.5, linewidth=0.5)
    
    plt.suptitle('Wave Packet Test - Should Look Like Physics Waves!', fontsize=14)
    plt.tight_layout()
    plt.savefig('wave_packet_test.png', dpi=150, bbox_inches='tight')
    print("✅ Saved wave_packet_test.png")
    
    # Test forward pass
    output = embedding(token_ids)
    print(f"✅ Forward pass works: {output.shape}")
    
    return True

if __name__ == "__main__":
    test_wave_packets()