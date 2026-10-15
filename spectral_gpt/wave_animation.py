"""
Wave Inference Animation

Creates animated visualizations of wave dynamics during text generation.
Shows how waves interfere to select the next token.

Usage:
    python wave_animation.py --prompt "To be or not to be" --output wave_inference.mp4
    python wave_animation.py --model path/to/model.pt --tokens 50
"""

import os
import sys
import math
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'prototyping'))
sys.path.insert(0, current_dir)

from wave_gpt import WaveGPT, WaveGPTConfig

# Custom colormap
WAVE_CMAP = LinearSegmentedColormap.from_list(
    'wave', ['#000033', '#003366', '#006699', '#00CCCC', '#00FFFF', '#FFFFFF']
)


class WaveStateCapture:
    """Captures wave states during forward pass for visualization"""
    
    def __init__(self, model: WaveGPT):
        self.model = model
        self.wave_states = []
        self.attention_patterns = []
        self.logits_history = []
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture wave states"""
        
        def capture_embedding(module, input, output):
            # Capture the wave embedding output
            self.wave_states.append({
                'embedding': output.detach().cpu().numpy(),
                'wave_ratio': torch.sigmoid(module.wave_ratio).item() if hasattr(module, 'wave_ratio') else 1.0
            })
        
        # Hook on embedding layer
        self.hooks.append(
            self.model.embedding.register_forward_hook(capture_embedding)
        )
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear(self):
        """Clear captured states"""
        self.wave_states = []
        self.attention_patterns = []
        self.logits_history = []
    
    def get_wave_components(self, token_id: int) -> dict:
        """Get wave components for a specific token"""
        with torch.no_grad():
            base_freqs = self.model.embedding.base_freqs[token_id].cpu().numpy()
            phases = self.model.embedding.phases[token_id].cpu().numpy()
            harm_amps = self.model.embedding.harmonic_amps[token_id].cpu().numpy()
            
        return {
            'base_freqs': base_freqs,
            'phases': phases,
            'harm_amps': harm_amps
        }
    
    def generate_wave_signal(self, token_id: int, t: np.ndarray, n_waves: int = 8) -> np.ndarray:
        """Generate the wave signal for a token over time t"""
        components = self.get_wave_components(token_id)
        
        signal = np.zeros_like(t)
        for w in range(min(n_waves, len(components['base_freqs']))):
            base_f = components['base_freqs'][w]
            phase = components['phases'][w]
            for h in range(components['harm_amps'].shape[-1]):
                amp = components['harm_amps'][w, h]
                freq = base_f * (h + 1)
                signal += amp * np.sin(freq * t + phase)
        
        return signal


def create_wave_animation(
    model: WaveGPT,
    tokenizer,
    prompt: str,
    max_tokens: int = 20,
    device: str = "cuda",
    output_path: str = "wave_inference.mp4",
    fps: int = 10
):
    """
    Create an animated visualization of wave dynamics during generation.
    
    Args:
        model: Trained WaveGPT model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Starting prompt
        max_tokens: Tokens to generate
        device: Device to run on
        output_path: Output video path
        fps: Frames per second
    """
    
    model.eval()
    capture = WaveStateCapture(model)
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # Storage for animation frames
    frames_data = []
    generated_tokens = list(prompt_tokens)
    generated_text = prompt
    
    print(f"🎬 Generating {max_tokens} tokens with wave capture...")
    
    with torch.no_grad():
        for step in range(max_tokens):
            # Capture wave states
            capture.register_hooks()
            
            # Forward pass
            logits, _ = model(idx)
            next_logits = logits[0, -1, :]
            
            # Get probabilities
            probs = F.softmax(next_logits / 0.8, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            
            # Get top-k tokens for visualization
            top_probs, top_indices = torch.topk(probs, k=10)
            
            # Capture frame data
            frame = {
                'step': step,
                'current_text': generated_text,
                'next_token_id': next_token.item(),
                'next_token_text': tokenizer.decode([next_token.item()]),
                'top_probs': top_probs.cpu().numpy(),
                'top_indices': top_indices.cpu().numpy(),
                'wave_states': capture.wave_states[-1] if capture.wave_states else None,
                'all_probs': probs.cpu().numpy()
            }
            
            # Get wave signals for top tokens
            t = np.linspace(0, 4 * np.pi, 200)
            frame['wave_signals'] = {}
            for tok_id in top_indices[:5].tolist():
                frame['wave_signals'][tok_id] = capture.generate_wave_signal(tok_id, t)
            
            frames_data.append(frame)
            
            # Update sequence
            idx = torch.cat([idx, next_token.unsqueeze(0)], dim=1)
            generated_tokens.append(next_token.item())
            generated_text = tokenizer.decode(generated_tokens)
            
            capture.remove_hooks()
            capture.clear()
            
            print(f"  Step {step+1}/{max_tokens}: '{frame['next_token_text']}'")
    
    print(f"\n🎨 Creating animation with {len(frames_data)} frames...")
    
    # Create animation
    fig = plt.figure(figsize=(16, 10), facecolor='black')
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1.5, 1])
    
    # Axes setup
    ax_text = fig.add_subplot(gs[0, :])  # Generated text
    ax_waves = fig.add_subplot(gs[1, :2])  # Wave interference
    ax_probs = fig.add_subplot(gs[1, 2])  # Token probabilities
    ax_phase = fig.add_subplot(gs[2, 0], projection='polar')  # Phase diagram
    ax_spectrum = fig.add_subplot(gs[2, 1])  # Frequency spectrum
    ax_info = fig.add_subplot(gs[2, 2])  # Info panel
    
    for ax in [ax_text, ax_waves, ax_probs, ax_spectrum, ax_info]:
        ax.set_facecolor('black')
    
    t = np.linspace(0, 4 * np.pi, 200)
    
    def animate(frame_idx):
        frame = frames_data[frame_idx]
        
        # Clear all axes
        for ax in [ax_text, ax_waves, ax_probs, ax_spectrum, ax_info]:
            ax.clear()
            ax.set_facecolor('black')
        ax_phase.clear()
        
        # 1. Text display
        ax_text.text(0.5, 0.5, frame['current_text'][-100:] + '█', 
                     fontsize=14, color='white', ha='center', va='center',
                     family='monospace', wrap=True)
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        ax_text.axis('off')
        ax_text.set_title(f"Step {frame['step']+1}: Generating...", color='cyan', fontsize=12)
        
        # 2. Wave interference plot
        colors_waves = ['cyan', 'magenta', 'yellow', 'lime', 'orange']
        top_5 = list(frame['wave_signals'].keys())[:5]
        
        combined_wave = np.zeros_like(t)
        for i, tok_id in enumerate(top_5):
            signal = frame['wave_signals'][tok_id]
            prob = frame['top_probs'][i]
            weighted_signal = signal * prob
            combined_wave += weighted_signal
            
            # Plot individual waves (faded)
            ax_waves.plot(t, signal * 0.3 + i * 2, color=colors_waves[i], 
                         alpha=0.4, linewidth=1)
        
        # Plot combined wave
        ax_waves.plot(t, combined_wave, color='white', linewidth=2, 
                     label='Interference (Selection)')
        ax_waves.fill_between(t, combined_wave, alpha=0.3, color='cyan')
        
        ax_waves.set_xlim(0, 4 * np.pi)
        ax_waves.set_xlabel('Phase', color='white')
        ax_waves.set_ylabel('Amplitude', color='white')
        ax_waves.set_title('Wave Interference → Token Selection', color='cyan')
        ax_waves.tick_params(colors='white')
        ax_waves.grid(True, alpha=0.2)
        
        # 3. Token probabilities
        top_10_probs = frame['top_probs']
        top_10_indices = frame['top_indices']
        y_pos = np.arange(len(top_10_probs))
        
        bars = ax_probs.barh(y_pos, top_10_probs, color='cyan', alpha=0.8)
        
        # Highlight selected token
        selected_idx = np.where(top_10_indices == frame['next_token_id'])[0]
        if len(selected_idx) > 0:
            bars[selected_idx[0]].set_color('lime')
        
        ax_probs.set_yticks(y_pos)
        ax_probs.set_yticklabels([f"T{i}" for i in top_10_indices], color='white')
        ax_probs.set_xlabel('Probability', color='white')
        ax_probs.set_title('Top-10 Tokens', color='cyan')
        ax_probs.tick_params(colors='white')
        ax_probs.invert_yaxis()
        
        # 4. Phase diagram (polar)
        for i, tok_id in enumerate(top_5):
            components = WaveStateCapture(model).get_wave_components(tok_id)
            phases = components['phases'][:8]  # First 8 waves
            ax_phase.scatter(phases, np.ones_like(phases) * (i + 1) * 0.2,
                           c=colors_waves[i], s=50, alpha=0.8)
        
        ax_phase.set_title('Phase Positions', color='cyan')
        ax_phase.set_facecolor('black')
        ax_phase.tick_params(colors='white')
        
        # 5. Frequency spectrum
        for i, tok_id in enumerate(top_5):
            components = WaveStateCapture(model).get_wave_components(tok_id)
            freqs = components['base_freqs'][:16]
            ax_spectrum.bar(np.arange(len(freqs)) + i*0.15, np.abs(freqs), 
                          width=0.15, color=colors_waves[i], alpha=0.7)
        
        ax_spectrum.set_xlabel('Wave Component', color='white')
        ax_spectrum.set_ylabel('Frequency', color='white')
        ax_spectrum.set_title('Frequency Spectrum', color='cyan')
        ax_spectrum.tick_params(colors='white')
        
        # 6. Info panel
        ax_info.text(0.1, 0.9, f"Selected Token:", color='cyan', fontsize=10,
                    transform=ax_info.transAxes)
        ax_info.text(0.1, 0.7, f"'{frame['next_token_text']}'", color='lime', fontsize=14,
                    transform=ax_info.transAxes, fontweight='bold')
        ax_info.text(0.1, 0.5, f"Probability: {frame['top_probs'][0]:.3f}", color='white', fontsize=10,
                    transform=ax_info.transAxes)
        
        if frame.get('wave_states') and 'wave_ratio' in frame['wave_states']:
            ax_info.text(0.1, 0.3, f"Wave Ratio: {frame['wave_states']['wave_ratio']:.3f}", 
                        color='magenta', fontsize=10, transform=ax_info.transAxes)
        
        ax_info.axis('off')
        
        plt.tight_layout()
        return []
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames_data),
        interval=1000//fps, blit=False
    )
    
    # Save
    print(f"💾 Saving to {output_path}...")
    
    if output_path.endswith('.mp4'):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_path, writer=writer)
    elif output_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
    else:
        # Default to mp4
        output_path = output_path + '.mp4'
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_path, writer=writer)
    
    plt.close(fig)
    print(f"✅ Animation saved to: {output_path}")
    print(f"   Generated text: {generated_text}")
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Wave Inference Animation")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="To be or not to be",
                        help="Starting prompt")
    parser.add_argument("--tokens", type=int, default=20,
                        help="Tokens to generate")
    parser.add_argument("--output", type=str, default="wave_inference.mp4",
                        help="Output file (mp4 or gif)")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames per second")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🌊 Wave Inference Animation")
    print(f"   Device: {device}")
    
    # Load or create model
    if args.model and os.path.exists(args.model):
        print(f"📥 Loading model from {args.model}")
        checkpoint = torch.load(args.model, map_location=device, weights_only=True)
        
        # Reconstruct config
        config_dict = checkpoint.get('config', {})
        config = WaveGPTConfig(**config_dict)
        model = WaveGPT(config).to(device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Create small demo model
        print("📦 Creating demo model (not trained)")
        config = WaveGPTConfig(
            vocab_size=1024, d_model=256, num_layers=4,
            num_heads=4, num_waves=32, num_harmonics=4,
            block_size=128, dropout=0.0
        )
        model = WaveGPT(config).to(device)
    
    # Load tokenizer
    tokenizer_path = os.path.join(current_dir, "benchmark_results", "tokenizer.json")
    if os.path.exists(tokenizer_path):
        import json
        from train import BasicTokenizer
        tokenizer = BasicTokenizer()
        with open(tokenizer_path, 'r') as f:
            data = json.load(f)
            tokenizer.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
        print(f"📖 Loaded tokenizer from {tokenizer_path}")
    else:
        print("⚠️  No tokenizer found, using character-level")
        # Simple char tokenizer fallback
        class CharTokenizer:
            def encode(self, text):
                return [ord(c) % 1024 for c in text]
            def decode(self, ids):
                return ''.join([chr(i % 128) for i in ids])
        tokenizer = CharTokenizer()
    
    # Generate animation
    create_wave_animation(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.tokens,
        device=device,
        output_path=args.output,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
