import matplotlib.pyplot as plt
import numpy as np

def plot_results(results, modes, base_dir, num_epochs):
    print("\n[Generating Enhanced Visualization Dashboard...]")
    N = len(modes)
    
    # Create a comprehensive dashboard figure with new layout
    # Grid: [Explanations (2 cols)] / [Weight | Frequency] x N models / [Learning Curves (2 cols)] / [Table (2 cols)]
    fig = plt.figure(figsize=(14, 6 + N * 3))
    gs = fig.add_gridspec(N + 3, 2, hspace=0.4, wspace=0.3,
                          left=0.08, right=0.95, top=0.94, bottom=0.05)
    
    # Title
    fig.suptitle('Spectral CNN: Comprehensive Model Analysis Dashboard', 
                 fontsize=16, fontweight='bold')
    
    # === ROW 0: Combined Explanations (Spanning both columns) ===
    ax_exp = fig.add_subplot(gs[0, :])
    ax_exp.axis('off')
    
    exp_text = """
    📊 UNDERSTANDING THE VISUALIZATIONS
    
    WEIGHT PATTERNS (Left Column - Spatial Domain)          |    FREQUENCY ANALYSIS (Right Column - Spectral Domain)
    • First layer weights reshaped to 28×28 images          |    • 2D Fourier Transform of weight patterns
    • Shows what each neuron "looks for" in the input        |    • Center = Low frequencies (smooth patterns)
    • Spectral models use wave interference patterns         |    • Edges = High frequencies (sharp details/edges)
    • Bright = High values, Dark = Low values                |    • Reveals if model uses smooth vs detailed features
    
    LEARNING CURVES (Bottom): Test accuracy over training epochs. Steep = fast learning, Flat = converged, Higher = better.
    """
    
    ax_exp.text(0.5, 0.5, exp_text, fontsize=10, verticalalignment='center', 
                horizontalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4, pad=1))
    
    # === ROWS 1 to N: Weight Patterns (Col 0) and Frequency Analysis (Col 1) ===
    for i, m in enumerate(modes):
        model = results[m]['model']
        W = model.get_first_layer_weight().detach().cpu().numpy()
        W_neuron0 = W[0].reshape(28, 28)
        
        # Column 0: Weight Pattern
        ax_weight = fig.add_subplot(gs[i + 1, 0])
        im1 = ax_weight.imshow(W_neuron0, cmap='viridis')
        ax_weight.set_title(f"{m} - Weight Pattern\nAcc: {results[m]['test_acc']:.1f}%", 
                           fontsize=11, fontweight='bold')
        ax_weight.axis('off')
        plt.colorbar(im1, ax=ax_weight, fraction=0.046, pad=0.04)
        
        # Column 1: Frequency Analysis
        ax_freq = fig.add_subplot(gs[i + 1, 1])
        f_transform = np.fft.fft2(W_neuron0)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)
        
        im2 = ax_freq.imshow(magnitude_spectrum, cmap='inferno')
        ax_freq.set_title(f"{m} - Frequency Spectrum (FFT)", 
                         fontsize=11, fontweight='bold')
        ax_freq.axis('off')
        plt.colorbar(im2, ax=ax_freq, fraction=0.046, pad=0.04)
    
    # === ROW N+1: Learning Curves (Spanning both columns) ===
    ax_train = fig.add_subplot(gs[N + 1, :])
    for m in modes:
        hist = results[m]['history']
        ax_train.plot(hist['test_acc'], label=f"{m}", linewidth=2.5, marker='o', markersize=5)
    
    ax_train.set_xlabel("Epochs", fontsize=12, fontweight='bold')
    ax_train.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight='bold')
    ax_train.set_title(f"Training Dynamics - Learning Curves ({num_epochs} Epochs)", 
                      fontsize=13, fontweight='bold')
    ax_train.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax_train.grid(True, alpha=0.3, linestyle='--')
    ax_train.set_ylim([0, 100])
    
    # === ROW N+2: Performance Summary Table (Spanning both columns) ===
    ax_table = fig.add_subplot(gs[N + 2, :])
    ax_table.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Model', 'Test Acc', 'Train Acc', 'Parameters', 'Inference Speed'])
    for m in modes:
        r = results[m]
        table_data.append([
            m,
            f"{r['test_acc']:.2f}%",
            f"{r['train_acc']:.2f}%",
            f"{r['params']:,}",
            f"{r['inference_speed']:.0f} samples/s"
        ])
    
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                           colWidths=[0.18, 0.14, 0.14, 0.18, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(5):
            table[(i, j)].set_facecolor(color)
    
    plt.savefig(f"{base_dir}/comprehensive_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Dashboard saved: {base_dir}/comprehensive_dashboard.png")

def plot_layer_waves(model, mode_name, base_dir):
    """
    Visualizes the internal waves of the first layer if supported.
    """
    if not hasattr(model.fc1, 'get_waves'):
        print(f"Skipping wave plot for {mode_name} (not supported)")
        return

    print(f"Generating wave plot for {mode_name}...")
    waves = model.fc1.get_waves().detach().cpu().numpy()
    
    num_waves = waves.shape[0]
    in_dim = waves.shape[2]
    side = int(np.sqrt(in_dim)) 
    
    waves_to_plot = min(num_waves, 8)
    
    # Create a 3-row visualization
    plt.figure(figsize=(waves_to_plot*2.5, 9))
    
    for i in range(waves_to_plot):
        wave_2d = waves[i, 0, :].reshape(side, side)
        
        # Row 1: 2D Heatmap of the wave
        plt.subplot(3, waves_to_plot, i+1)
        vmax = max(np.abs(wave_2d).max(), 1e-8)  # Prevent zero range
        plt.imshow(wave_2d, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        plt.title(f"Wave {i}\n(2D)", fontsize=9)
        plt.axis('off')
        
        # Row 2: 3 Slices (Top, Mid, Bottom)
        plt.subplot(3, waves_to_plot, i+1 + waves_to_plot)
        
        # Slice indices
        idx_top = 1 # Avoid 0 just in case
        idx_mid = side // 2
        idx_bot = side - 2
        
        plt.plot(wave_2d[idx_top, :], color='red', alpha=0.6, linewidth=1, label='Top')
        plt.plot(wave_2d[idx_mid, :], color='green', alpha=0.6, linewidth=1, label='Mid')
        plt.plot(wave_2d[idx_bot, :], color='blue', alpha=0.6, linewidth=1, label='Bot')
        
        plt.title("3 Slices", fontsize=9)
        plt.grid(True, alpha=0.3)
        max_amp = max(np.abs(waves[:waves_to_plot, 0, :]).max(), 1e-8)
        plt.ylim(-max_amp, max_amp)
        if i == 0: plt.legend(fontsize=6)
        
    # Row 3: Superposition (how waves combine)
    plt.subplot(3, 1, 3)
    final_weight = np.zeros(side)
    for i in range(waves_to_plot):
        wave_2d = waves[i, 0, :].reshape(side, side)
        mid_row = wave_2d[side//2, :]
        plt.plot(mid_row, alpha=0.4, linewidth=1, label=f"Wave {i}")
        final_weight += mid_row
    
    plt.plot(final_weight, 'k-', linewidth=2, label='Sum (Final Weight)')
    plt.title(f"Superposition: How {waves_to_plot} Waves Combine", fontsize=11, fontweight='bold')
    plt.xlabel("Position along weight vector")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right', fontsize=8, ncol=3)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f"Wave Analysis for {mode_name} (Neuron 0)", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{mode_name}_waves.png", dpi=150)
    plt.close()


def plot_wave_decomposition(model, mode_name, base_dir, wave_idx=0):
    """
    Visualizes the Fourier decomposition of a single wave.
    Shows: Fourier Components → Combined Wave → 2D Heatmap
    """
    if not hasattr(model.fc1, 'get_wave_components'):
        print(f"Skipping Fourier decomposition for {mode_name} (not supported)")
        return

    print(f"Generating Fourier decomposition for {mode_name}, Wave {wave_idx}...")
    components = model.fc1.get_wave_components()
    
    if wave_idx >= len(components):
        wave_idx = 0
    
    comp_data = components[wave_idx]
    learned_coeffs = comp_data['coeffs'].cpu().numpy()  # Get learned coefficients
    harmonic_freqs = comp_data['harmonic_freqs'].cpu().numpy()  # Get frequency multipliers
    num_harmonics = len(learned_coeffs)
    
    # Get first component to determine shape
    comp1 = comp_data['comp1'].detach().cpu().numpy()
    side = int(np.sqrt(comp1.shape[1]))
    
    # Determine grid layout based on number of harmonics
    ncols = min(num_harmonics, 4)  # Max 4 columns
    
    plt.figure(figsize=(ncols * 3.5, 10))
    
    # Row 1: Individual Fourier Components (1D) - show up to 8
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    for h in range(min(num_harmonics, 8)):
        comp = comp_data[f'comp{h+1}'].detach().cpu().numpy()
        comp_2d = comp[0, :].reshape(side, side)
        mid_row = comp_2d[side//2, :]
        
        plt.subplot(3, min(num_harmonics, 8), h+1)
        label = f"{learned_coeffs[h]:.3f}·cos({int(harmonic_freqs[h])}θ)"
        plt.plot(mid_row, color=colors[h % len(colors)], linewidth=2, label=label)
        plt.title(f"Component: {label}", fontweight='bold', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.ylabel("Amplitude", fontsize=8)
        plt.legend(fontsize=7)
    
    # Row 2: Combined Wave
    # We switch to a 3x2 grid for the rest of the plot
    # Indices 1,2 are Row 1 (but we used 3xN there, so we skip to Row 2)
    # Row 2 corresponds to indices 3 and 4 in a 3x2 grid
    
    plt.subplot(3, 2, 3)
    combined_wave = sum(comp_data[f'comp{h+1}'].detach().cpu().numpy() for h in range(num_harmonics))
    combined_2d = combined_wave[0, :].reshape(side, side)
    mid_row_combined = combined_2d[side//2, :]
    
    # Plot individual components faintly
    for h in range(min(num_harmonics, 8)):
        comp = comp_data[f'comp{h+1}'].detach().cpu().numpy()
        comp_mid = comp[0, :].reshape(side, side)[side//2, :]
        plt.plot(comp_mid, alpha=0.3, linewidth=1, 
                label=f'{learned_coeffs[h]:.3f}·cos({int(harmonic_freqs[h])}θ)', 
                color=colors[h % len(colors)])
    
    plt.plot(mid_row_combined, 'k-', linewidth=2.5, label='Combined Wave')
    plt.title("Wave Superposition (1D)", fontweight='bold', fontsize=12)
    plt.xlabel("Position")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    
    # 2D Heatmap of combined wave
    plt.subplot(3, 2, 4)
    plt.imshow(combined_2d, cmap='viridis')
    plt.title("Combined Wave (2D Heatmap)", fontweight='bold', fontsize=12)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Row 3: Frequency spectrum with LEARNED amplitudes  
    plt.subplot(3, 2, 5)
    plt.stem(harmonic_freqs, learned_coeffs, basefmt=" ")
    plt.title("Learned Frequency Spectrum", fontweight='bold')
    plt.xlabel("Frequency (×ω)")
    plt.ylabel("Learned Amplitude")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Add text showing coefficients
    plt.subplot(3, 2, 6)
    plt.axis('off')
    coeff_text = f"Learned Coefficients (Wave {wave_idx}):\n\n"
    for h in range(num_harmonics):
        coeff_text += f"c_{h+1} = {learned_coeffs[h]:.4f} (×{int(harmonic_freqs[h])}ω)\n"
    coeff_text += "\nFormula:\nwave = " + " + ".join([f"c_{h+1}·cos({int(harmonic_freqs[h])}θ)" for h in range(num_harmonics)])
    
    plt.text(0.5, 0.5, coeff_text,
             ha='center', va='center', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"Fourier Decomposition: {mode_name} - Wave {wave_idx}\n({num_harmonics} Harmonics, Optimized Coefficients)", 
                 fontsize=14, fontweight='bold', y=0.995)
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(f"{base_dir}/{mode_name}_fourier_decomposition_wave{wave_idx}.png", dpi=150)
    plt.close()
