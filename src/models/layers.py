"""Wave-parameterized layers.

Contains ``UserWaveLinear`` (superposed waves with Fourier harmonics), baseline
layers used for comparison (``PolyLinear``, ``WaveletLinear``, ``FactorLinear``,
``SirenLinear``), ``GatedWaveLinear``, and ``SpectralConv1d/2d/3d`` which
generate convolution kernels from a ``UserWaveLinear``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 1. USER WAVE
class UserWaveLinear(nn.Module):
    """Linear layer whose weight matrix is a superposition of waves.

    Instead of storing an ``[out_dim, in_dim]`` matrix, the layer stores wave
    parameters (frequencies, phases or coordinate vectors, Fourier coefficients,
    amplitudes) and builds the weight matrix from them on each forward pass.

    Parameter shapes depend on ``wave_mode`` (``W`` = num_waves, ``H`` = num_harmonics):

    - ``outer_product``: ``u [W, out, 1]``, ``v [W, in, 1]``, ``freqs [W, 1, 1]``, ``amplitudes [W]``
    - ``fourier_series``: ``freqs [W]``, ``phases [W]``, ``amplitudes [out, W]``
    - ``gabor``: as ``fourier_series`` plus ``centers [W]`` and ``sigmas [W]``

    In all modes ``fourier_coeffs`` is ``[W, H]`` (``[W, out, H]`` if
    ``per_neuron_coeffs``), ``harmonic_freqs`` is ``[H]`` (a parameter if
    ``adaptive_freqs``, otherwise a buffer) and ``bias`` is ``[out]``.
    """
    def __init__(self, in_dim, out_dim, num_waves=12, num_harmonics=3, 
                 adaptive_freqs=False, per_neuron_coeffs=False, wave_mode="outer_product",
                 force_linear_coords=False, init_mode="standard"):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            num_waves: Number of waves to superpose
            num_harmonics: Number of Fourier components per wave (default: 3)
            adaptive_freqs: If True, harmonic frequencies are learnable (default: False, uses [1, 2, 4, 8, ...])
            per_neuron_coeffs: If True, each output neuron has its own coefficients (default: False, shared)
            wave_mode: "outer_product" (2D patterns) or "fourier_series" (1D smooth sinusoids)
            force_linear_coords: If True, forces u and v to be linear ramps [0, 1] (frozen)
            init_mode: "standard", "dft", "holographic", "standing_wave" (Physics-First strategies)
        """
        super().__init__()
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics
        self.adaptive_freqs = adaptive_freqs
        self.per_neuron_coeffs = per_neuron_coeffs
        self.wave_mode = wave_mode
        self.force_linear_coords = force_linear_coords
        self.init_mode = init_mode
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if wave_mode == "outer_product":
            # Original 2D outer product approach
            rank = 1
            if force_linear_coords:
                # Force coordinates to be linear ramps [1.0, 10.0]
                u_linspace = torch.linspace(1.0, 10.0, out_dim).reshape(1, out_dim, 1).repeat(num_waves, 1, 1)
                v_linspace = torch.linspace(1.0, 10.0, in_dim).reshape(1, in_dim, 1).repeat(num_waves, 1, 1)
                self.u = nn.Parameter(u_linspace, requires_grad=False)
                self.v = nn.Parameter(v_linspace, requires_grad=False)
            else:
                self.u = nn.Parameter(torch.randn(num_waves, out_dim, rank) * 1.0)
                self.v = nn.Parameter(torch.randn(num_waves, in_dim, rank) * 1.0)
            
            # Base frequency per wave
            if init_mode == "dft":
                # DFT Basis: Frequencies are exact harmonics
                init_freqs = torch.arange(1, num_waves + 1).float()
            else:
                init_freqs = torch.tensor([1.5**i for i in range(num_waves)]).float()
                
            if force_linear_coords:
                init_freqs *= 10.0 
            self.freqs = nn.Parameter(init_freqs.view(num_waves, 1, 1))
        
        elif wave_mode == "gabor":
            # Gabor Wavelet approach
            if init_mode == "standing_wave":
                # Standing Wave: Gaussian packets centered at regular intervals
                self.centers = nn.Parameter(torch.linspace(0, 1, num_waves))
                self.sigmas = nn.Parameter(torch.ones(num_waves) * (1.0 / num_waves)) # Narrow packets
            else:
                self.centers = nn.Parameter(torch.rand(num_waves))
                self.sigmas = nn.Parameter(torch.rand(num_waves) * 0.1 + 0.05)
            
            if init_mode == "dft":
                init_freqs = torch.arange(1, num_waves + 1).float() * 2.0 * np.pi
                self.freqs = nn.Parameter(init_freqs)
                self.phases = nn.Parameter(torch.zeros(num_waves)) # Phase 0 for DFT
            else:
                init_freqs = torch.rand(num_waves) * 10.0 + 0.5
                self.freqs = nn.Parameter(init_freqs)
                self.phases = nn.Parameter(torch.rand(num_waves) * 2 * np.pi)
        
        elif wave_mode == "fourier_series":
            # 1D Fourier series approach
            if init_mode == "dft":
                init_freqs = torch.arange(1, num_waves + 1).float()
                self.freqs = nn.Parameter(init_freqs)
                self.phases = nn.Parameter(torch.zeros(num_waves))
            else:
                init_freqs = torch.rand(num_waves) * 10.0 + 0.5
                self.freqs = nn.Parameter(init_freqs)
                self.phases = nn.Parameter(torch.rand(num_waves) * 2 * np.pi)
        
        # Harmonic multipliers
        if adaptive_freqs:
            init_harm_freqs = torch.tensor([2.0**i for i in range(num_harmonics)]).float()
            self.harmonic_freqs = nn.Parameter(init_harm_freqs)
        else:
            self.register_buffer('harmonic_freqs', torch.tensor([2.0**i for i in range(num_harmonics)]).float())
        
        # Fourier coefficients
        if per_neuron_coeffs:
            init_coeffs = torch.randn(num_waves, out_dim, num_harmonics) * 0.1
        else:
            init_coeffs = torch.randn(num_waves, num_harmonics) * 0.1
            
        if init_mode == "holographic":
            # Holographic Init: 1/f decay for coefficients (Pink Noise)
            for h in range(num_harmonics):
                scale = 1.0 / (h + 1)
                if per_neuron_coeffs:
                    init_coeffs[:, :, h] *= scale
                else:
                    init_coeffs[:, h] *= scale
        else:
            # Standard gentle init
            for h in range(num_harmonics):
                if per_neuron_coeffs:
                    init_coeffs[:, :, h] += 0.5 + (h / num_harmonics) * 0.2
                else:
                    init_coeffs[:, h] += 0.5 + (h / num_harmonics) * 0.2
                    
        self.fourier_coeffs = nn.Parameter(init_coeffs)
        
        # Amplitudes
        if wave_mode in ["fourier_series", "gabor"]:
            amp_init = torch.randn(out_dim, num_waves) * 0.1
        else:
            amp_init = torch.randn(num_waves) * 0.1
            
        if init_mode == "holographic":
            # Holographic: 1/f decay for wave amplitudes
            for w in range(num_waves):
                scale = 1.0 / (w + 1)
                if wave_mode in ["fourier_series", "gabor"]:
                    amp_init[:, w] = torch.randn(out_dim) * 0.1 * scale
                else:
                    amp_init[w] = torch.randn(1) * 0.1 * scale
        elif init_mode != "dft": # DFT keeps uniform amplitudes initially
            for w in range(num_waves):
                freq_boost = 1.0 + (w / num_waves) * 0.2
                if wave_mode in ["fourier_series", "gabor"]:
                    amp_init[:, w] += 0.08 * freq_boost
                else:
                    amp_init[w] += 0.08 * freq_boost
                    
        self.amplitudes = nn.Parameter(amp_init)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def constrain_energy(self):
        """Hamiltonian Descent: Normalize wave amplitudes."""
        with torch.no_grad():
            if hasattr(self, 'amplitudes'):
                # Normalize amplitudes across waves to conserve energy
                norm = torch.norm(self.amplitudes, p=2, dim=-1, keepdim=True)
                self.amplitudes.div_(norm + 1e-8)
    
    def stabilize_waves(self):
        """
        Stabilize wave parameters while preserving interference patterns.
        Called during training to prevent explosion while maintaining shape.
        """
        with torch.no_grad():
            # Normalize amplitudes while preserving relative ratios
            if hasattr(self, 'amplitudes'):
                norm = torch.norm(self.amplitudes)
                if norm > 10:  # Only normalize if too large
                    self.amplitudes.mul_(10 / norm)  # Scale down, preserving ratios
            
            # Keep frequencies reasonable
            if hasattr(self, 'freqs'):
                self.freqs.clamp_(-20, 20)
            
            # Wrap phases to [-π, π]
            if hasattr(self, 'phases'):
                self.phases.remainder_(2 * math.pi)
                self.phases.sub_(math.pi)  # Center at 0

    def forward(self, x):
        """Apply the wave-generated linear map.

        Args:
            x: Input of shape ``[batch, in_dim]``, float32.

        Returns:
            Tensor of shape ``[batch, out_dim]``.
        """
        if self.wave_mode == "outer_product":
            return self._forward_outer_product(x)
        elif self.wave_mode == "fourier_series":
            return self._forward_fourier_series(x)
        else:
            return self._forward_gabor(x)
    
    def _forward_outer_product(self, x):
        """Original 2D outer product approach"""
        theta_base = torch.bmm(self.u, self.v.transpose(1, 2))
        theta = theta_base * self.freqs
        W = torch.zeros(self.u.shape[1], self.v.shape[1], device=x.device)
        
        for i in range(self.num_waves):
            wave = torch.zeros_like(theta[i])
            for h in range(self.num_harmonics):
                harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                if self.per_neuron_coeffs:
                    wave = wave + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                else:
                    wave = wave + self.fourier_coeffs[i, h] * harmonic
            W = W + self.amplitudes[i] * wave
        return x @ W.t() + self.bias
    
    def _forward_fourier_series(self, x):
        """1D Fourier series approach - smooth sinusoidal basis functions"""
        # Create position indices for input dimension
        positions = torch.arange(self.in_dim, dtype=torch.float32, device=x.device) / self.in_dim
        
        # Build weight matrix [out_dim, in_dim]
        W = torch.zeros(self.out_dim, self.in_dim, device=x.device)
        
        for wave_idx in range(self.num_waves):
            # Create 1D wave along input dimension
            wave_1d = torch.zeros(self.in_dim, device=x.device)
            
            for h in range(self.num_harmonics):
                # Smooth sinusoidal component
                freq = self.freqs[wave_idx] * self.harmonic_freqs[h]
                phase = self.phases[wave_idx]
                
                if self.per_neuron_coeffs:
                    # Each output neuron has different coefficients
                    for out_idx in range(self.out_dim):
                        harmonic = torch.cos(2 * np.pi * freq * positions + phase)
                        W[out_idx, :] += (self.fourier_coeffs[wave_idx, out_idx, h] * 
                                         self.amplitudes[out_idx, wave_idx] * harmonic)
                else:
                    # Shared coefficients
                    harmonic = torch.cos(2 * np.pi * freq * positions + phase)
                    wave_1d += self.fourier_coeffs[wave_idx, h] * harmonic
            
            if not self.per_neuron_coeffs:
                # Add wave to all output neurons with learned amplitudes
                for out_idx in range(self.out_dim):
                    W[out_idx, :] += self.amplitudes[out_idx, wave_idx] * wave_1d
        
        return x @ W.t() + self.bias

    def _forward_gabor(self, x):
        """Gabor Wavelet approach - Gaussian-windowed sinusoids"""
        positions = torch.arange(self.in_dim, dtype=torch.float32, device=x.device) / self.in_dim
        W = torch.zeros(self.out_dim, self.in_dim, device=x.device)
        
        for wave_idx in range(self.num_waves):
            # Gaussian Window
            center = self.centers[wave_idx]
            sigma = torch.abs(self.sigmas[wave_idx]) + 1e-6 # Ensure positive
            gaussian = torch.exp(-((positions - center)**2) / (2 * sigma**2))
            
            wave_1d = torch.zeros(self.in_dim, device=x.device)
            
            for h in range(self.num_harmonics):
                freq = self.freqs[wave_idx] * self.harmonic_freqs[h]
                phase = self.phases[wave_idx]
                
                # Windowed Sinusoid
                harmonic = gaussian * torch.cos(2 * np.pi * freq * positions + phase)
                
                if self.per_neuron_coeffs:
                    for out_idx in range(self.out_dim):
                        W[out_idx, :] += (self.fourier_coeffs[wave_idx, out_idx, h] * 
                                         self.amplitudes[out_idx, wave_idx] * harmonic)
                else:
                    wave_1d += self.fourier_coeffs[wave_idx, h] * harmonic
            
            if not self.per_neuron_coeffs:
                for out_idx in range(self.out_dim):
                    W[out_idx, :] += self.amplitudes[out_idx, wave_idx] * wave_1d
                    
        return x @ W.t() + self.bias

    def get_weight(self):
        """Weight matrix of shape ``[out_dim, in_dim]`` (float32), computed without gradient tracking."""
        with torch.no_grad():
            if self.wave_mode == "outer_product":
                return self._get_weight_outer_product()
            elif self.wave_mode == "fourier_series":
                return self._get_weight_fourier_series()
            else:
                return self._get_weight_gabor()
    
    def _get_weight_outer_product(self):
        theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
        W = torch.zeros(self.u.shape[1], self.v.shape[1], device=self.u.device)
        for i in range(self.num_waves):
            wave = torch.zeros_like(theta[i])
            for h in range(self.num_harmonics):
                harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                if self.per_neuron_coeffs:
                    wave = wave + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                else:
                    wave = wave + self.fourier_coeffs[i, h] * harmonic
            W = W + self.amplitudes[i] * wave
        return W
    
    def _get_weight_fourier_series(self):
        positions = torch.arange(self.in_dim, dtype=torch.float32, device=self.freqs.device) / self.in_dim
        W = torch.zeros(self.out_dim, self.in_dim, device=self.freqs.device)
        
        for wave_idx in range(self.num_waves):
            wave_1d = torch.zeros(self.in_dim, device=self.freqs.device)
            for h in range(self.num_harmonics):
                freq = self.freqs[wave_idx] * self.harmonic_freqs[h]
                phase = self.phases[wave_idx]
                if self.per_neuron_coeffs:
                    for out_idx in range(self.out_dim):
                        harmonic = torch.cos(2 * np.pi * freq * positions + phase)
                        W[out_idx, :] += (self.fourier_coeffs[wave_idx, out_idx, h] * 
                                         self.amplitudes[out_idx, wave_idx] * harmonic)
                else:
                    harmonic = torch.cos(2 * np.pi * freq * positions + phase)
                    wave_1d += self.fourier_coeffs[wave_idx, h] * harmonic
            
            if not self.per_neuron_coeffs:
                for out_idx in range(self.out_dim):
                    W[out_idx, :] += self.amplitudes[out_idx, wave_idx] * wave_1d
        return W

    def _get_weight_gabor(self):
        positions = torch.arange(self.in_dim, dtype=torch.float32, device=self.freqs.device) / self.in_dim
        W = torch.zeros(self.out_dim, self.in_dim, device=self.freqs.device)
        
        for wave_idx in range(self.num_waves):
            center = self.centers[wave_idx]
            sigma = torch.abs(self.sigmas[wave_idx]) + 1e-6
            gaussian = torch.exp(-((positions - center)**2) / (2 * sigma**2))
            
            wave_1d = torch.zeros(self.in_dim, device=self.freqs.device)
            for h in range(self.num_harmonics):
                freq = self.freqs[wave_idx] * self.harmonic_freqs[h]
                phase = self.phases[wave_idx]
                harmonic = gaussian * torch.cos(2 * np.pi * freq * positions + phase)
                
                if self.per_neuron_coeffs:
                    for out_idx in range(self.out_dim):
                        W[out_idx, :] += (self.fourier_coeffs[wave_idx, out_idx, h] * 
                                         self.amplitudes[out_idx, wave_idx] * harmonic)
                else:
                    wave_1d += self.fourier_coeffs[wave_idx, h] * harmonic
            
            if not self.per_neuron_coeffs:
                for out_idx in range(self.out_dim):
                    W[out_idx, :] += self.amplitudes[out_idx, wave_idx] * wave_1d
        return W
    
    def get_waves(self):
        """Returns the individual wave components for visualization."""
        with torch.no_grad():
            if self.wave_mode == "outer_product":
                return self._get_waves_outer_product()
            elif self.wave_mode == "fourier_series":
                return self._get_waves_fourier_series()
            else:
                return self._get_waves_gabor()
    
    def _get_waves_outer_product(self):
        theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
        waves = []
        for i in range(self.num_waves):
            wave = torch.zeros_like(theta[i])
            for h in range(self.num_harmonics):
                harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                if self.per_neuron_coeffs:
                    wave = wave + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                else:
                    wave = wave + self.fourier_coeffs[i, h] * harmonic
            waves.append(wave * self.amplitudes[i])
        return torch.stack(waves)
    
    def _get_waves_fourier_series(self):
        """Get 1D Fourier series waves for visualization"""
        positions = torch.arange(self.in_dim, dtype=torch.float32, device=self.freqs.device) / self.in_dim
        waves = []
        
        for wave_idx in range(self.num_waves):
            # Build complete wave for first output neuron
            wave_1d = torch.zeros(self.in_dim, device=self.freqs.device)
            for h in range(self.num_harmonics):
                freq = self.freqs[wave_idx] * self.harmonic_freqs[h]
                phase = self.phases[wave_idx]
                harmonic = torch.cos(2 * np.pi * freq * positions + phase)
                if self.per_neuron_coeffs:
                    wave_1d += self.fourier_coeffs[wave_idx, 0, h] * harmonic
                else:
                    wave_1d += self.fourier_coeffs[wave_idx, h] * harmonic
            
            # Broadcast to match expected shape [out_dim, in_dim]
            wave_2d = wave_1d.unsqueeze(0).expand(self.out_dim, -1)
            waves.append(wave_2d * (self.amplitudes[0, wave_idx] if self.wave_mode =="fourier_series" else self.amplitudes[wave_idx]))
        
        return torch.stack(waves)

    def _get_waves_gabor(self):
        """Get Gabor waves for visualization"""
        positions = torch.arange(self.in_dim, dtype=torch.float32, device=self.freqs.device) / self.in_dim
        waves = []
        
        for wave_idx in range(self.num_waves):
            center = self.centers[wave_idx]
            sigma = torch.abs(self.sigmas[wave_idx]) + 1e-6
            gaussian = torch.exp(-((positions - center)**2) / (2 * sigma**2))
            
            wave_1d = torch.zeros(self.in_dim, device=self.freqs.device)
            for h in range(self.num_harmonics):
                freq = self.freqs[wave_idx] * self.harmonic_freqs[h]
                phase = self.phases[wave_idx]
                harmonic = gaussian * torch.cos(2 * np.pi * freq * positions + phase)
                
                if self.per_neuron_coeffs:
                    wave_1d += self.fourier_coeffs[wave_idx, 0, h] * harmonic
                else:
                    wave_1d += self.fourier_coeffs[wave_idx, h] * harmonic
            
            wave_2d = wave_1d.unsqueeze(0).expand(self.out_dim, -1)
            waves.append(wave_2d * self.amplitudes[0, wave_idx])
        
        return torch.stack(waves)
    
    def get_wave_components(self):
        """Returns the individual Fourier components for each wave."""
        with torch.no_grad():
            if self.wave_mode == "outer_product":
                return self._get_components_outer_product()
            elif self.wave_mode == "fourier_series":
                return self._get_components_fourier_series()
            else:
                return self._get_components_gabor()
    
    def _get_components_outer_product(self):
        theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
        components = []
        for i in range(self.num_waves):
            comp_dict = {}
            for h in range(self.num_harmonics):
                harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                if self.per_neuron_coeffs:
                    comp = self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic * self.amplitudes[i]
                else:
                    comp = self.fourier_coeffs[i, h] * harmonic * self.amplitudes[i]
                comp_dict[f'comp{h+1}'] = comp
            
            comp_dict['theta'] = theta[i]
            comp_dict['harmonic_freqs'] = self.harmonic_freqs.clone()
            if self.per_neuron_coeffs:
                comp_dict['coeffs'] = self.fourier_coeffs[i, 0, :].clone()
            else:
                comp_dict['coeffs'] = self.fourier_coeffs[i, :].clone()
            components.append(comp_dict)
        return components
    
    def _get_components_fourier_series(self):
        """Get Fourier series components for visualization"""
        positions = torch.arange(self.in_dim, dtype=torch.float32, device=self.freqs.device) / self.in_dim
        components = []
        
        for wave_idx in range(self.num_waves):
            comp_dict = {}
            for h in range(self.num_harmonics):
                freq = self.freqs[wave_idx] * self.harmonic_freqs[h]
                phase = self.phases[wave_idx]
                harmonic = torch.cos(2 * np.pi * freq * positions + phase)
                
                if self.per_neuron_coeffs:
                    # Use first output neuron's coefficients for visualization
                    comp = self.fourier_coeffs[wave_idx, 0, h] * harmonic * self.amplitudes[0, wave_idx]
                else:
                    comp = self.fourier_coeffs[wave_idx, h] * harmonic
                    # Scale by amplitude for first neuron
                    comp = comp * self.amplitudes[0, wave_idx]
                
                # Broadcast to match expected shape [out_dim, in_dim]
                comp_dict[f'comp{h+1}'] = comp.unsqueeze(0).expand(self.out_dim, -1)
            
            comp_dict['theta'] = None  # No theta in Fourier series mode
            comp_dict['harmonic_freqs'] = self.harmonic_freqs.clone()
            if self.per_neuron_coeffs:
                comp_dict['coeffs'] = self.fourier_coeffs[wave_idx, 0, :].clone()
            else:
                comp_dict['coeffs'] = self.fourier_coeffs[wave_idx, :].clone()
            components.append(comp_dict)
        return components

    def _get_components_gabor(self):
        """Get Gabor components for visualization"""
        positions = torch.arange(self.in_dim, dtype=torch.float32, device=self.freqs.device) / self.in_dim
        components = []
        
        for wave_idx in range(self.num_waves):
            center = self.centers[wave_idx]
            sigma = torch.abs(self.sigmas[wave_idx]) + 1e-6
            gaussian = torch.exp(-((positions - center)**2) / (2 * sigma**2))
            
            comp_dict = {}
            for h in range(self.num_harmonics):
                freq = self.freqs[wave_idx] * self.harmonic_freqs[h]
                phase = self.phases[wave_idx]
                harmonic = gaussian * torch.cos(2 * np.pi * freq * positions + phase)
                
                if self.per_neuron_coeffs:
                    comp = self.fourier_coeffs[wave_idx, 0, h] * harmonic * self.amplitudes[0, wave_idx]
                else:
                    comp = self.fourier_coeffs[wave_idx, h] * harmonic * self.amplitudes[0, wave_idx]
                
                comp_dict[f'comp{h+1}'] = comp.unsqueeze(0).expand(self.out_dim, -1)
            
            comp_dict['theta'] = None
            comp_dict['harmonic_freqs'] = self.harmonic_freqs.clone()
            if self.per_neuron_coeffs:
                comp_dict['coeffs'] = self.fourier_coeffs[wave_idx, 0, :].clone()
            else:
                comp_dict['coeffs'] = self.fourier_coeffs[wave_idx, :].clone()
            components.append(comp_dict)
        return components
    
    def get_l1_loss(self):
        """Returns L1 penalty on Fourier coefficients for sparsity."""
        return torch.abs(self.fourier_coeffs).sum()

    def constrain(self):
        """No-op; present so all layer types share the ``constrain`` interface."""
        pass
    
    def freeze_high_frequencies(self, threshold=0.5):
        """
        Freeze high-frequency harmonics for curriculum learning.
        Start with low frequencies, progressively unfreeze high frequencies.
        
        Args:
            threshold: Fraction of harmonics to keep trainable (0.5 = freeze top half)
        """
        num_to_freeze = int(self.num_harmonics * (1 - threshold))
        freeze_start = self.num_harmonics - num_to_freeze
        
        # Freeze the high-frequency part of Fourier coefficients
        if self.per_neuron_coeffs:
            # Shape: [num_waves, out_dim, num_harmonics]
            for i in range(freeze_start, self.num_harmonics):
                self.fourier_coeffs.data[:, :, i].requires_grad_(False)
        else:
            # Shape: [num_waves, num_harmonics]
            for i in range(freeze_start, self.num_harmonics):
                self.fourier_coeffs.data[:, i].requires_grad_(False)
        
        return freeze_start
    
    def unfreeze_frequencies(self, up_to_harmonic=None):
        """
        Unfreeze frequency harmonics up to a certain index.
        
        Args:
            up_to_harmonic: Index to unfreeze up to (None = unfreeze all)
        """
        if up_to_harmonic is None:
            up_to_harmonic = self.num_harmonics
        
        self.fourier_coeffs.requires_grad_(True)
    
    def progressive_unfreeze_schedule(self, current_epoch, total_epochs, strategy='linear'):
        """
        Determine which harmonics to unfreeze based on training progress.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
            total_epochs: Total number of epochs
            strategy: 'linear' or 'exponential'
        
        Returns:
            Number of harmonics that should be unfrozen
        """
        progress = (current_epoch + 1) / total_epochs
        
        if strategy == 'linear':
            # Linearly unfreeze harmonics throughout training
            num_unfrozen = max(1, int(progress * self.num_harmonics))
        elif strategy == 'exponential':
            # Start with few, rapidly unfreeze later
            num_unfrozen = max(1, int((progress ** 2) * self.num_harmonics))
        else:
            num_unfrozen = self.num_harmonics
        
        self.unfreeze_frequencies(up_to_harmonic=num_unfrozen)
        return num_unfrozen

# 2. POLY NET
class PolyLinear(nn.Module):
    """Linear layer whose weight is a sum of powers of ``tanh(u_i v_i^T)``.

    Weight ``W = sum_i coeffs[i] * tanh(u_i v_i^T) ** (i + 1)`` with ``u [W, out, 1]``,
    ``v [W, in, 1]``, ``coeffs [W]`` and ``bias [out]`` (``W`` = num_waves).
    """
    def __init__(self, in_dim, out_dim, num_waves=12):
        super().__init__()
        self.num_waves = num_waves
        rank = 1
        self.u = nn.Parameter(torch.randn(num_waves, out_dim, rank) * 1.5)
        self.v = nn.Parameter(torch.randn(num_waves, in_dim, rank) * 1.5)
        self.coeffs = nn.Parameter(torch.randn(num_waves) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        """Map ``x`` of shape ``[batch, in_dim]`` to ``[batch, out_dim]``."""
        theta = torch.bmm(self.u, self.v.transpose(1, 2))
        t = torch.tanh(theta)
        W = torch.zeros(self.u.shape[1], self.v.shape[1], device=x.device)
        for i in range(self.num_waves):
            W = W + (self.coeffs[i] * torch.pow(t[i], i+1))
        return x @ W.t() + self.bias
    def get_weight(self):
        """Weight matrix of shape ``[out_dim, in_dim]`` (float32), computed without gradient tracking."""
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2))
            t = torch.tanh(theta)
            W = torch.zeros_like(theta[0])
            for i in range(self.num_waves):
                W = W + (self.coeffs[i] * torch.pow(t[i], i+1))
        return W
    def constrain(self):
        """No-op; present so all layer types share the ``constrain`` interface."""
        pass

# 3. WAVELET NET
class WaveletLinear(nn.Module):
    """Linear layer whose weight is a sum of Gaussian-windowed cosines.

    Weight ``W = sum_i amplitudes[i] * exp(-t_i**2) * cos(5 * t_i)`` where
    ``t_i = u_i v_i^T``; parameters are ``u [W, out, 1]``, ``v [W, in, 1]``,
    ``amplitudes [W]`` and ``bias [out]``.
    """
    def __init__(self, in_dim, out_dim, num_waves=12):
        super().__init__()
        self.num_waves = num_waves
        rank = 1
        self.u = nn.Parameter(torch.randn(num_waves, out_dim, rank) * 2.0)
        self.v = nn.Parameter(torch.randn(num_waves, in_dim, rank) * 2.0)
        self.amplitudes = nn.Parameter(torch.randn(num_waves) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        """Map ``x`` of shape ``[batch, in_dim]`` to ``[batch, out_dim]``."""
        theta = torch.bmm(self.u, self.v.transpose(1, 2))
        W = torch.zeros(self.u.shape[1], self.v.shape[1], device=x.device)
        for i in range(self.num_waves):
            t = theta[i]
            envelope = torch.exp(-torch.pow(t, 2))
            oscillation = torch.cos(5.0 * t)
            W = W + (self.amplitudes[i] * envelope * oscillation)
        return x @ W.t() + self.bias
    def get_weight(self):
        """Weight matrix of shape ``[out_dim, in_dim]`` (float32), computed without gradient tracking."""
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2))
            W = torch.zeros_like(theta[0])
            for i in range(self.num_waves):
                t = theta[i]
                W = W + (self.amplitudes[i] * torch.exp(-torch.pow(t, 2)) * torch.cos(5.0*t))
        return W
    def constrain(self):
        """No-op; present so all layer types share the ``constrain`` interface."""
        pass

# 4. FACTOR NET
class FactorLinear(nn.Module):
    """Low-rank baseline: ``W = U @ V.T`` with rank ``max(2, min(in_dim, out_dim) // 2)``.

    Parameters are ``U [out, rank]``, ``V [in, rank]`` and ``bias [out]``.
    ``num_waves`` is accepted only for interface compatibility.
    """
    def __init__(self, in_dim, out_dim, num_waves=12):
        super().__init__()
        limit = min(in_dim, out_dim)
        target = max(2, limit // 2)
        self.rank = target
        k = 1.0 / np.sqrt(in_dim)
        self.U = nn.Parameter(torch.randn(out_dim, self.rank) * k)
        self.V = nn.Parameter(torch.randn(in_dim, self.rank) * k)
        self.bias = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        """Map ``x`` of shape ``[batch, in_dim]`` to ``[batch, out_dim]``."""
        W = self.U @ self.V.t()
        return x @ W.t() + self.bias
    def get_weight(self):
        """Return ``U @ V.T`` with shape ``[out_dim, in_dim]``."""
        return self.U @ self.V.t()
    def constrain(self):
        """No-op; present so all layer types share the ``constrain`` interface."""
        pass

# 5. SIREN NET
class Sine(nn.Module):
    """Sine activation ``sin(w0 * x)`` used by :class:`SirenLinear`.

    Args:
        w0: Frequency scale applied before the sine.
    """
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, input):
        """Return ``sin(w0 * input)`` with the same shape as ``input``."""
        return torch.sin(self.w0 * input)
class SirenLinear(nn.Module):
    """Linear layer whose weight matrix is predicted by a small SIREN network.

    A 3-layer sine-activated MLP (hidden width 64) maps Fourier-embedded
    ``(row, col)`` grid coordinates to one weight value per matrix entry.
    ``num_waves`` is accepted only for interface compatibility.
    """
    def __init__(self, in_dim, out_dim, num_waves=12):
        super().__init__()
        self.rows = out_dim
        self.cols = in_dim
        hidden_dim = 64
        self.L = 4
        input_dim = 2 + (2 * 2 * self.L)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), Sine(w0=30.0),
            nn.Linear(hidden_dim, hidden_dim), Sine(w0=30.0),
            nn.Linear(hidden_dim, 1)
        )
        
        # SIREN Initialization
        first = True
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                if first:
                    nn.init.uniform_(m.weight, -1 / num_input, 1 / num_input)
                    first = False
                else:
                    nn.init.uniform_(m.weight, -np.sqrt(6 / num_input) / 30.0, np.sqrt(6 / num_input) / 30.0)
                
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, out_dim), torch.linspace(-1, 1, in_dim), indexing='ij')
        coords = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)
        # Keep Fourier features as they help high frequency details even in SIRENs sometimes, 
        # but pure SIREN usually takes raw coords. 
        # Let's keep the embedding to minimize structural changes but use Sine activations.
        embeds = [coords]
        for i in range(self.L):
            freq = 2.0**i
            embeds.append(torch.sin(freq * torch.pi * coords))
            embeds.append(torch.cos(freq * torch.pi * coords))
        self.register_buffer('embedded_coords', torch.cat(embeds, dim=-1))
        self.bias = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        """Map ``x`` of shape ``[batch, in_dim]`` to ``[batch, out_dim]``."""
        W_flat = self.net(self.embedded_coords)
        W = W_flat.view(self.rows, self.cols)
        return x @ W.t() + self.bias
    def get_weight(self):
        """Weight matrix of shape ``[out_dim, in_dim]`` (float32)."""
        return self.net(self.embedded_coords).view(self.rows, self.cols)
    def constrain(self):
        """No-op; present so all layer types share the ``constrain`` interface."""
        pass

# 6. GATED WAVE NET (V14 Optimized)
class GatedWaveLinear(nn.Module):
    """Wave-synthesized weight matrix multiplied elementwise by a learned sigmoid gate.

    The signal part matches ``UserWaveLinear`` in ``outer_product`` form
    (``u``, ``v``, ``freqs``, ``fourier_coeffs``, ``amplitudes [num_waves]``).
    The gate is ``sigmoid(u_gate @ v_gate.T + gate_bias)`` with rank 2, and
    ``W = signal * gate``. ``wave_mode`` is stored but only the outer-product
    construction is implemented.
    """
    def __init__(self, in_dim, out_dim, num_waves=12, num_harmonics=3, 
                 adaptive_freqs=False, per_neuron_coeffs=False, wave_mode="outer_product",
                 force_linear_coords=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.signal_waves = num_waves # Renamed num_waves to signal_waves for consistency
        self.num_harmonics = num_harmonics
        self.adaptive_freqs = adaptive_freqs
        self.per_neuron_coeffs = per_neuron_coeffs
        self.wave_mode = wave_mode
        self.force_linear_coords = force_linear_coords
        rank = 1

        # Signal Init
        # 1. Coordinate Vectors (u, v) for Outer Product
        # We learn the "grid" itself! 
        # u: [signal_waves, out_dim, 1]
        # v: [signal_waves, in_dim, 1]
        if force_linear_coords:
            # Force coordinates to be linear ramps [0, 1]
            # This ensures rows/cols are perfect sine waves (with varying freq)
            u_linspace = torch.linspace(0, 1, out_dim).reshape(1, out_dim, 1).repeat(self.signal_waves, 1, 1)
            v_linspace = torch.linspace(0, 1, in_dim).reshape(1, in_dim, 1).repeat(self.signal_waves, 1, 1)
            self.u = nn.Parameter(u_linspace, requires_grad=False)
            self.v = nn.Parameter(v_linspace, requires_grad=False)
        else:
            # Learnable coordinates (can warp/distort)
            self.u = nn.Parameter(torch.randn(self.signal_waves, out_dim, rank) * 1.0)
            self.v = nn.Parameter(torch.randn(self.signal_waves, in_dim, rank) * 1.0)
        init_freqs = torch.tensor([1.5**i for i in range(self.signal_waves)]).float()
        self.freqs = nn.Parameter(init_freqs.view(self.signal_waves, 1, 1))

        # Harmonic multipliers (1×, 2×, 4×, 8×, ...)
        if adaptive_freqs:
            init_harm_freqs = torch.tensor([2.0**i for i in range(num_harmonics)]).float()
            self.harmonic_freqs = nn.Parameter(init_harm_freqs)
        else:
            self.register_buffer('harmonic_freqs', torch.tensor([2.0**i for i in range(num_harmonics)]).float())

        # Fourier coefficients
        if per_neuron_coeffs:
            init_coeffs = torch.randn(self.signal_waves, out_dim, num_harmonics) * 0.1
            for h in range(num_harmonics):
                init_coeffs[:, :, h] += (0.5 ** h)
            self.fourier_coeffs = nn.Parameter(init_coeffs)
        else:
            init_coeffs = torch.randn(self.signal_waves, num_harmonics) * 0.1
            for h in range(num_harmonics):
                init_coeffs[:, h] += (0.5 ** h)
            self.fourier_coeffs = nn.Parameter(init_coeffs)
        
        self.amplitudes = nn.Parameter(torch.randn(self.signal_waves) * 0.1)

        # Gate Init (Rank 2)
        self.gate_rank = 2
        self.u_gate = nn.Parameter(torch.randn(out_dim, self.gate_rank) * 0.1)
        self.v_gate = nn.Parameter(torch.randn(in_dim, self.gate_rank) * 0.1)
        self.gate_bias = nn.Parameter(torch.ones(1) * 2.0)

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        """Apply the gated wave-generated linear map.

        Args:
            x: Input of shape ``[batch, in_dim]``, float32.

        Returns:
            Tensor of shape ``[batch, out_dim]``.
        """
        theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
        signal = torch.zeros(self.u.shape[1], self.v.shape[1], device=x.device)
        
        for i in range(self.signal_waves):
            w = torch.zeros_like(theta[i])
            for h in range(self.num_harmonics):
                harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                if self.per_neuron_coeffs:
                    w = w + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                else:
                    w = w + self.fourier_coeffs[i, h] * harmonic
            signal = signal + self.amplitudes[i] * w

        gate = torch.sigmoid((self.u_gate @ self.v_gate.t()) + self.gate_bias)
        W = signal * gate
        return x @ W.t() + self.bias

    def get_weight(self):
        """Weight matrix of shape ``[out_dim, in_dim]`` (float32), computed without gradient tracking."""
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            signal = torch.zeros(self.u.shape[1], self.v.shape[1], device=self.u.device)
            for i in range(self.signal_waves):
                w = torch.zeros_like(theta[i])
                for h in range(self.num_harmonics):
                    harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                    if self.per_neuron_coeffs:
                        w = w + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                    else:
                        w = w + self.fourier_coeffs[i, h] * harmonic
                signal = signal + self.amplitudes[i] * w
            gate = torch.sigmoid((self.u_gate @ self.v_gate.t()) + self.gate_bias)
            W = signal * gate
        return W
    
    def get_waves(self):
        """Return each wave's contribution before gating, for plotting.

        Returns:
            Tensor of shape ``[num_waves, out_dim, in_dim]``.
        """
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            waves = []
            for i in range(self.signal_waves):
                w = torch.zeros_like(theta[i])
                for h in range(self.num_harmonics):
                    harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                    if self.per_neuron_coeffs:
                        w = w + self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic
                    else:
                        w = w + self.fourier_coeffs[i, h] * harmonic
                waves.append(w * self.amplitudes[i])
            return torch.stack(waves)
    
    def get_wave_components(self):
        """Returns the individual Fourier components for each wave."""
        with torch.no_grad():
            theta = torch.bmm(self.u, self.v.transpose(1, 2)) * self.freqs
            components = []
            for i in range(self.signal_waves):
                comp_dict = {}
                for h in range(self.num_harmonics):
                    harmonic = torch.cos(self.harmonic_freqs[h] * theta[i])
                    if self.per_neuron_coeffs:
                        comp = self.fourier_coeffs[i, :, h].unsqueeze(1) * harmonic * self.amplitudes[i]
                    else:
                        comp = self.fourier_coeffs[i, h] * harmonic * self.amplitudes[i]
                    comp_dict[f'comp{h+1}'] = comp
                
                comp_dict['theta'] = theta[i]
                comp_dict['harmonic_freqs'] = self.harmonic_freqs.clone()
                if self.per_neuron_coeffs:
                    comp_dict['coeffs'] = self.fourier_coeffs[i, 0, :].clone()
                else:
                    comp_dict['coeffs'] = self.fourier_coeffs[i, :].clone()
                components.append(comp_dict)
            return components
    
    def get_l1_loss(self):
        """Return the L1 norm (scalar tensor) of ``fourier_coeffs``."""
        return torch.abs(self.fourier_coeffs).sum()

    def constrain(self):
        """No-op; present so all layer types share the ``constrain`` interface."""
        pass


# 7. SPECTRAL CONV2D
class SpectralConv2d(nn.Module):
    """2D convolution whose kernel is generated by a ``UserWaveLinear`` layer.

    The kernel ``[out_channels, in_channels, kH, kW]`` is built from the flattened
    ``[out_channels, in_channels * kH * kW]`` weight of the internal ``weight_gen``.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 num_waves=12, num_harmonics=3, adaptive_freqs=False, 
                 per_neuron_coeffs=False, wave_mode="outer_product"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # We generate the weight tensor [out_channels, in_channels, kH, kW]
        # by flattening the input dimensions: [out_channels, in_channels * kH * kW]
        self.flat_in_dim = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.flat_out_dim = out_channels
        
        # Use UserWaveLinear to generate the weights
        self.weight_gen = UserWaveLinear(
            self.flat_in_dim, self.flat_out_dim, 
            num_waves=num_waves, 
            num_harmonics=num_harmonics,
            adaptive_freqs=adaptive_freqs,
            per_neuron_coeffs=per_neuron_coeffs,
            wave_mode=wave_mode
        )
        
        # Bias is handled by UserWaveLinear but we need it separate for Conv2d usually,
        # but UserWaveLinear includes a bias term in its forward/get_weight.
        # However, UserWaveLinear.get_weight() returns just the weight matrix.
        # UserWaveLinear.bias is a separate parameter.
        # We can use that bias.

    def forward(self, x):
        # 1. Generate weights [out_channels, flat_in_dim]
        """Convolve ``x`` with the wave-generated kernel.

        Args:
            x: Input of shape ``[batch, in_channels, H, W]``, float32.

        Returns:
            Output of shape ``[batch, out_channels, H', W']`` (per stride/padding).
        """
        w_flat = self.weight_gen.get_weight()
        
        # 2. Reshape to [out_channels, in_channels, kH, kW]
        weight = w_flat.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        
        # 3. Get bias
        bias = self.weight_gen.bias
        
        # 4. Convolve
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)
    
    def get_l1_loss(self):
        """Return the L1 penalty of the underlying ``UserWaveLinear``."""
        return self.weight_gen.get_l1_loss()
    
    def constrain_all(self):
        """Call ``constrain`` on the underlying layer (currently a no-op)."""
        self.weight_gen.constrain()


# 8. SPECTRAL CONV3D (Volumetric)
class SpectralConv3d(nn.Module):
    """3D convolution whose kernel is generated by a ``UserWaveLinear`` layer.

    The kernel ``[out_channels, in_channels, kD, kH, kW]`` is built from the
    flattened weight of the internal ``weight_gen``.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 num_waves=12, num_harmonics=3, adaptive_freqs=False, 
                 per_neuron_coeffs=False, wave_mode="outer_product"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Kernel size can be int or tuple (kD, kH, kW)
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        self.stride = stride
        self.padding = padding
        
        # We generate the weight tensor [out_channels, in_channels, kD, kH, kW]
        # by flattening the input dimensions: [out_channels, in_channels * kD * kH * kW]
        self.flat_in_dim = in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.flat_out_dim = out_channels
        
        # Use UserWaveLinear to generate the weights
        # This proves the "Universal Wave Computing" concept:
        # The same 1D/2D wave generation logic creates 3D volumetric filters!
        self.weight_gen = UserWaveLinear(
            self.flat_in_dim, self.flat_out_dim, 
            num_waves=num_waves, 
            num_harmonics=num_harmonics,
            adaptive_freqs=adaptive_freqs,
            per_neuron_coeffs=per_neuron_coeffs,
            wave_mode=wave_mode
        )

    def forward(self, x):
        # 1. Generate weights [out_channels, flat_in_dim]
        """Convolve ``x`` with the wave-generated kernel.

        Args:
            x: Input of shape ``[batch, in_channels, D, H, W]``, float32.

        Returns:
            Output of shape ``[batch, out_channels, D', H', W']``.
        """
        w_flat = self.weight_gen.get_weight()
        
        # 2. Reshape to [out_channels, in_channels, kD, kH, kW]
        weight = w_flat.view(self.out_channels, self.in_channels, 
                           self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
        
        # 3. Get bias
        bias = self.weight_gen.bias
        
        # 4. Convolve 3D
        return F.conv3d(x, weight, bias, stride=self.stride, padding=self.padding)
    
    def get_l1_loss(self):
        """Return the L1 penalty of the underlying ``UserWaveLinear``."""
        return self.weight_gen.get_l1_loss()
    
    def constrain_all(self):
        """Call ``constrain`` on the underlying layer (currently a no-op)."""
        self.weight_gen.constrain()


# 9. SPECTRAL CONV1D (Time Series / Audio)
class SpectralConv1d(nn.Module):
    """1D convolution whose kernel is generated by a ``UserWaveLinear`` layer.

    Defaults to ``wave_mode="fourier_series"``. The kernel
    ``[out_channels, in_channels, kW]`` is built from the flattened weight of the
    internal ``weight_gen``.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 num_waves=12, num_harmonics=3, adaptive_freqs=False, 
                 per_neuron_coeffs=False, wave_mode="fourier_series"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.kernel_size = self.kernel_size[0]
        
        self.stride = stride
        self.padding = padding
        
        # We generate the weight tensor [out_channels, in_channels, kW]
        # by flattening the input dimensions: [out_channels, in_channels * kW]
        self.flat_in_dim = in_channels * self.kernel_size
        self.flat_out_dim = out_channels
        
        # Use UserWaveLinear to generate the weights
        self.weight_gen = UserWaveLinear(
            self.flat_in_dim, self.flat_out_dim, 
            num_waves=num_waves, 
            num_harmonics=num_harmonics,
            adaptive_freqs=adaptive_freqs,
            per_neuron_coeffs=per_neuron_coeffs,
            wave_mode=wave_mode
        )

    def forward(self, x):
        # 1. Generate weights [out_channels, flat_in_dim]
        """Convolve ``x`` with the wave-generated kernel.

        Args:
            x: Input of shape ``[batch, in_channels, L]``, float32.

        Returns:
            Output of shape ``[batch, out_channels, L']``.
        """
        w_flat = self.weight_gen.get_weight()
        
        # 2. Reshape to [out_channels, in_channels, kW]
        weight = w_flat.view(self.out_channels, self.in_channels, self.kernel_size)
        
        # 3. Get bias
        bias = self.weight_gen.bias
        
        # 4. Convolve 1D
        return F.conv1d(x, weight, bias, stride=self.stride, padding=self.padding)
    
    def get_l1_loss(self):
        """Return the L1 penalty of the underlying ``UserWaveLinear``."""
        return self.weight_gen.get_l1_loss()
    
    def constrain_all(self):
        """Call ``constrain`` on the underlying layer (currently a no-op)."""
        self.weight_gen.constrain()

