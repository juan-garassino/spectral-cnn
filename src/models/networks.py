"""Network definitions: ``UniversalMLP`` (swappable layer type) and ``SpectralCNN``."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import UserWaveLinear, PolyLinear, WaveletLinear, FactorLinear, SirenLinear, GatedWaveLinear

class UniversalMLP(nn.Module):
    """Three-layer MLP (784 -> 12 -> 12 -> 10) with a selectable layer type.

    Args:
        layer_type: One of ``"Standard"``, ``"UserWave"``, ``"Poly"``, ``"Wavelet"``,
            ``"Factor"``, ``"Siren"``, ``"GatedWave"``.
        num_waves: Waves per layer (ignored for ``"Standard"``).
        num_harmonics: Fourier harmonics per wave (``UserWave``/``GatedWave`` only).
        adaptive_freqs: Make harmonic frequencies learnable (``UserWave``/``GatedWave``).
        per_neuron_coeffs: One coefficient set per output neuron (``UserWave``/``GatedWave``).
        wave_mode: ``"outer_product"``, ``"fourier_series"`` or ``"gabor"``.
    """
    def __init__(self, layer_type, num_waves=12, num_harmonics=3, 
                 adaptive_freqs=False, per_neuron_coeffs=False, wave_mode="outer_product"):
        super().__init__()
        if layer_type == "UserWave": Layer = UserWaveLinear
        elif layer_type == "Poly":   Layer = PolyLinear
        elif layer_type == "Wavelet":Layer = WaveletLinear
        elif layer_type == "Factor": Layer = FactorLinear
        elif layer_type == "Siren":  Layer = SirenLinear
        elif layer_type == "GatedWave": Layer = GatedWaveLinear
        elif layer_type == "Standard":Layer = nn.Linear

        HIDDEN = 12
        if layer_type == "Standard":
            self.fc1 = nn.Linear(28*28, HIDDEN)
            self.fc2 = nn.Linear(HIDDEN, HIDDEN)
            self.fc3 = nn.Linear(HIDDEN, 10)
        elif layer_type in ["UserWave", "GatedWave"]:
            # Spectral layers with Fourier configuration
            self.fc1 = Layer(28*28, HIDDEN, num_waves=num_waves, num_harmonics=num_harmonics,
                           adaptive_freqs=adaptive_freqs, per_neuron_coeffs=per_neuron_coeffs,
                           wave_mode=wave_mode)
            self.fc2 = Layer(HIDDEN, HIDDEN, num_waves=num_waves, num_harmonics=num_harmonics,
                           adaptive_freqs=adaptive_freqs, per_neuron_coeffs=per_neuron_coeffs,
                           wave_mode=wave_mode)
            self.fc3 = Layer(HIDDEN, 10, num_waves=num_waves, num_harmonics=num_harmonics,
                           adaptive_freqs=adaptive_freqs, per_neuron_coeffs=per_neuron_coeffs,
                           wave_mode=wave_mode)
        else:
            self.fc1 = Layer(28*28, HIDDEN, num_waves=num_waves)
            self.fc2 = Layer(HIDDEN, HIDDEN, num_waves=num_waves)
            self.fc3 = Layer(HIDDEN, 10, num_waves=num_waves)
        self.type = layer_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify flattened images.

        Args:
            x: Input of shape ``[batch, 1, 28, 28]`` (or any shape that flattens to 784 per sample).

        Returns:
            Logits of shape ``[batch, 10]``.
        """
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def constrain_all(self):
        """Call ``constrain`` on each layer for wave-based types (currently a no-op); does nothing otherwise."""
        if self.type in ["UserWave", "GatedWave"]:
            self.fc1.constrain(); self.fc2.constrain(); self.fc3.constrain()
    
    def get_l1_loss(self):
        """Returns total L1 penalty from all layers (for sparsity regularization)."""
        if self.type not in ["UserWave", "GatedWave"]:
            return 0.0
        return self.fc1.get_l1_loss() + self.fc2.get_l1_loss() + self.fc3.get_l1_loss()

    def get_first_layer_weight(self):
        """Return the first layer's weight matrix, shape ``[12, 784]``."""
        if self.type == "Standard": return self.fc1.weight
        return self.fc1.get_weight()

class SpectralCNN(nn.Module):
    """Small MNIST CNN built from spectral layers.

    Two ``SpectralConv2d`` layers (1 -> 16 -> 32 channels, 5x5 kernels, padding 2,
    each followed by ReLU and 2x2 max-pool) and a ``UserWaveLinear`` classifier
    (1568 -> 10).

    Args:
        num_waves: Waves per layer.
        num_harmonics: Fourier harmonics per wave.
        adaptive_freqs: Make harmonic frequencies learnable.
        per_neuron_coeffs: One coefficient set per output neuron.
        wave_mode: ``"outer_product"``, ``"fourier_series"`` or ``"gabor"``.
    """
    def __init__(self, num_waves=12, num_harmonics=3, 
                 adaptive_freqs=False, per_neuron_coeffs=False, wave_mode="outer_product"):
        super().__init__()
        from .layers import SpectralConv2d, UserWaveLinear
        
        self.type = "SpectralCNN"
        
        # Spectral Convolutional Layers
        # Input: 1x28x28
        self.conv1 = SpectralConv2d(1, 16, kernel_size=5, stride=1, padding=2,
                                  num_waves=num_waves, num_harmonics=num_harmonics,
                                  adaptive_freqs=adaptive_freqs, per_neuron_coeffs=per_neuron_coeffs,
                                  wave_mode=wave_mode)
        # Output: 16x28x28 -> MaxPool -> 16x14x14
        
        self.conv2 = SpectralConv2d(16, 32, kernel_size=5, stride=1, padding=2,
                                  num_waves=num_waves, num_harmonics=num_harmonics,
                                  adaptive_freqs=adaptive_freqs, per_neuron_coeffs=per_neuron_coeffs,
                                  wave_mode=wave_mode)
        # Output: 32x14x14 -> MaxPool -> 32x7x7
        
        # Fully Connected Layer (Spectral)
        # Input: 32*7*7 = 1568
        self.fc1 = UserWaveLinear(32*7*7, 10, 
                                num_waves=num_waves, num_harmonics=num_harmonics,
                                adaptive_freqs=adaptive_freqs, per_neuron_coeffs=per_neuron_coeffs,
                                wave_mode=wave_mode)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv 1
        """Classify images.

        Args:
            x: Input reshapeable to ``[batch, 1, 28, 28]``, float32.

        Returns:
            Logits of shape ``[batch, 10]``.
        """
        x = x.view(-1, 1, 28, 28) # Ensure 4D input
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Conv 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC
        x = self.fc1(x)
        return x

    def constrain_all(self):
        """Call ``constrain`` on every spectral layer (currently a no-op)."""
        self.conv1.constrain_all()
        self.conv2.constrain_all()
        self.fc1.constrain()
    
    def get_l1_loss(self):
        """Return the summed L1 penalty (scalar tensor) over all spectral layers."""
        return self.conv1.get_l1_loss() + self.conv2.get_l1_loss() + self.fc1.get_l1_loss()
    
    def get_first_layer_weight(self):
        # Return first conv layer weights for visualization
        # Shape: [16, 1, 5, 5]
        """Return the first conv kernel, shape ``[16, 1, 5, 5]``, for visualization."""
        return self.conv1.weight_gen.get_weight().view(16, 1, 5, 5)
