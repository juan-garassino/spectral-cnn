"""
Regression tests added by the guardian maintenance loop.

- WavePacketEmbedding must return the documented (B, T, d_model) tensor
  (it returned a raw (freqs, phases, amps) tuple, crashing WaveGPT.forward).
- CheckpointManager must load checkpoints with weights_only=True so a
  tampered .pt file cannot run arbitrary code via pickle.
"""

import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spectral_gpt'))

from spectral_gpt.wave_gpt import WaveGPT, WaveGPTConfig, WavePacketEmbedding
from monitoring import CheckpointManager


def test_wave_packet_embedding_returns_tensor():
    emb = WavePacketEmbedding(vocab_size=50, d_model=32, num_waves=8, num_harmonics=2)
    idx = torch.randint(0, 50, (2, 7))
    out = emb(idx)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 7, 32)
    # annealing path keeps the same contract
    assert emb(idx, standard_embed_ratio=0.5).shape == (2, 7, 32)


def test_wavegpt_default_forward_with_wave_embeddings():
    config = WaveGPTConfig(vocab_size=50, d_model=32, num_layers=1, num_heads=4,
                           num_waves=8, num_harmonics=2, block_size=16,
                           dropout=0.0, use_wave_embeddings=True, model_type="wave")
    model = WaveGPT(config)
    x = torch.randint(0, 50, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 50)
    assert torch.isfinite(loss)


class _Payload:
    """Pickle payload that would create a marker file when unpickled."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, 'w'))


def test_checkpoint_load_refuses_arbitrary_pickle(tmp_path):
    manager = CheckpointManager(experiment_dir=str(tmp_path), save_interval=1)
    marker = tmp_path / 'pwned'
    ckpt = manager.checkpoint_dir / 'checkpoint_step_1.pt'
    torch.save({'step': 1, 'evil': _Payload(str(marker))}, ckpt,
               pickle_protocol=pickle.DEFAULT_PROTOCOL)

    assert manager.load_checkpoint(1) is None
    assert not marker.exists()
