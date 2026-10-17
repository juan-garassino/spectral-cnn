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


def test_interference_attention_forward_runs():
    """G-010: forward referenced undefined freq/phase/amp/interference names."""
    from spectral_gpt.wave_gpt import InterferenceAttention
    attn = InterferenceAttention(num_waves=4, num_harmonics=2, num_heads=2, dropout=0.0)
    B, T, W, Hh = 2, 5, 4, 2
    freqs = torch.rand(B, T, W) + 0.1
    phases = torch.rand(B, T, W)
    amps = torch.rand(B, T, W, Hh)
    f, p, a = attn(freqs, phases, amps)
    assert f.shape == (B, T, W) and p.shape == (B, T, W) and a.shape == (B, T, W, Hh)
    assert torch.isfinite(f).all() and torch.isfinite(a).all()
    assert attn.last_interference.shape == (B, 2, T, T)


def test_pure_wave_block_forward_runs():
    from spectral_gpt.wave_gpt import PureWaveBlock
    config = WaveGPTConfig(vocab_size=50, d_model=32, num_layers=1, num_heads=2,
                           num_waves=4, num_harmonics=2, block_size=16, dropout=0.0)
    block = PureWaveBlock(config)
    freqs = torch.rand(1, 6, 4) + 0.1
    out = block(freqs, torch.rand(1, 6, 4), torch.rand(1, 6, 4, 2))
    assert out[0].shape == (1, 6, 4) and out[2].shape == (1, 6, 4, 2)


def test_plot_wave_packets_writes_png(tmp_path):
    """G-011: leftover plotting code referenced an undefined `t`, so the
    NameError was swallowed and the wave packet figure was never written."""
    from monitoring import VisualizationManager
    emb = WavePacketEmbedding(vocab_size=60, d_model=32, num_waves=8, num_harmonics=2)
    viz = VisualizationManager(viz_dir=str(tmp_path), viz_interval=1000)
    viz._plot_wave_packets(0, emb)
    assert (tmp_path / "wave_packets_step_0.png").exists()


def test_get_dataset_shakespeare_loads_text(tmp_path, monkeypatch):
    """G-012: `--dataset shakespeare` called a load_shakespeare() that a refactor
    had deleted, so it crashed with NameError."""
    import pytest
    pytest.importorskip("datasets")
    pytest.importorskip("tiktoken")
    import wave_experiments
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tiny_shakespeare.txt").write_text("To be, or not to be")
    monkeypatch.setattr(wave_experiments, "current_dir", str(tmp_path))
    assert wave_experiments.get_dataset("shakespeare", console=None) == "To be, or not to be"


def test_wave_packet_embedding_small_num_waves():
    """G-005: an empty frequency band reached .min() in a debug print."""
    for w in (1, 2, 3, 4):
        emb = WavePacketEmbedding(vocab_size=20, d_model=16, num_waves=w, num_harmonics=2)
        assert emb(torch.randint(0, 20, (1, 3))).shape == (1, 3, 16)
