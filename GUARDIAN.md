# GUARDIAN

Automated maintenance log for this repository (security + correctness). Branch: `claude/guardian`.

## Checks

| Area | Command |
|---|---|
| Dependency vulns (lock) | `uv export --frozen --no-hashes --no-emit-project > req.txt && pip-audit -r req.txt --no-deps --disable-pip` |
| Dependency vulns (OSV) | `osv-scanner --lockfile=requirements.txt:req.txt` (osv-scanner 1.9 cannot parse `uv.lock` directly; api.osv.dev was unreachable from the sandbox this run) |
| Static analysis | `bandit -r . -x ./tests,./.venv` (B101 asserts ignored) |
| Risky-code grep | `grep -rnE 'torch\.load\|pickle\.load\|joblib\|yaml\.load\|shell=True\|verify=False\|\beval\(\|\bexec\(\|allow_origins\|debug=True\|0\.0\.0\.0' --include=*.py .` |
| Secrets (history) | `gitleaks detect --source . --redact -v` |
| Secrets (working tree) | `gitleaks detect --no-git --source . --redact` |
| CI / containers | no `.github/workflows` and no Dockerfile present |
| Tests | Python 3.12 venv outside the repo, `torch` 2.14.1 + matplotlib/numpy/rich/pygments/tiktoken/pytest; `python -m pytest -q -p no:cacheprovider tests` |
| Lint | none configured in `pyproject.toml` |

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | medium | bug: `WavePacketEmbedding.forward` returned a `(freqs, phases, amps)` tuple instead of the documented `(B,T,d_model)` tensor, so `WaveGPT` with the default `use_wave_embeddings=True` crashed in `LayerNorm` (13 tests failing) | spectral_gpt/wave_gpt.py | fixed | 2026-10-15 |
| G-002 | medium | vuln: `eval()` on keys read from `tokenizer.json` (code execution from a data file) -> `ast.literal_eval` | spectral_gpt/prototyping/benchmark.py | fixed | 2026-10-15 |
| G-003 | medium | vuln: `torch.load` of checkpoints without `weights_only=True` (pickle code execution on older torch; `torch` is unpinned in pyproject) | spectral_gpt/monitoring.py, spectral_gpt/wave_animation.py | fixed | 2026-10-15 |
| G-004 | medium | bug: `WaveBlock` builds `InterferenceAttention(d_model, num_heads, num_waves, dropout)` positionally and calls it with one tensor, but the class now takes `(num_waves, num_harmonics, num_heads, dropout)` and `forward(freqs, phases, amps)`; `use_interference_attention=True` always crashes (11 tests failing). Needs a design decision (adapter vs. finishing the pure-wave refactor) | spectral_gpt/wave_gpt.py | open | 2026-10-15 |
| G-005 | low | bug: `WavePacketEmbedding` init crashes for small `num_waves` (< ~5) because an empty frequency band reaches `.min()` in a debug print | spectral_gpt/wave_gpt.py | open | 2026-10-15 |
| G-006 | low | hygiene: test suite overwrites tracked `results/dashboard_test/comprehensive_dashboard.png` | tests/test_dashboard.py | open | 2026-10-15 |
| G-007 | low | hygiene: `.coverage` was tracked despite being in `.gitignore` -> untracked | .coverage | fixed | 2026-10-15 |
| G-008 | low | hygiene: two ~60 MB `.pt` checkpoints committed under `spectral_gpt/benchmark_results/models/` (consider Git LFS / release assets; history not rewritten) | spectral_gpt/benchmark_results/models | open | 2026-10-15 |
| G-009 | low | supply chain: `load_dataset("HuggingFaceFW/fineweb-edu")` without a pinned `revision` | spectral_gpt/wave_experiments.py | open | 2026-10-15 |

Reviewed and judged acceptable (no ledger entry): `urllib.request.urlretrieve` of fixed https URLs (bandit B310), `subprocess.run` with list args for `git`/`pandoc` (B603/B607), `try/except/pass` (B110/B112). Secrets scans: no findings.

## Metrics

| metric | before | after |
|---|---|---|
| open vulns crit/high/med/low | 0/0/2/1 | 0/0/0/1 |
| dependency CVEs (pip-audit, 41 locked pkgs) | 0 | 0 |
| failing tests (`tests/`) | 24 | 11 |
| passing tests | 72 (7 skipped) | 88 (7 skipped; 3 new) |
| lint errors | n/a (no linter configured) | n/a |
| coverage | not measured | not measured |

## Backlog

- G-004: decide how `WaveBlock` should use the pure-wave `InterferenceAttention` (or restore an embedding-space interference attention) and make the 11 remaining tests green.
- G-005: guard the debug print / band split for small `num_waves`.
- G-006: make `tests/test_dashboard.py` write into `tmp_path`.
- Root-level `test_*.py` scripts (`test_dry_run.py`, `test_paper_generation.py`, `test_wave_packets.py`) are collected by a bare `pytest`; move them or add `testpaths = ["tests"]`.
- Pin `torch` lower bound (`>=2.6`) so `weights_only=True` is always available.

## Learnings

- `uv.lock` is not understood by osv-scanner 1.9; export via `uv export --frozen` and audit the requirements file.
- The CPU wheel index (download.pytorch.org) is not reachable from the sandbox; the PyPI `torch` wheel installs fine with `uv pip`.
- The full `tests/` suite takes ~1.5-2.5 min on CPU.

## Ideas

1. Add a minimal GitHub Actions workflow (SHA-pinned actions, `permissions: contents: read`) running `pytest tests` on CPU.
2. Add `[tool.pytest.ini_options] testpaths = ["tests"]` and ruff config to `pyproject.toml`.
3. Split `spectral_gpt/wave_gpt.py` (~2.6k lines) so experimental pure-wave classes don't break the stable `WaveGPT` path.

## Log

- 2026-10-15 run 1: baseline scans (pip-audit 0 CVEs, gitleaks clean, bandit triaged); fixed G-001 (embedding tuple crash, 13 tests green), G-002 (`eval` -> `ast.literal_eval`), G-003 (`torch.load(weights_only=True)`), untracked `.coverage`; added `tests/test_guardian_regressions.py`; left G-004 (interference attention API mismatch) open for the owner.
