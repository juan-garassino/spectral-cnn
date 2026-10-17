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
| Tests | venv outside the repo (scratchpad), `torch` 2.14.1 + torchvision + matplotlib/numpy 2.x/rich/pygments/tiktoken/datasets/pytest; `python -m pytest -q -p no:cacheprovider` (`testpaths = ["tests"]` since run 2) |
| Lint | none configured in `pyproject.toml`; guardian runs `ruff check --isolated --select F821 .` (undefined names = latent crashes) and tracks the total of `ruff check --isolated .` |

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | medium | bug: `WavePacketEmbedding.forward` returned a `(freqs, phases, amps)` tuple instead of the documented `(B,T,d_model)` tensor, so `WaveGPT` with the default `use_wave_embeddings=True` crashed in `LayerNorm` (13 tests failing) | spectral_gpt/wave_gpt.py | fixed | 2026-10-15 |
| G-002 | medium | vuln: `eval()` on keys read from `tokenizer.json` (code execution from a data file) -> `ast.literal_eval` | spectral_gpt/prototyping/benchmark.py | fixed | 2026-10-15 |
| G-003 | medium | vuln: `torch.load` of checkpoints without `weights_only=True` (pickle code execution on older torch; `torch` is unpinned in pyproject) | spectral_gpt/monitoring.py, spectral_gpt/wave_animation.py | fixed | 2026-10-15 |
| G-004 | medium | bug: `WaveBlock` builds `InterferenceAttention(d_model, num_heads, num_waves, dropout)` positionally and calls it with one tensor, but the class now takes `(num_waves, num_harmonics, num_heads, dropout)` and `forward(freqs, phases, amps)`; `use_interference_attention=True` always crashes (11 tests failing). Needs a design decision (adapter vs. finishing the pure-wave refactor) | spectral_gpt/wave_gpt.py | open | 2026-10-15 |
| G-005 | low | bug: `WavePacketEmbedding` init crashes for small `num_waves` (< ~5) because an empty frequency band reaches `.min()` in a debug print | spectral_gpt/wave_gpt.py | fixed | 2026-10-17 |
| G-006 | low | hygiene: test suite overwrites tracked `results/dashboard_test/comprehensive_dashboard.png` -> writes to `tmp_path` and asserts the png exists | tests/test_dashboard.py | fixed | 2026-10-17 |
| G-007 | low | hygiene: `.coverage` was tracked and not ignored -> untracked + added to `.gitignore` | .coverage | fixed | 2026-10-15 |
| G-008 | low | hygiene: two ~60 MB `.pt` checkpoints committed under `spectral_gpt/benchmark_results/models/` (consider Git LFS / release assets; history not rewritten) | spectral_gpt/benchmark_results/models | open | 2026-10-15 |
| G-009 | low | supply chain: `load_dataset("HuggingFaceFW/fineweb-edu")` without a pinned `revision` | spectral_gpt/wave_experiments.py | open | 2026-10-15 |

| G-010 | medium | bug: `InterferenceAttention.forward` used undefined names (`freq`, `phase`, `amp`, `interference`) left over from a rename, so every call (incl. `PureWaveBlock`) raised `NameError` | spectral_gpt/wave_gpt.py | fixed | 2026-10-17 |
| G-011 | low | bug: `VisualizationManager._plot_wave_packets` had leftover code plotting an undefined `t`; the `NameError` was swallowed, so the wave-packet figure was never written (and the figure leaked) | spectral_gpt/monitoring.py | fixed | 2026-10-17 |
| G-012 | medium | bug: `get_dataset("shakespeare")` (`--dataset shakespeare`, and the default of `run_experiments(dataset=...)`) called `load_shakespeare`, which a refactor had deleted -> `NameError`; restored | spectral_gpt/wave_experiments.py | fixed | 2026-10-17 |
| G-013 | low | bug: `EXPERIMENTS` dict literal defines the key `"pure_wave"` twice; the first config ("PureWaveAttn (ELU+1)", `model_type="wave"`) is silently shadowed and unreachable. Needs the owner to pick a new key name | spectral_gpt/wave_experiments.py | open | 2026-10-17 |
| G-014 | low | bug: `demo_full_monitoring_system` printed `tracker.*` (undefined; variable is `config_tracker`) -> `NameError` at the end of the demo; also removed unreachable dead code with undefined names after a `return` in `PhaseInterferometerGate.forward` | spectral_gpt/demo_monitoring.py, spectral_gpt/wave_gpt.py | fixed | 2026-10-17 |

Reviewed and judged acceptable (no ledger entry): `urllib.request.urlretrieve` of fixed https URLs (bandit B310), `subprocess.run` with list args for `git`/`pandoc` (B603/B607), `try/except/pass` (B110/B112). Secrets scans: no findings.

## Metrics

| metric | before | after |
|---|---|---|
| open vulns crit/high/med/low (run 1) | 0/0/2/1 | 0/0/0/1 |
| open vulns crit/high/med/low (run 2) | 0/0/0/1 | 0/0/0/1 |
| dependency CVEs (pip-audit, lock export) | 0 | 0 |
| secrets (gitleaks, full history 57 commits + tree) | 0 | 0 |
| failing tests (run 1) | 24 | 11 |
| failing tests (run 2) | 11 | 11 (all G-004) |
| passing tests (run 2) | 88 (7 skipped) | 93 (7 skipped; 5 new) |
| ruff F821 undefined names (run 2) | 19 | 0 |
| ruff `--isolated` total (run 2) | 733 | 713 |
| coverage | not measured | not measured |

## Backlog

- G-004: decide how `WaveBlock` should use the pure-wave `InterferenceAttention` (or restore an embedding-space interference attention) and make the 11 remaining tests green.
- G-013: rename the shadowed first `"pure_wave"` experiment key (owner picks the name).
- Root-level `test_*.py` scripts are no longer collected (`testpaths` added in run 2); consider moving them to `demo/`.
- `StandardEmbedding` / `StandardBlock` are defined twice in `wave_gpt.py` (ruff F811); the second definitions win - dedupe.
- Pin `torch` lower bound (`>=2.6`) so `weights_only=True` is always available.

## Learnings

- `uv.lock` is not understood by osv-scanner 1.9; export via `uv export --frozen` and audit the requirements file.
- The CPU wheel index (download.pytorch.org) is not reachable from the sandbox; the PyPI `torch` wheel installs fine with `uv pip`.
- The full `tests/` suite takes ~1-2.5 min on CPU.
- `ruff --select F821` is the highest-yield bug finder here: half-finished refactors leave undefined names that only explode at runtime (and some are swallowed by broad `try/except`).
- `datasets` needs numpy 2.x + a recent pyarrow; the lockfile pins numpy 2.5.x, so install that rather than numpy 1.x.
- The CPU wheel index was again unreachable (proxy tunnel error) in run 2; PyPI `torch` works.

## Ideas

1. Add a minimal GitHub Actions workflow (SHA-pinned actions, `permissions: contents: read`) running `pytest tests` on CPU.
2. Add a `[tool.ruff]` config to `pyproject.toml` (`testpaths` done in run 2).
3. Split `spectral_gpt/wave_gpt.py` (~2.6k lines) so experimental pure-wave classes don't break the stable `WaveGPT` path.
4. Add a CI lint gate limited to `ruff --select F821,F601,F811` so undefined names/duplicate keys cannot land again.
5. Replace broad `except Exception: print(...)` in `VisualizationManager` with logging that includes the traceback (it hid G-011).

## Log

- 2026-10-15 run 1: baseline scans (pip-audit 0 CVEs, gitleaks clean, bandit triaged); fixed G-001 (embedding tuple crash, 13 tests green), G-002 (`eval` -> `ast.literal_eval`), G-003 (`torch.load(weights_only=True)`), untracked `.coverage`; added `tests/test_guardian_regressions.py`; left G-004 (interference attention API mismatch) open for the owner.
- 2026-10-17 run 2: rescans clean (pip-audit 0 CVEs, gitleaks history+tree 0, no workflows/Dockerfiles, bandit unchanged); ruff F821 sweep found 5 latent crashes -> fixed G-010 (InterferenceAttention NameError), G-011 (wave-packet plot never written), G-012 (missing `load_shakespeare`), G-014 (demo NameError + dead code), plus G-005 (small `num_waves`) and G-006 (test writes into `tmp_path`); added `testpaths`; opened G-013 (duplicate `"pure_wave"` key). G-004 still open for the owner. No review comments on the PR to address.
