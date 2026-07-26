"""Cross-cutting infrastructure package for the F1 StratLab CLI.

Hosts modules that do not fit any single domain sub-package under ``src/``
(`agents`, `nlp`, `rag`, `simulation`, …): currently just the first-run
data cache resolver and HuggingFace Hub downloader used by ``f1-strat``
and ``f1-sim`` to bootstrap the ~15 GB of models and race data when the
user installs the CLI via ``uv tool install`` without cloning the repo.
"""
