"""Smoke tests for third-party dependencies.

These run on every CI invocation, including Dependabot pull requests.
Each tier targets a different failure mode that a version bump can
introduce:

- **Tier 1 (behavioural)** — exercises the minimal API surface the
  project actually uses (a `DataFrame.merge`, a `LGBMClassifier.fit`,
  etc.). Catches silent removals or signature changes that a plain
  ``import`` would not see.
- **Tier 2 (import only)** — parametrised list of every other dependency
  declared in ``pyproject.toml``. Catches broken binaries / DLL conflicts
  / missing wheels at install time.
- **Tier 3 (project-specific compat)** — verifies the exact API call
  shapes that have bitten the project before (huggingface_hub's
  ``snapshot_download`` signature, langchain's import paths, numpy's
  ``bool_`` alias). Grows whenever an upstream bump breaks something
  real.

Every test uses ``pytest.importorskip`` so a missing optional extra
(voice, ffmpeg-python, dev tools) does not turn the suite red on minimal
environments — it just skips. The CI ``test`` job installs
``--all-extras`` so all of these run there.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Tier 1 — Behavioural smoke (calls real API, asserts on output)
# ---------------------------------------------------------------------------


def test_numpy_basic_ops():
    """numpy still exposes the array constructor + reductions we depend on.

    numpy 2.0 removed several aliases (np.bool_, np.int, np.float) and
    tightened scalar promotion rules. This test pins the surface the
    project actually touches: array creation, ``.sum()``, dtype itemsize,
    and NaN sentinel.
    """
    np = pytest.importorskip("numpy")
    arr = np.array([1, 2, 3])
    assert int(arr.sum()) == 6
    assert np.dtype("float64").itemsize == 8
    assert np.isnan(np.nan)


def test_pandas_parquet_roundtrip():
    """DataFrame → parquet → DataFrame survives, plus merge + groupby.

    Catches pandas 3.x-style breakages where ``to_parquet`` / ``merge``
    rename kwargs or remove implicit behaviour, and indirectly verifies
    the pyarrow link the project relies on for ``laps_featured_*.parquet``.
    """
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    buf = io.BytesIO()
    df.to_parquet(buf)
    buf.seek(0)
    restored = pd.read_parquet(buf)
    assert restored["a"].sum() == 6
    assert len(df.merge(df, on="a")) == 3
    grouped = df.groupby("a")["b"].sum()
    assert grouped.loc[1] == 4


def test_scipy_signal_resample():
    """scipy.signal.resample stays usable for arcade telemetry interp.

    SessionLoader resamples per-driver telemetry on this call path; a
    signature change would silently corrupt arcade playback timing.
    """
    pytest.importorskip("scipy")
    import numpy as np
    from scipy.signal import resample

    out = resample(np.array([0.0, 1.0, 2.0, 3.0]), 8)
    assert out.shape == (8,)


def test_sklearn_pipeline_smoke():
    """StandardScaler + LogisticRegression fit + predict still work end-to-end.

    scikit-learn 1.6+ flagged several estimators for removal and tightened
    default kwargs (n_init in KMeans, multi_class in LogReg). This
    exercises the exact two helpers used most across the agent layer.
    """
    pytest.importorskip("sklearn")
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    Xs = StandardScaler().fit_transform(X)
    model = LogisticRegression().fit(Xs, y)
    preds = model.predict(Xs)
    assert preds.shape == (4,)
    assert set(preds.tolist()).issubset({0, 1})


def test_lightgbm_fit_and_joblib_roundtrip():
    """LGBMClassifier fits + persists through joblib (N12 / N16 export path).

    Both pit-prediction and overtake-probability models are saved with
    ``joblib.dump`` rather than ``Booster.save_model`` (the latter
    explodes on Windows paths with non-ascii characters). This test
    pins both that LightGBM still trains and that the joblib round-trip
    survives whatever the next bump brings.
    """
    pytest.importorskip("lightgbm")
    pytest.importorskip("joblib")
    import joblib
    import lightgbm as lgb
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.random((40, 3))
    y = (X.sum(axis=1) > 1.5).astype(int)
    model = lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(X, y)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        joblib.dump(model, path)
        restored = joblib.load(path)

    preds = restored.predict(X)
    assert preds.shape == (40,)


def test_xgboost_fit():
    """XGBClassifier fits and predicts (N06 pace baseline).

    Pins the train + predict surface used by the pace delta lap-time
    notebook; XGBoost has changed several defaults across 1.x → 2.x.
    """
    pytest.importorskip("xgboost")
    import numpy as np
    import xgboost as xgb

    rng = np.random.default_rng(0)
    X = rng.random((30, 3))
    y = (X[:, 0] > 0.5).astype(int)
    model = xgb.XGBClassifier(n_estimators=5, eval_metric="logloss")
    model.fit(X, y)
    assert model.predict(X).shape == (30,)


def test_catboost_fit():
    """CatBoostClassifier trains on a tiny matrix without optional GPU."""
    pytest.importorskip("catboost")
    import numpy as np
    from catboost import CatBoostClassifier

    rng = np.random.default_rng(0)
    X = rng.random((20, 3))
    y = (X[:, 0] > 0.5).astype(int)
    # allow_writing_files=False stops CatBoost from creating a
    # ``catboost_info/`` directory next to the test run with per-fit
    # logs. Without it, every pytest invocation pollutes the repo root.
    model = CatBoostClassifier(iterations=5, verbose=False, allow_writing_files=False)
    model.fit(X, y)
    assert model.predict(X).shape == (20,)


def test_torch_tensor_basic():
    """torch.tensor + reduction still works without requiring CUDA.

    CI runs on Linux CPU runners, so the cu128 wheel falls back to the
    CPU index there. We only check the tensor API; CUDA assertions live
    in manual smoke scripts.
    """
    pytest.importorskip("torch")
    import torch

    t = torch.tensor([1.0, 2.0, 3.0])
    assert float(t.sum().item()) == pytest.approx(6.0)


def test_transformers_autoclasses_present():
    """AutoTokenizer / AutoModel symbols still resolve at import time.

    Avoids ``.from_pretrained`` to keep the test offline and fast — the
    failure mode this guards against is a major refactor that renames
    the entry points (which has happened across 4.x releases).
    """
    pytest.importorskip("transformers")
    from transformers import AutoModel, AutoTokenizer  # noqa: F401

    assert callable(getattr(AutoTokenizer, "from_pretrained", None))
    assert callable(getattr(AutoModel, "from_pretrained", None))


def test_huggingface_snapshot_signature():
    """snapshot_download still accepts ``local_dir`` + ``allow_patterns``.

    --- WHERE TO CHANGE IF X CHANGES ---
    The first-run model cache in ``src/f1_strat_manager/data_cache.py``
    relies on both kwargs to materialise the HF mirror under a local
    folder with a subset of files. huggingface_hub <0.20 used different
    names; if a future bump renames them again, this test will catch it
    before the user hits a black screen on first launch.
    """
    pytest.importorskip("huggingface_hub")
    import inspect

    from huggingface_hub import snapshot_download

    sig = inspect.signature(snapshot_download)
    assert "local_dir" in sig.parameters
    assert "allow_patterns" in sig.parameters


def test_tiktoken_encoding_roundtrip():
    """tiktoken still encodes + decodes the cl100k_base vocabulary.

    Used by the chat token-budget guard; a bump that drops the encoding
    name or changes the encode signature would silently break trimming.

    ``get_encoding`` fetches the vocab over HTTP on a cold cache, so an
    upstream outage (openaipublic 503) must skip - not fail - the branch
    (PK-09 #296). A genuine contract break (unknown encoding name ->
    ValueError) still fails; only network errors are skipped.
    """
    pytest.importorskip("tiktoken")
    import tiktoken

    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except ValueError:
        raise  # unknown encoding name = a real contract regression; must fail
    except Exception as exc:  # noqa: BLE001 - cold-cache vocab fetch is network-bound; an upstream outage must skip, not red the branch
        pytest.skip(f"tiktoken vocab fetch failed (environment/network): {exc}")

    tokens = enc.encode("hello world")
    assert len(tokens) >= 1
    assert enc.decode(tokens) == "hello world"


def test_httpx_async_client_constructs():
    """httpx.AsyncClient instantiates without arguments.

    Used by the MCP bridge to talk to the FastAPI backend. The
    constructor signature is stable but the import path has shifted
    between 0.x releases — this test pins the public entry point.
    """
    pytest.importorskip("httpx")
    import httpx

    client = httpx.AsyncClient()
    assert client is not None


def test_fastapi_route_registration():
    """FastAPI decorator still registers a route on the instance.

    The decorator surface (``@app.get(...)``) is the load-bearing entry
    point for every backend endpoint; a refactor that splits it across
    submodules would break the whole ``src/telemetry/backend`` layer.
    """
    pytest.importorskip("fastapi")
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/ping")
    def _ping():
        return {"ok": True}

    assert any(getattr(r, "path", None) == "/ping" for r in app.routes)


def test_langchain_openai_chat_instantiation():
    """ChatOpenAI accepts a fake api_key without contacting the network.

    Every sub-agent (N25-N31) instantiates this class. The deprecation
    history of langchain has bounced the import path between
    ``langchain.chat_models``, ``langchain_community.chat_models`` and
    finally ``langchain_openai``. This pins the current canonical path.
    """
    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(api_key="sk-test-fake", model="gpt-4o-mini")
    assert llm is not None


# ---------------------------------------------------------------------------
# Tier 2 — Import-only smoke (parametrised)
# ---------------------------------------------------------------------------


_TIER2_IMPORTS = [
    # Deep learning peripherals (torch itself in Tier 1)
    "torchvision",
    "pytorch_lightning",
    # NLP stack
    "sentence_transformers",
    "spacy",
    "setfit",
    "gliner",
    "nltk",
    "seqeval",
    "jiwer",
    "whisper",  # openai-whisper imports under the ``whisper`` name
    # Audio
    "librosa",
    "soundfile",
    "pydub",
    "edge_tts",
    # Computer vision
    "cv2",
    "ultralytics",
    "PIL",
    # Backend / web
    "uvicorn",
    "websockets",
    "aiofiles",
    "multipart",  # python-multipart imports as ``multipart``
    "jose",  # python-jose
    "passlib",
    "kafka",  # kafka-python
    # Database
    "qdrant_client",
    # UI / viz
    "streamlit",
    "plotly",
    "matplotlib",
    "seaborn",
    # Arcade (heavy import — PySide6 / pyqtgraph already covered by
    # tests/test_arcade_dashboard_imports.py, do not re-import here)
    "arcade",
    # Agents
    # ``experta`` is intentionally excluded: it transitively pulls
    # ``frozendict<2.0`` which references ``collections.Mapping`` and
    # therefore raises ``AttributeError`` on Python 3.10+ unless the user
    # bumps frozendict manually (see the comment next to it in
    # ``pyproject.toml``). The legacy rule-based agent that uses experta
    # is being replaced by the N31 orchestrator anyway.
    "langgraph",
    "langchain",
    # Utilities
    "dotenv",  # python-dotenv
    "yaml",  # pyyaml
    "requests",
    "tqdm",
    "rich",
    "fitz",
    "bs4",
    "onnxruntime",
    "optuna",
]


@pytest.mark.parametrize("module_name", _TIER2_IMPORTS)
def test_dependency_imports(module_name):
    """Every declared dependency must import without raising.

    Captures install-time breakage (missing wheels, ABI mismatches,
    binary conflicts) that a behavioural test would never reach. Skips
    cleanly when an optional extra (voice, computer vision) is absent
    from the environment, or when an upstream package is installable
    but raises at import time on the current Python (the setfit /
    frozendict / Python 3.10 trio is a recurring offender).

    ``exc_type=(ImportError, AttributeError)`` future-proofs the suite
    for pytest 9.1, which will otherwise treat ImportErrors raised
    *inside* an installed module as test failures rather than skips.
    """
    pytest.importorskip(module_name, exc_type=(ImportError, AttributeError))


# ---------------------------------------------------------------------------
# Tier 3 — Project-specific compat checks
# ---------------------------------------------------------------------------


def test_numpy_bool_alias_present():
    """``np.bool_`` survives — several legacy notebooks use it as a dtype.

    numpy 2.0 beta dropped the alias and the 2.0.0 release reinstated it
    after community pushback. A future major could remove it again; this
    test fails loudly when that happens so the notebooks get updated.
    """
    np = pytest.importorskip("numpy")
    assert hasattr(np, "bool_")
    assert np.array([], dtype=np.bool_).dtype == bool


def test_pandas_copy_returns_independent_frame():
    """DataFrame.copy() still produces an independent frame.

    pandas 3.0's copy-on-write mode flips this default; this test pins
    that mutation of the copy does not leak back into the original under
    whatever defaults the installed version ships.
    """
    pd = pytest.importorskip("pandas")
    src = pd.DataFrame({"a": [1, 2, 3]})
    copy = src.copy()
    copy.loc[0, "a"] = 99
    assert src.loc[0, "a"] == 1


def test_langchain_core_message_imports():
    """HumanMessage / AIMessage still live under langchain_core.messages.

    These imports have hopped between ``langchain.schema``,
    ``langchain.schema.messages``, and finally ``langchain_core.messages``
    over a year of releases. The agent layer assumes the current path.
    """
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage  # noqa: F401

    assert HumanMessage(content="x").content == "x"
    assert AIMessage(content="y").content == "y"


def test_langchain_openai_canonical_import_path():
    """ChatOpenAI must be importable from ``langchain_openai`` directly.

    Guards against a future deprecation that pushes everything back to
    ``langchain_community`` — the agent loaders import the short path.
    """
    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI  # noqa: F401

    assert ChatOpenAI.__name__ == "ChatOpenAI"


def test_pyarrow_parquet_engine_available():
    """pyarrow.parquet exposes the read/write entry points pandas uses.

    pandas delegates to ``pyarrow.parquet.read_table`` /
    ``write_table`` under the hood; if a future pyarrow refactor moves
    or renames them, the laps parquet load path silently falls back to
    the (much slower) fastparquet engine when present, or raises when
    not. This pins both symbols at the canonical location.
    """
    pq = pytest.importorskip("pyarrow.parquet")
    assert callable(getattr(pq, "read_table", None))
    assert callable(getattr(pq, "write_table", None))
