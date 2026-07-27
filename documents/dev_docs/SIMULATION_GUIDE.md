# F1 StratLab — Dev Testing & Simulation Guide

Quick reference for running, testing, and verifying every layer of the stack.

---

## 0. Prerequisites

```powershell
# Activate your Python env from repo root
# (adjust path to your actual venv/conda)
.\.venv\Scripts\Activate.ps1
```

**LM Studio** — must be running on `http://localhost:1234` for any LLM step.
Load a model (e.g. Llama-3-8B or equivalent) before running with-LLM tests.

---

## 1. Agent unit smoke (no LLM, no GPU required)

### Fast import check

```powershell
python -c "from src.agents.pace_agent import run_pace_agent_from_state; print('pace OK')"
python -c "from src.agents.tire_agent import run_tire_agent_from_state; print('tire OK')"
python -c "from src.agents.race_situation_agent import run_race_situation_agent_from_state; print('situation OK')"
python -c "from src.agents.pit_strategy_agent import run_pit_strategy_agent_from_state; print('pit OK')"
python -c "from src.agents.radio_agent import run_radio_agent_from_state; print('radio OK')"
python -c "from src.agents.rag_agent import run_rag_agent; print('rag OK')"
python -c "from src.agents.strategy_orchestrator import RaceState, run_strategy_orchestrator_from_state; print('orchestrator OK')"
```

### pytest suites

```powershell
# Agent imports + Pydantic schemas + voice config
pytest tests/agents/test_agents.py -v

# Repo structure + simulation imports + Melbourne parquet
pytest tests/infra/test_smoke.py -v

# All at once
pytest tests/ -v
```

---

## 2. Single-agent debug (`debug_agent.py`)

Runs one agent in isolation using real Melbourne 2025 lap data (falls back to synthetic defaults if parquet not found).

### Pace agent (N25)

```powershell
python scripts/debug_agent.py --agent pace --gp Melbourne --lap 20 --driver NOR --team McLaren
```

### Tire agent (N26)

```powershell
python scripts/debug_agent.py --agent tire --gp Melbourne --lap 20 --driver NOR --team McLaren

# Force specific tyre state
python scripts/debug_agent.py --agent tire --gp Melbourne --lap 35 --driver NOR --team McLaren `
    --override tyre_life=28 compound=MEDIUM position=2
```

### Race situation agent (N27)

```powershell
python scripts/debug_agent.py --agent situation --gp Melbourne --lap 20 --driver NOR --team McLaren
```

### Pit strategy agent (N28)

```powershell
python scripts/debug_agent.py --agent pit --gp Melbourne --lap 28 --driver NOR --team McLaren

# Print full lap_state before running
python scripts/debug_agent.py --agent pit --gp Melbourne --lap 28 --driver NOR --team McLaren --print-state

# Override compound
python scripts/debug_agent.py --agent pit --gp Bahrain --lap 20 --driver NOR --team McLaren `
    --override compound=MEDIUM tyre_life=18
```

### Radio agent (N29)

```powershell
# With explicit radio message (exercises full NLP path)
python scripts/debug_agent.py --agent radio --gp Melbourne --lap 10 --driver NOR --team McLaren `
    --radio "Box box, tyres are gone completely"

# No radio — exercises RCM path only
python scripts/debug_agent.py --agent radio --gp Bahrain --lap 25 --driver HAM --team Mercedes
```

### RAG agent (N30) — requires LM Studio

```powershell
# Auto-generated query from gp_name
python scripts/debug_agent.py --agent rag --gp Melbourne --lap 20 --driver NOR --team McLaren

# Custom regulation question
python scripts/debug_agent.py --agent rag --gp Monaco --lap 50 --driver LEC --team Ferrari `
    --query "Can a driver change tyres twice in the same lap under VSC?"
```

### Full orchestrator (N31) — requires LM Studio

```powershell
python scripts/debug_agent.py --agent orchestrator --gp Melbourne --lap 20 --driver NOR --team McLaren
```

---

## 3. `--override` reference

Overrides any field in the `driver` dict of the lap state without touching the rest.

```powershell
--override tyre_life=28 compound=MEDIUM position=3 lap_time_s=88.5
```

Supported fields: `tyre_life`, `compound`, `compound_id`, `position`, `lap_time_s`,
`speed_st`, `fuel_load`, `stint`, `fresh_tyre`, `gap_to_leader_s`.

---

## 4. Full system CLI simulation (`run_simulation_cli.py`)

Renders a live Rich table (lap-by-lap) using the full RaceReplayEngine pipeline.

### Without LLM — ML + MC scores only (fast)

```powershell
# All laps, Melbourne, NOR
python scripts/run_simulation_cli.py Melbourne NOR McLaren --no-llm

# Specific lap range
python scripts/run_simulation_cli.py Melbourne NOR McLaren --laps 15-35 --no-llm

# Another GP
python scripts/run_simulation_cli.py Bahrain HAM Mercedes --laps 20-40 --no-llm

# With full tracebacks on error
python scripts/run_simulation_cli.py Silverstone NOR McLaren --laps 1-57 --no-llm --verbose
```

### With LLM synthesis — requires LM Studio

```powershell
python scripts/run_simulation_cli.py Melbourne NOR McLaren --laps 20-30

# Custom parquet paths
python scripts/run_simulation_cli.py Monaco LEC Ferrari --laps 50-78 `
    --raw-dir data/raw/2025 `
    --featured data/processed/laps_featured_2025.parquet
```

### Real team-radio corpus (default) vs legacy mock injection

By default every run feeds the N29 Radio Agent from the static OpenF1
corpus built by `scripts/build_radio_dataset.py`. At startup the CLI
calls `ensure_radio_corpus(year, gp_name)` in
`src/f1_strat_manager/data_cache.py` to lazily download the per-GP MP3
tree (~3 MB/race) from the HuggingFace Dataset if it is not already on
disk, then builds a `RadioPipelineRunner`
(`src/nlp/radio_runner.py`) that transcribes per-lap slices with Whisper
on demand. Transcripts land in
`data/processed/radio_nlp/{year}/{slug}/transcripts.json`, keyed by
Whisper model name so switching `--whisper-model` re-transcribes cleanly.
The run header advertises `radios=N` and the summary panel shows a
`radio src corpus/Nr·Mrcm` row so you can tell where the radios came
from. End-to-end validation lives in
[`notebooks/agents/N34_radio_runner_smoke.ipynb`](../../notebooks/agents/N34_radio_runner_smoke.ipynb).

```powershell
# Skip the real corpus — legacy mock radios only (fastest path)
python scripts/run_simulation_cli.py Bahrain NOR McLaren --laps 1-10 --no-real-radios

# Override the Whisper model (default: turbo; also tiny / base / small / medium / large)
python scripts/run_simulation_cli.py Bahrain NOR McLaren --laps 1-10 --whisper-model small
```

`--radio-every` still works on top of the real corpus, so stress tests
that need extra mock radios keep layering them over the OpenF1 stream.

**Output columns:**
```
Lap | Cmpd | Life | Action | Conf | STAY / PIT / UDCT / OVCT | Reasoning
```

---

## 5. Telemetry backend — local deployment (sin Docker)

Run from inside the submodule directory (`src/telemetry/`).

### Install dependencies

```powershell
cd src\telemetry

# New voice deps (Nemotron + Qwen3-TTS) — add to requirements
pip install transformers accelerate
pip install qwen-tts soundfile

# Existing backend deps
pip install -r backend/requirements.txt
```

> **Note:** `requirements.txt` still lists `openai-whisper` and `edge-tts` — estas líneas
> son ahora dead code. Puedes borrarlas manualmente o simplemente ignorarlas;
> el backend ya no las importa.

### Start backend (uvicorn)

```powershell
cd src\telemetry
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start frontend (Streamlit)

```powershell
# Separate terminal
cd src\telemetry
streamlit run frontend/app/main.py --server.port 8501
```

### Docker (backend + frontend juntos)

```powershell
cd src\telemetry
docker-compose up --build
```

---

## 6. Strategy endpoints — curl/PowerShell verification

Con el backend corriendo en `http://localhost:8000`.

### Pace endpoint (N25)

```powershell
$body = '{"lap_state":{"driver":{"driver":"NOR","position":1,"compound":"SOFT","tyre_life":10,"lap_number":20,"lap_time_s":90.5,"speed_st":310,"fuel_load":0.6,"stint":1,"fresh_tyre":false,"gap_to_leader_s":0},"session_meta":{"total_laps":57,"gp_name":"Melbourne","year":2025},"weather":{"air_temp":22,"track_temp":35,"rainfall":false},"rivals":[]}}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/strategy/pace" -Method POST -ContentType "application/json" -Body $body
```

### Tire endpoint (N26)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/strategy/tire" -Method POST -ContentType "application/json" -Body $body
```

### Situation endpoint (N27)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/strategy/situation" -Method POST -ContentType "application/json" -Body $body
```

### Pit endpoint (N28)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/strategy/pit" -Method POST -ContentType "application/json" -Body $body
```

### Radio endpoint (N29)

```powershell
$radioBody = '{"lap_state":{"driver":{"driver":"NOR","position":1,"compound":"SOFT","tyre_life":10,"lap_number":20,"lap_time_s":90.5,"speed_st":310,"fuel_load":0.6,"stint":1,"fresh_tyre":false,"gap_to_leader_s":0},"session_meta":{"total_laps":57,"gp_name":"Melbourne","year":2025},"weather":{"air_temp":22,"track_temp":35,"rainfall":false},"rivals":[]},"radio_msgs":[{"driver":"NOR","timestamp":0,"text":"Box box, tyres are gone","team_radio":true}],"rcm_events":[]}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/strategy/radio" -Method POST -ContentType "application/json" -Body $radioBody
```

### RAG endpoint (N30) — requires LM Studio

```powershell
$ragBody = '{"question":"What are the rules for pit stop minimum time?"}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/strategy/rag" -Method POST -ContentType "application/json" -Body $ragBody
```

### Full orchestrator — /recommend (N31) — requires LM Studio

```powershell
$recBody = '{"lap_state":{"driver":{"driver":"NOR","position":1,"compound":"SOFT","tyre_life":18,"lap_number":28,"lap_time_s":91.2,"speed_st":308,"fuel_load":0.4,"stint":1,"fresh_tyre":false,"gap_to_leader_s":0},"session_meta":{"total_laps":57,"gp_name":"Melbourne","year":2025},"weather":{"air_temp":22,"track_temp":35,"rainfall":false},"rivals":[{"driver":"PIA","position":2,"interval_to_driver_s":2.1}]},"gap_ahead_s":2.1,"risk_tolerance":0.5}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/strategy/recommend" -Method POST -ContentType "application/json" -Body $recBody
```

---

## 7. Voice endpoints — verificación

### Health check (no model load needed)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/voice/health" -Method GET
```

### STT service import check (Nemotron — no GPU needed para el import)

```powershell
cd src\telemetry
python -c "from backend.services.voice.stt_service import STTService; print('STT import OK')"
python -c "from backend.services.voice.tts_service import TTSService; print('TTS import OK')"
```

### Voice config — verificar que Whisper/EdgeTTS están eliminados

```powershell
cd src\telemetry
python -c "
from backend.core.voice_config import NEMOTRON_MODEL, QWEN3_TTS_MODEL, QWEN3_SAMPLE_RATE
print('Nemotron model:', NEMOTRON_MODEL)
print('Qwen3-TTS model:', QWEN3_TTS_MODEL)
print('Sample rate:', QWEN3_SAMPLE_RATE)
import backend.core.voice_config as vc
assert not hasattr(vc, 'WHISPER_MODEL'), 'ERROR: WHISPER_MODEL still present'
assert not hasattr(vc, 'TTS_ENGINE'), 'ERROR: TTS_ENGINE still present'
print('Config clean — no Whisper/EdgeTTS constants')
"
```

### Transcribe audio (Nemotron) — requiere GPU o CPU lento

```powershell
# Graba un WAV y pásalo al endpoint
$audioBytes = [System.IO.File]::ReadAllBytes("test_audio.wav")
$b64 = [Convert]::ToBase64String($audioBytes)
$sttBody = "{`"audio_base64`":`"$b64`"}"
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/voice/transcribe" -Method POST -ContentType "application/json" -Body $sttBody
```

### TTS synthesis (Qwen3-TTS) — requiere model download en primer uso

```powershell
$ttsBody = '{"text":"Box box, pit this lap. Undercut window is open."}'
$response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/voice/synthesize" -Method POST -ContentType "application/json" -Body $ttsBody -OutFile "test_tts_output.wav"
# Reproduce test_tts_output.wav para verificar
```

---

## 8. Chat endpoint — verificar routing de tool MCP (Step 9b)

```powershell
# Strategy query — el LLM debe elegir predict_pit / predict_tire / recommend_strategy via MCP
$chatBody = '{"text":"Should NOR pit this lap? He is on lap 22 with MEDIUMS in Monza."}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/chat/tool-message" -Method POST -ContentType "application/json" -Body $chatBody

# Telemetry comparison — el LLM debe elegir compare_drivers (Phase 2 OpenAPI tool)
$chatBody2 = '{"text":"compare VER vs HAM telemetry at Monza 2024"}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/chat/tool-message" -Method POST -ContentType "application/json" -Body $chatBody2
```

---

## 9. N32 Smoke test notebook

Ejecuta todos los agentes (N25–N31) + CLI de 3 vueltas via subprocess.

```powershell
# En Jupyter
jupyter notebook notebooks/agents/N32_smoke_test.ipynb
# → Kernel > Restart & Run All
# Criterio: todas las celdas exit code 0, sin [ERROR] en ningún output
```

---

## 10. Tips

- **Sin parquet Melbourne?** `debug_agent.py` cae a valores sintéticos y avisa. La CLI aborta si no hay `--raw-dir`.
- **Startup lento (~20-40s)?** Normal — NLP models (RoBERTa, BERT-large, SetFit) se cargan en memoria la primera vez que se importa `radio_agent`.
- **LM Studio no corre?** Usa `--no-llm` en la CLI, o evita `--agent orchestrator`/`--agent rag` en debug.
- **Nemotron/Qwen3-TTS — primer arranque lento?** Los modelos se descargan de HuggingFace en el primer uso (~1-3 GB). Después quedan en caché local (`~/.cache/huggingface/`).
- **GPU no disponible?** Cambia `NEMOTRON_DEVICE = 0` → `NEMOTRON_DEVICE = "cpu"` en `voice_config.py`. Nemotron funciona en CPU pero más lento (~500ms/chunk vs ~80ms GPU).
