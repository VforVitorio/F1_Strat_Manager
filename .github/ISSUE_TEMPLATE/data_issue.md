---
name: Data issue
about: CSV / parquet mismatches, model prediction outliers, FastF1 / OpenF1 data anomalies.
title: "[DATA] "
labels: data
---

## Data issue

**Type:** <!-- CSV output mismatch / Model prediction outlier / FastF1 loading error / OpenF1 radio drift / Parquet schema -->

### Data problem
<!-- Describe what is wrong with the numbers. -->

### Dataset pointer
<!-- e.g. `data/processed/laps_featured_2025.parquet`, `data/raw/2025/Suzuka/`, HF repo path. -->

### Expected vs observed values
```
Expected:
Observed:
```

### Reproduction snippet
```python
import pandas as pd

df = pd.read_parquet("data/processed/…")
# …
```

### Upstream source
<!-- FastF1 session id, OpenF1 meeting key, manual CSV origin. -->

**Severity:** <!-- Breaks the model / Small drift / Cosmetic -->
