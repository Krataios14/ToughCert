# Steel Fracture Toughness Prediction (Neural Model)

A modern, memory-conscious neural network pipeline for predicting steel fracture toughness from easier-to-measure experimental data (composition, processing, microstructure, and basic mechanical properties). The project is built to generalize well to unseen data and to train comfortably within **<16 GB RAM**.

## Goals
- High-accuracy fracture toughness prediction with strong generalization.
- Robust training and evaluation with a fixed validation split (seeded).
- Production-ready inference with saved artifacts.
- Clear documentation and reproducible runs.

## What's Included
- **Default model: Gradient-Boosted Trees** (HistGBR) for strong tabular performance.
- **Optional Tabular Transformer** (FT-Transformer style) for mixed numerical/categorical features.
- **Modern training recipe**: Huber loss, cosine LR schedule, weight decay, early stopping, stochastic weight averaging (SWA), mixup for tabular data, and gradient clipping.
- **Data pipeline**: schema validation, robust scaling, categorical encoding, missing-value handling.
- **Visualizations**: learning curves, parity plots, residuals, and permutation-based feature importance.
- **Config-driven** training via YAML.
- **Tests**: synthetic data smoke test.

## Modeling Choices (Why This Generalizes Well)
- **Feature tokenization** via per-feature embeddings and Transformer self-attention to capture feature interactions.
- **Robust loss** (Huber) to reduce sensitivity to outliers typical in mechanical testing data.
- **Regularization** with dropout, weight decay, and early stopping.
- **SWA** to improve generalization from flat minima.
- **Mixup** (numerical features) as a light augmentation to reduce overfitting.
- **Missing-aware tokens** for numeric features so the model can handle partial feature availability per material.

## Project Structure
```
configs/          Training configs
 data/             Place raw/processed CSV files here
 notebooks/        Optional analysis notebooks
 reports/          Figures and metrics
 reports/figures/  Output plots
 src/              Source code
 tests/            Pytest tests
```

## Data Format
The training CSV should be a **single table** with one row per specimen/measurement.

Required columns:
- `fracture_toughness` (target, e.g., K_IC or J_IC in consistent units)

Recommended feature columns (examples):
- **Composition**: `C`, `Mn`, `Cr`, `Mo`, `Ni`, `V`, `Si`, ...
- **Processing**: `austenitizing_temp`, `tempering_temp`, `cooling_rate`, `rolling_reduction`, ...
- **Microstructure**: `grain_size`, `phase_fraction_ferrite`, `phase_fraction_martensite`, ...
- **Mechanical**: `yield_strength`, `tensile_strength`, `elongation`, `hardness`, ...
- **Categorical**: `steel_grade`, `heat_treatment_route`, `product_form`, ...

Missing values are supported and handled by the pipeline.

## Quickstart
### 1) Create a Python environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Prepare data
Transform the provided assets into model-ready files:
```bash
python -m src.prepare_data ^
  --train assets/combined_fracture_training.csv ^
  --unseen assets/combined_fracture_unseen.csv ^
  --out_train data/processed_train.csv ^
  --out_unseen data/processed_unseen.csv
```
This also writes `data/schema.json` (feature lists) which the training config uses automatically.

### 3) Train
```bash
python -m src.train --config configs/default.yaml
```

### 4) Tune (optional, recommended)
```bash
python -m src.tune_gbdt --config configs/default.yaml --trials 40
```

### 5) Improve accuracy (recommended)
Use the built-in auto model selection and feature pruning:
- `model.type: auto` in `configs/default.yaml`
- `data.min_non_zero_fraction` prunes ultra-rare element features
- `data.min_known_numeric` trades coverage for accuracy

### 6) Segment models (recommended for heterogeneous data)
Train separate models by phase and material condition:
```bash
python -m src.segment_train --config configs/default.yaml
python -m src.segment_evaluate --segments_root runs/segments_YYYYMMDD_HHMMSS
```

### 7) Segment-aware prediction (global fallback)
```bash
python -m src.segment_predict ^
  --config configs/default.yaml ^
  --data data/processed_unseen.csv ^
  --out predictions.csv
```
You can also specify explicit paths:
```bash
python -m src.segment_predict ^
  --config configs/default.yaml ^
  --data data/processed_unseen.csv ^
  --segments_root runs/segments_YYYYMMDD_HHMMSS ^
  --global_run runs/run_YYYYMMDD_HHMMSS ^
  --out predictions.csv
```

### 4) Evaluate & Visualize
```bash
python -m src.evaluate --config configs/default.yaml
python -m src.visualize --config configs/default.yaml
```

### 5) Inference
```bash
python -m src.predict --model runs/best_model.pt --data data/new_samples.csv
```

## Configuration
Edit `configs/default.yaml` to control:
- Feature lists and target
- Model size (d_model, depth, heads)
- Training settings (batch size, epochs, LR)
- Regularization and augmentation
- Output paths

## RAM Guidance
To stay under 16 GB RAM:
- Keep `batch_size` between 128-1024 depending on feature count.
- Use `mixed_precision: true` if a GPU is available.
- Limit `d_model` to 128-256 and depth to 3-6.

## Outputs
Training generates artifacts in `runs/`:
- Best model checkpoint
- Scalers and encoders
- Metrics JSON
- Plots (learning curves, parity, residuals)

## Testing
```bash
pytest -q
```

## Notes
- Replace or extend the feature list in `configs/default.yaml`.
- Ensure all measurements are in consistent units.
- The model supports both CPU and GPU.
