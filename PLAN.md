# PLAN

## Objective
Build a neural-network-based fracture toughness prediction pipeline for steels using tabular experimental data, optimized for generalization on unseen data and capped to <16 GB RAM during training.

## Approach
1. **Define data schema**
   - Target: `fracture_toughness`.
   - Numerical and categorical features configurable via YAML.
   - Robust missing-data handling.

2. **Model architecture**
   - FT-Transformer style: embeddings + Transformer encoder + MLP head.
   - LayerNorm, dropout, residual connections.
   - Optional feature-wise mixup.

3. **Training recipe**
   - Huber loss (robust to outliers).
   - AdamW with cosine schedule, warmup, weight decay.
   - Early stopping and SWA.
   - Gradient clipping.

4. **Evaluation & visualization**
   - Fixed validation split with seeded reproducibility.
   - Metrics: MAE, RMSE, R2.
   - Plots: learning curves, parity, residuals, permutation importance.

5. **Artifacts**
   - Model checkpoint.
   - Scaler/encoders.
   - Metrics JSON.
   - Figures.

6. **Testing**
   - Synthetic data smoke test for the training pipeline.

## Files to implement
- `src/data.py`: loading, schema validation, preprocessing.
- `src/model.py`: FT-Transformer model.
- `src/train.py`: training loop and checkpointing.
- `src/evaluate.py`: metrics and evaluation.
- `src/visualize.py`: plots.
- `src/predict.py`: inference.
- `configs/default.yaml`: configuration.
- `tests/test_smoke.py`: smoke tests.

## Constraints
- RAM usage under 16 GB.
- CPU-first; GPU optional.
- Deterministic seed for reproducibility.
