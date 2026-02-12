# Discord Intel (Feb 15, 2026)

## Key Competitor Techniques (from Wunderfund Discord)

### sultanmunirov (competitive player)
1. **Data augmentation** — variance stretch/compress of features. Got **5x more training data** this way.
2. **Chrono initialization** for GRU — claimed "huge boost" in performance.
3. **Highway head** instead of linear output — claimed significant improvement.

### peter (competitive player)
4. **SWA (Stochastic Weight Averaging)** — average last 3-5 checkpoints. Claimed "+0.005 R2".

### Other competitors
5. **2-stage boosting** — train model2 on model1 residuals.
6. Someone achieved **0.292 with just raw features** — proves architecture/training is the gap, not features.

## Official Baseline
- Single GRU model via ONNX
- Architecture: **3-layer GRU, h=64, 32 raw features, linear output (64→2)**
- ~44K parameters total
- Fixed 100-step context window at inference (not stateful recurrent)
- **Scored 0.2761 on LB** — beats our 0.2689 ensemble!

## Key Insight
Our 42-feature, 9-model, 268K-params-per-model ensemble is OVERFITTING.
The simpler baseline (32 raw features, 77K params, single model) generalizes better.

## Implications for Our Strategy
1. **Revert to raw features** — derived features hurt generalization
2. **Reduce model capacity** — h=64 + 3 layers beats h=144 + 2 layers
3. **Linear output** — simpler than MLP, less overfitting
4. **Focus on training techniques** (augmentation, SWA, chrono init) over architecture
5. **Single strong models > large ensembles** with weak individuals
