# Gemini Deep Research Prompt

## Copy-paste the text below as your Gemini Deep Research query. Upload the other 7 files as attachments.

---

I'm competing in the Wunderfund Predictorium, a Limit Order Book (LOB) prediction competition hosted by an HFT prop trading firm. The task is to predict two targets (t0, t1) from 32 pre-scaled LOB features using step-by-step online inference with a recurrent neural network. The metric is Weighted Pearson Correlation (weighted by |target|, clipped to [-6,6], averaged across t0 and t1). Scoring runs on 1 vCPU CPU-only with a 70-minute time limit.

**Current state**: I'm ranked 73/4728 (score 0.2885) with a 10-model ensemble of identical vanilla GRU models (3-layer, hidden=64, raw 32 features, no normalization, plain MSE loss, linear output). The top competitor scores 0.3240 (+0.0355 above me). My target is top 50 (~0.2920+).

**The critical phenomenon**: My vanilla GRU has a consistent POSITIVE val-to-LB gap (+0.018 for the 10-model ensemble, +0.008 for single models). The test set appears EASIER than validation for simple models. However, every attempt to make the model more complex (larger hidden size, LayerNorm, MLP head, feature engineering, different losses, different architectures) causes a NEGATIVE gap — val improves but LB collapses. A complex pipeline with the same h=64 3L GRU but with input projection + LayerNorm + MLP + derived features + z-score normalization scored 0.2394 on LB despite 0.2738 on val (a catastrophic -0.034 gap).

**What I've exhaustively tried and KILLED** (full details in CONTEXT.md):
- Architecture: larger hidden (128/144/192 all worse), LSTM (-0.011), attention, TCN, transformer, CVML, SE-Net gate, input projection, LayerNorm, MLP output
- Features: derived 42 features (spreads, imbalances), microstructure, lag diffs, z-score, RevIN (full and causal)
- Loss: Pearson blend, Pearson primary, Huber (catastrophic -0.025), combined weighted MSE, auxiliary heads, recency weighting
- Training: full-data training, stronger regularization, SAM optimizer, EMA, cosine warmup, Adam, higher LR, bagging
- Ensemble diversity: 20 models (= 10 models), greedy selection (WORSE), recipe variants (barely decorrelate), LSTM mixing, checkpoint epochs, Pearson blend mixing (genuine diversity at 0.78 correlation but WRONG direction on LB)
- Inference: windowed (same as step-by-step)

**Key constraints**:
- Final ranking uses a DIFFERENT dataset — generalization > val optimization
- Top competitor says "improve generalisation, not validation score" and "99% pure ML"
- "Almost all financial features do not provide an increase" (from #3 on leaderboard)
- t1 is universally hard (best ~0.17 vs t0 ~0.47) — any t1 improvement has outsized metric impact
- Train-val distribution shift is material (KS stat up to 0.33 in price features)
- 1 vCPU, 16GB RAM, 70min timeout, 20MB zip limit, no GPU at inference

**I need you to research and propose fundamentally new approaches to break past 0.2885. Specifically:**

1. **What model architectures from recent LOB/time-series literature (2023-2026) could match vanilla GRU's generalization while being more powerful?** I need architectures that are inherently regularized or generalization-friendly, not just "add more capacity." Consider: structured state space models (S4/Mamba), lightweight attention variants, wavelet-based approaches, or other sequence models that have shown strong OOD performance.

2. **Why does the positive val-to-LB gap exist and how can I exploit it?** The test set being easier for simple models is unusual. What statistical properties of the test data could cause this? Is there a way to deliberately train for this kind of transfer?

3. **What training paradigms could improve generalization without increasing model complexity?** Consider: meta-learning, distribution-robust optimization (DRO), invariant risk minimization (IRM), domain generalization techniques, curriculum learning, or data augmentation strategies specific to financial time series.

4. **How do top competitors likely achieve 0.32+ with "99% pure ML"?** Given the same raw features and simple-is-better pattern, what specific techniques from the financial ML / LOB prediction literature could explain a +0.04 gap? Consider ensemble methods, post-processing, calibration, or training tricks.

5. **How can I improve t1 prediction specifically?** t1 has lag-1 autocorrelation of 0.976 (extremely persistent) and 1.6x the variance of t0. What specialized approaches work for highly autocorrelated, high-variance targets?

6. **What post-processing or inference-time techniques could help?** Consider: prediction clipping strategies, per-sequence calibration, temporal smoothing, outlier handling, or other test-time adaptations that don't require model changes.

7. **Evaluate these 7 untested ideas from our codebase assistant (Codex) — which are most promising and which should be killed without testing?**
   - Regime-gated experts (train GRUs on different data regimes, tiny gating model)
   - Adversarial-validation density-ratio weighting (detect train-vs-val shift, use as sample weights)
   - Mega-teacher distillation (40-100 models -> 1-2 students)
   - Tree model blend (LightGBM/CatBoost on sequence summaries)
   - Prediction neutralization (regress out nuisance basis functions)
   - Variance-penalized stacking (max mean_corr - beta*std_corr)
   - Chunk-wise inference calibration (per-sequence demeaning/winsorizing)

**Please provide concrete, implementable recommendations ranked by expected ROI (impact vs. implementation effort). For each recommendation, explain WHY it would work given my specific findings (positive gap, simplicity = generalization, t1 difficulty, distribution shift). I have ~1 month and 5 submissions/day remaining.**

The attached files contain: my full experiment history and kill list (CONTEXT.md), the winning model architecture (gru_baseline.py), inference code (solution.py), winning config (config.yaml), loss functions (losses.py), the official scoring metric (utils.py), and validation data statistics (data_profile.txt).
