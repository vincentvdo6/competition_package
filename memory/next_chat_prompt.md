# Next Chat Prompt — Seamless Continuation

## Where We Are (Feb 16, 2026 — Round 3)

### Round 1+2 Results (ALL Discord techniques KILLED)
- **baseline_match**: mean 0.2663 (consistent control, 2 rounds)
- **chrono_v2**: -0.0204 (KILLED — even with bias_hh only, T=10)
- **aug_v2**: -0.0035 (KILLED — even with gentle 0.95-1.05 scale)
- **swa_v2**: -0.0017 (KILLED — only helped seed 42, hurt 43/44)
- **windowed inference**: +0.0003 (NOT the gap to baseline)

### Gap Analysis
- Our baseline_match val: 0.2663. Official baseline LB: 0.2761. Gap: ~0.01.
- Architecture/inference ruled out — gap must be in **training recipe**.
- Key differences not yet tested:
  1. Loss function (we use CombinedLoss, baseline likely uses simple MSE)
  2. Normalization (we normalize, baseline feeds raw features)

### Round 3 Ablations Ready (Codex-agreed)
- Configs created: gru_baseline_mse.yaml, gru_baseline_raw.yaml, gru_baseline_mse_raw.yaml
- Notebook 15 updated for Round 3 (cells 3-5 = MSE/raw/MSE+raw, cell 6 = eval)
- Code committed and pushed

### What To Do Next
1. **Run notebook 15 on Colab** — Cell 2 (skip if already have baseline_match), Cells 3-5 (new ablations), Cell 6 (eval)
2. **Evaluate results** against baseline_match control (mean 0.2663)
3. **If pass**: Scale up winning config with more seeds, build submission
4. **If fail**: Try windowed training (100-step BPTT), different optimizer, or analyze baseline.onnx weights

### Submission Budget
- 5 subs/day. Official baseline already submitted and scored 0.2761.
