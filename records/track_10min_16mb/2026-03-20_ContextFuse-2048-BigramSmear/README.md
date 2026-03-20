This records the submissions for `ContextFuse-2048-BigramSmear`.

`val_bpb` is the primary target for this entry. This package builds directly on our earlier PR `#143` submission, `ContextFuse-2048`, and pushes the same baseline-derived path toward better compression-aware quality rather than changing the tokenizer or relying on evaluation quirks.

This is not presented as a new SOTA claim. It is a stronger, reproducible follow-up submission with three full `8x H100 SXM` runs.

## Summary

- Prior submission: PR `#143`, `ContextFuse-2048`, `val_bpb=1.17792945`
- New canonical run: `val_bpb=1.15369190`
- Improvement over PR `#143`: `-0.02423755` BPB
- Three-seed mean: `1.15543586`
- Three-seed median: `1.15369190`

## Three-Seed Results

| seed | run id | val_loss | val_bpb | steps | ms/step | train time | eval time | model bytes | standalone total bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1337 | `attempt007_h100x8_pr162_safeeval_s1337` | `1.94795528` | `1.15369190` | `7113` | `82.94` | `589978ms` | `150351ms` | `15267279` | `15330779` |
| 42 | `attempt007_h100x8_pr162_safeeval_s0042` | `1.95728867` | `1.15921967` | `6113` | `96.53` | `590062ms` | `151064ms` | `15295571` | `15359071` |
| 7 | `attempt007_h100x8_pr162_safeeval_s0007` | `1.94745570` | `1.15339601` | `7083` | `83.30` | `590045ms` | `150441ms` | `15242469` | `15305969` |

Notes:
- All three runs satisfy the local hard limits: training `< 600000ms`, evaluation `< 600000ms`, artifact `< 16000000` bytes.
- Seed `42` ran materially slower than seeds `1337` and `7`, so the three-run spread mixes seed variance and throughput variance.
- The two normal-throughput runs (`1337`, `7`) are tightly clustered around `1.1535` BPB.

## What Changed From PR #143

PR `#143` (`ContextFuse-2048`) already had:
- `TRAIN_SEQ_LEN=2048`
- sliding-window final eval with `EVAL_STRIDE=64`
- fp16 embedding preservation

This follow-up adds the strongest compression-aware pieces we found that still transferred honestly:
1. `BigramHashEmbedding` on the input path.
2. `SmearGate` to blend each token representation with the previous token.
3. Mixed `int6` export for large `mlp` and `attn` matrices.
4. `MLP_HIDDEN=1536` so the int6 budget buys back a larger MLP.
5. `MUON_WEIGHT_DECAY=0.02`.
6. `SWA_ENABLED=1` with averaging over the late low-LR phase.
7. Corrected control-tensor handling so only `bigram.scale` is exempted from normal quantization, not the entire bigram module.
8. A narrower fp16 keep rule: `FP16_KEEP_NAME_PATTERNS=tok_emb,blocks.8.attn.c_k`.

The result is a much better BPB-focused export/training stack while keeping the same challenge-valid evaluation framing.

## Method Credit

This submission intentionally credits the prior work it builds on:

- PR `#143`: our original `ContextFuse-2048` submission.
  - Base scaffold for train@2048, sliding eval, and fp16 embedding preservation.
- PR `#135`:
  - `BigramHash + SmearGate + mixed int6 + 3x MLP` as the first strong public architecture showing this family could compete near the frontier.
- PR `#162`:
  - `Muon WD + SWA + refined fp16 keep pattern` as the strongest method family we found after our later attempts plateaued.
- Public record folders already in this repo:
  - `2026-03-18_LongContextSeq2048` for long-context training.
  - `2026-03-19_SlidingWindowEval` for richer-context evaluation.

We did not blindly copy PR `#162`'s scoring path. We kept the safer local sliding-window evaluator and did not reproduce the Codex-reviewed tail-token double-count issue mentioned on that PR.

## Configuration

- Layout:
  - `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
  - `MLP_HIDDEN=1536`
- Input enhancements:
  - `USE_SMEAR_GATE=1`
  - `BIGRAM_VOCAB_SIZE=4096`
  - `BIGRAM_DIM=128`
- Export strategy:
  - `MIXED_INT6_EXPORT=1`
  - `INT6_CATEGORIES=mlp,attn`
  - `FP16_EMBED_PASSTHROUGH=1`
  - `FP16_LATE_K_LAYERS=0`
  - `FP16_KEEP_NAME_PATTERNS=tok_emb,blocks.8.attn.c_k`
  - `CONTROL_TENSOR_NAME_PATTERNS=attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,bigram.scale`
- Training:
  - `TRAIN_BATCH_TOKENS=786432`
  - `TRAIN_SEQ_LEN=2048`
  - `MAX_WALLCLOCK_SECONDS=590`
  - `TIED_EMBED_LR=0.03 MATRIX_LR=0.02 SCALAR_LR=0.02`
  - `MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500`
  - `MUON_BACKEND_STEPS=5`
  - `ADAM_WEIGHT_DECAY=0.01`
  - `MUON_WEIGHT_DECAY=0.02`
  - `GRAD_CLIP_NORM=0.3`
  - `WARMDOWN_ITERS=3000`
  - `SWA_ENABLED=1 SWA_START_FRAC=0.5 SWA_EVERY=200`
- Evaluation:
  - `EVAL_STRIDE=64`
  - `EVAL_BATCH_SEQS=32`

## Artifact Accounting

- The included logs are the original automatically produced logs from the successful H100 runs.
- Those logs were generated from the live Modal-launched execution snapshot, so the in-run `Code size` lines are larger than the standalone file in this folder.
- The intended submission artifact is the standalone `train_gpt.py` in this record folder, not the Modal wrapper.

Standalone artifact accounting for this folder:
- `train_gpt.py`: `63500` bytes
- canonical compressed model (`seed 1337`): `15267279` bytes
- canonical standalone total: `15330779` bytes
- worst standalone total across the three included runs: `15359071` bytes

That leaves `640929` bytes of headroom under the `16000000` byte cap even on the largest of the three runs.

## Reproduction Command

```bash
cd records/track_10min_16mb/2026-03-20_ContextFuse-2048-BigramSmear
RUN_ID=contextfuse2048_bigramsmear \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=590 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
TRAIN_BATCH_TOKENS=786432 \
VAL_BATCH_SIZE=524288 \
TRAIN_SEQ_LEN=2048 \
NUM_LAYERS=9 \
MLP_HIDDEN=1536 \
USE_SMEAR_GATE=1 \
BIGRAM_VOCAB_SIZE=4096 \
BIGRAM_DIM=128 \
MIXED_INT6_EXPORT=1 \
INT6_CATEGORIES=mlp,attn \
CONTROL_TENSOR_NAME_PATTERNS=attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,bigram.scale \
FP16_EMBED_PASSTHROUGH=1 \
FP16_LATE_K_LAYERS=0 \
FP16_KEEP_NAME_PATTERNS=tok_emb,blocks.8.attn.c_k \
ORTHO_INIT=1 \
SWA_ENABLED=1 \
SWA_START_FRAC=0.5 \
SWA_EVERY=200 \
TIED_EMBED_LR=0.03 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_BACKEND_STEPS=5 \
ADAM_WEIGHT_DECAY=0.01 \
MUON_WEIGHT_DECAY=0.02 \
GRAD_CLIP_NORM=0.3 \
WARMDOWN_ITERS=3000 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py` — standalone record-folder training/eval/export script for this recipe
- `train.log` — canonical `SEED=1337` run
- `train_seed42.log` — full `SEED=42` rerun
- `train_seed7.log` — full `SEED=7` rerun
- `submission.json` — metadata for the entry
- `README.md` — this file
