# RTX A5000 Port of LeakyReLU^2 + Legal TTT + Parallel Muon

This folder documents an evidence-driven port of the record run in
`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
to a single **RTX A5000** pod.

This is not yet a submission. It is a working record for:

- the exact source code provenance
- the constraints we must preserve
- the porting decisions made for a single 24 GB GPU
- every experiment, log, and artifact copied back from the pod

## Source Run Provenance

Source record:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed42.log`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed2025.log`

Key source-run facts:

- Record score: `1.1194 val_bpb` post-TTT, `3-seed mean`
- Hardware: `8xH100 SXM`
- Script size: `89,458 bytes`
- Quantized model size: about `15.89 MB`
- Total artifact size: about `15.95 MB`
- Training stopped by the `600s` wall-clock cap at about `7,179` steps
- Peak memory in `train_seed1337.log`: `21,471 MiB allocated`, `22,002 MiB reserved`

## Current Target Hardware

Verified on the active pod on `2026-03-25`:

- GPU: `NVIDIA RTX A5000`
- VRAM: `24,564 MiB`
- Driver: `570.211.01`
- PyTorch: `2.9.1+cu128`
- CUDA: `12.8`
- Compute capability: `(8, 6)`
- `torch.cuda.is_bf16_supported()`: `True`
- `flash_attn_interface`: importable on the pod

## Hard Constraints

- No training-time access to ungraded validation tokens.
- No use of `val` tokens except where already allowed by the challenge rules and the source record's legal TTT protocol.
- The adaptation work should continue to respect a `600s` training wall-clock when we run challenge-shaped experiments.
- Changes must be evidence-backed. No speculative "probably better" edits without documentation.
- All commands, code diffs, logs, metrics, and conclusions must be copied back into this local repository.

## Initial Porting Hypothesis

The source record is tightly coupled to `8xH100` in three major ways:

1. `grad_accum_steps` is derived from `8 // world_size`, which encodes the original hardware budget directly.
2. The workload is tuned for a global train batch of `786,432` tokens and about `83.4 ms/step` on 8 GPUs.
3. The script assumes the record environment's attention and communication profile, including parallel Muon overlap tuned around distributed execution.

For a single RTX A5000, the first goal is not leaderboard parity. The first goal is:

- preserve the source architecture and legal evaluation behavior
- make the script run correctly and reproducibly on one GPU
- choose the smallest set of hardware-driven changes needed to recover a promising training trajectory

## Experiment Protocol

For each run we will record:

- experiment id
- exact code state or diff summary
- exact command line
- relevant environment variables
- measured GPU memory
- elapsed time and steps reached
- pre-quant and post-quant metrics if available
- whether the result is promoted, discarded, or requires follow-up

## Artifact Checklist

Files expected to accumulate here:

- `README.md` for narrative and conclusions
- `results.tsv` for the experiment table
- remote run logs copied back locally
- `train_gpt_a40.py` snapshots if the code evolves materially between experiments

## Current Status

Completed:

- Source record identified and inspected locally.
- RTX A5000 hardware/software facts verified on the pod.
- Local documentation scaffold created.
- `train_gpt_a40.py` created from the source record and synced to the pod.
- Initial single-GPU patch set added:
  - `GRAD_ACCUM_STEPS` override support
  - explicit train-batch divisibility checks
  - `EVAL_BATCH_SEQS` and `TTT_SCORE_BATCH_SEQS` knobs

## Early Findings

### 1. Record per-microbatch load does not fit on RTX A5000

A direct forward/backward probe on the uncompiled model with the source record's
effective per-microbatch load:

- `TRAIN_BATCH_TOKENS=786432`
- `WORLD_SIZE=1`
- `GRAD_ACCUM_STEPS=8`
- local microbatch = `48 x 2048`

failed with CUDA OOM in the XSA path at about `23.0 GiB` allocated.

### 2. Measured fit sweep for single forward/backward micro-step

See `batch_probe.tsv` for the exact table. Key results:

- `786432` global tokens: OOM
- `655360` global tokens: OOM
- `524288` global tokens: fits, about `19.8 GiB`
- `393216` global tokens: fits, about `14.9 GiB`
- `262144` global tokens: fits, about `10.0 GiB`
- `131072` global tokens: fits, about `5.1 GiB`

### 3. Full compiled script now runs end-to-end on the pod

Two end-to-end smoke runs were completed and copied back locally:

- `a5000_port_smoke_001.txt`
- `a5000_port_smoke_002.txt`

Both runs completed training, diagnostic eval, export, and int6 roundtrip eval.
Both then hung after the final printed metric, so the remote process was killed
manually after the useful outputs were safely logged. This looks like a teardown
issue rather than a training/eval correctness issue.

Smoke comparison:

| Run | Train Batch Tokens | Step 1 train time | Step 2 cumulative train time | Approx. steady-state step | Peak memory | Final int6 roundtrip bpb |
|-----|-------------------:|------------------:|-----------------------------:|--------------------------:|------------:|-------------------------:|
| smoke_001 | 262144 | 74.229 s | 75.373 s | ~1.144 s | 7585 MiB | 4.10604768 |
| smoke_002 | 131072 | 70.991 s | 71.661 s | ~0.593 s | 4087 MiB | 4.10604872 |

Interpretation:

- The dominant fixed cost is `torch.compile`, about `71-74 s`.
- After compile, `131072` global tokens is about `1.93x` faster per step than `262144`.
- `131072` also leaves much more VRAM headroom for future eval or TTT experiments.

### 4. First 600-second RTX A5000 trajectory run

Run log copied locally:

- `a5000_traj_001.txt`

Configuration:

- `TRAIN_BATCH_TOKENS=131072`
- `TRAIN_SEQ_LEN=2048`
- `GRAD_ACCUM_STEPS=8`
- `WARMUP_STEPS=20`
- `MAX_WALLCLOCK_SECONDS=600`
- `TTT_ENABLED=0`
- `EVAL_STRIDE=0`

Observed results:

- timed training stop: `step 887`
- timed-stop standard eval: `val_bpb 1.4972`
- peak memory: `4084 MiB allocated`, `4164 MiB reserved`
- post-EMA diagnostic eval: `val_bpb 1.6124`
- final int6 roundtrip eval: `val_bpb 2.1564`
- total quantized artifact size: `6,704,096 bytes`

Interpretation:

- The port is genuinely training on RTX A5000; this is no longer just a compatibility pass.
- The stop-time model is materially better than the EMA-applied model on this run.
- The export path is currently not preserving quality well at this training horizon.
- The next useful work is on averaging/export fidelity before any compression-research phase.

### 5. Concrete code-behavior mismatch found

The script and README/command surface imply EMA and SWA are configurable, but the code path inspected so far shows:

- EMA decay is hardcoded to `0.997`
- EMA is always applied at export time
- SWA state is tracked, but not actually applied in the final averaging path

That makes this a high-priority correctness issue for the next experiment, because the current run already shows the raw stop-time checkpoint beating the EMA-applied export.

### 6. Averaging-control fix verified

Follow-up control run copied locally:

- `a5000_avgctrl_smoke_001.txt`

Changes:

- added `EMA_ENABLED`
- added `EMA_DECAY`
- added `FINAL_AVERAGING` with explicit modes: `raw`, `ema`, `swa`, `lawa`

Verification run:

- `EMA_ENABLED=0`
- `SWA_ENABLED=0`
- `FINAL_AVERAGING=raw`
- `ITERATIONS=2`
- `TRAIN_BATCH_TOKENS=131072`

Observed behavior:

- log prints `ema_enabled:False ... final_averaging:raw`
- log prints `final_averaging:keeping raw stop-time weights`
- post-averaging diagnostic matches the stop-time metric exactly
- raw int6 roundtrip also logs successfully

This means the next longer comparison run can cleanly answer whether the bad
post-export metrics in `traj_001` were mostly caused by forced EMA, by the
quantization/export path itself, or by both.

Next:

- use the documented evidence to pick the first longer RTX A5000 trajectory run
- keep TTT disabled until the training-side trajectory is understood
- copy every longer-run log back into this folder before making further changes
- if smoke test success, work on integrating: `https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/` carefully so we can achieve better compression and efficiency during training to drive val_loss and val_bpb lower without cheating or gaming scores.
