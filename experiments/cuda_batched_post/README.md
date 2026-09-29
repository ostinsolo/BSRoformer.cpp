# Native batched CUDA post (isolated experiment)

Goal (§3 of `windows/cuda-final-opt`): replace Python per-frame iSTFT with:

- persistent `cufftPlanMany` (batched iSTFT for 6 stems × 2 channels)
- persistent device buffers
- custom mask/reconstruction kernel (input: saved `mask_output.f32` from `--probe-dir`)
- Hann/window + GPU overlap-add where useful

**Parity gate vs C++ `PostProcessAndISTFT`:** Pearson > 0.99999, max abs < 1e-3, high SI-SDR per stem.

**Benchmark shapes:** 81920, 176400, 409600 — report mask kernel, cuFFT, window/OLA, transfer, total.

**Modes:**

- **A:** host mask → GPU post → host audio
- **B:** mask resident on GPU → GPU post → host audio (target VST path)

Implementation status: **scaffold only** — build wiring and kernels land on this branch after F32/FP16 baselines.
