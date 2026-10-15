# Changelog

## 0.1.0 (2026-10-15) — first public release

- `--fp16` for RTX 20-series (Turing) and other GPUs without bfloat16; `--bf16` now warns when the GPU lacks native support.
- Default `--ride-speed` lowered from 40 to 20 px/s. 40 missed a confirmed ride on the Belmar cam; 20 finds it.
- Clear startup errors for the three common setup problems: not logged in to Hugging Face, no CUDA GPU, ffmpeg missing.
- Progress line now shows an ETA.
- `--crop` is validated against the frame size instead of failing deep in the model.
- Added a GPU-free regression test (`tests/`) using a real track.csv with a known ride.
- README rewritten for first-time users; LICENSE (MIT), requirements files, issue template.
