# Surf tracker

Goal: track a surfer in screen recordings of Surfline cam rewinds and detect which waves they caught.
Entry point: surf_tracker.py (click to select → SAM 3 streaming track → ride detection → ffmpeg clips).
README.md has setup and usage for end users; this file is for working on the code.

## Environment notes
- Reference machine: Windows 11, RTX 2070 (8 GB, Turing, compute capability 7.5), Python 3.12 venv at .venv,
  torch 2.6 cu124 + torchvision 0.21 (torchvision is required by Sam3ImageProcessor), transformers 5.x.
- Measured 2026-10-15 on that machine, 1196x384 video, --stride 2: fp32 0.8 fps / 3.2 GB peak VRAM,
  fp16 2.3 fps / 1.8 GB, identical tracking scores. Turing has fp16 tensor cores but only emulates bf16,
  so torch.cuda.is_bf16_supported() is True there and is NOT a useful check; use compute capability >= 8.
- SAM 3 weights are gated on HF (facebook/sam3): accept the license there, then `hf auth login`.
- The script cannot run unattended: it opens an OpenCV window and needs a click on the surfer. A headless
  harness for benchmarking the model path (no window, click taken from a track.csv) is easy to write
  from open_video + prune_session; see tests/ for the GPU-free detection test.

## Domain facts
- Surfers are ~10–20 px tall on a Surfline cam at native size; the camera is fixed (no pan/zoom).
- Belmar cam (NJ): shore is at the bottom of the frame (--shore down). OBS recording of the player is
  1196x384 at 30 fps once cropped to the video region.
- A ride is 3–15 s of fast motion, mostly lateral along the wave with only a small shoreward (+y) component.
  Sitting/paddling 3–14 px/s, riding ~25–60 px/s. Mask area jumps from ~35 px to 100–190 px during a ride (spray).
- --ride-speed 20 finds the confirmed ride at 1:48 of the Belmar sample; 40 (the old default) misses it.
- After a wave, re-click on the surfer, not on the whitewater: the whitewater segments badly.
- Waves hide the surfer for 1–2 s routinely, so --lost-seconds must stay above that.

## Conventions
- Keep it a single script unless it gets unwieldy.
- Never load the whole video into memory (streaming mode only); prune_session keeps VRAM bounded.
- Don't change the track.csv columns without updating load_track/--redetect and tests/test_redetect.py.
- tests/fixtures/track_belmar_ride_at_108s.csv is a real track from the Belmar sample (95–130 s) with the
  known ride at ~108 s. `python -m pytest tests` must pass before a release; it needs no GPU.
- Sample recordings live outside the repo (Surfline footage is never committed). On the reference machine
  they are in C:\Users\Colin\Downloads (sample.mp4, sample-crop.mp4).
- Measure fp16/fp32 changes against the fixture video range, not a synthetic clip.
