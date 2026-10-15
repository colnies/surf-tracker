"""
surf_tracker.py — track yourself in a Surfline cam recording and detect your rides.

Pipeline:
  1. Open the recording, jump to --start, click on yourself, press Enter.
  2. SAM 3 (streaming mode) tracks you frame by frame. If it loses you, the
     window pauses and asks you to click again (or press S to skip ahead).
  3. Your centroid per frame is written to <out>/track.csv.
  4. Rides are detected from sustained speed in the shoreward direction and
     written to <out>/rides.csv; clips are cut with ffmpeg into <out>/clips/.
  5. <out>/annotated.mp4 shows your trail and a RIDE banner.

Keys while it runs:  R = re-click on yourself   Q = stop and finish up
Keys when selecting: click = choose point        Enter = confirm   S = skip (stay lost until R)

Usage:
  python surf_tracker.py session.mp4 --start 95 --out run1 --fp16
  python surf_tracker.py session.mp4 --crop 0,120,1920,840 --stride 2 --bf16   # RTX 30-series and newer
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch


# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Track a surfer in a cam recording and detect rides.")
    p.add_argument("video", help="path to screen recording (mp4/mkv/mov)")
    p.add_argument("--out", default="surf_out", help="output folder")
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--end", type=float, default=None, help="end time in seconds (default: end of file)")
    p.add_argument("--crop", default=None, help="x,y,w,h crop in pixels to drop player UI / zoom on lineup")
    p.add_argument("--stride", type=int, default=2, help="process every Nth frame (2 = ~15 fps on 30 fps video)")
    prec = p.add_mutually_exclusive_group()
    prec.add_argument("--fp16", action="store_true",
                      help="run the model in float16: 2-3x faster and half the VRAM. Use this on RTX 20-series "
                           "(Turing) and any GPU without native bfloat16")
    prec.add_argument("--bf16", action="store_true",
                      help="run the model in bfloat16: same speedup as --fp16 with less risk of overflow. "
                           "Needs an RTX 30-series (Ampere) or newer GPU")
    p.add_argument("--allow-cpu", action="store_true",
                   help="run without a CUDA GPU. Roughly 50-100x slower than a mid-range GPU; only useful "
                        "for a few seconds of video")
    p.add_argument("--model", default="facebook/sam3", help="HF model id")
    # loss handling
    p.add_argument("--lost-seconds", type=float, default=3.0,
                   help="seconds of continuous loss before asking for a re-click; waves hide you for "
                        "1-2 s routinely, so keep this above that")
    p.add_argument("--min-score", type=float, default=2.0,
                   help="object score logit below which a frame counts as lost (good tracks score ~4-9, "
                        "junk re-identifications score ~0)")
    p.add_argument("--reacquire-radius", type=float, default=25.0,
                   help="px; after a loss, you are assumed to still be within this radius of where you "
                        "were last seen, growing by --max-speed px per second lost")
    p.add_argument("--max-speed", type=float, default=80.0,
                   help="px/s; fastest you can plausibly move (riding). A mask farther from the last "
                        "good fix than radius + speed*elapsed is someone else and counts as lost")
    # ride detection
    p.add_argument("--ride-speed", type=float, default=20.0,
                   help="px/s above which you're riding. 20 suits Surfline cams at native resolution, where "
                        "paddling is 3-14 px/s and a ride 25-60 px/s; raise it if you get false rides")
    p.add_argument("--min-ride", type=float, default=2.0, help="minimum ride duration in seconds")
    p.add_argument("--merge-gap", type=float, default=1.0, help="merge rides separated by less than this (s)")
    p.add_argument("--shore", default="down", choices=["down", "up", "left", "right", "any"],
                   help="which screen direction is toward the beach")
    p.add_argument("--pad", type=float, default=3.0, help="seconds of padding before/after each clip")
    p.add_argument("--no-clips", action="store_true", help="skip ffmpeg clip cutting")
    p.add_argument("--no-video", action="store_true", help="skip writing annotated.mp4")
    p.add_argument("--redetect", action="store_true",
                   help="skip tracking; re-run ride detection on <out>/track.csv with new thresholds")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------
# Interactive point picker
# ----------------------------------------------------------------------------

def pick_point(frame_bgr, title="Click on yourself, then press Enter (S = skip, Q = quit)"):
    """Show a frame, let the user click a point. Returns (x, y), 'skip', or 'quit'."""
    state = {"pt": None}
    win = "surf_tracker"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (x, y)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = frame_bgr.copy()
        cv2.putText(disp, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(disp, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if state["pt"] is not None:
            cv2.circle(disp, state["pt"], 8, (0, 255, 0), 2)
            cv2.circle(disp, state["pt"], 2, (0, 255, 0), -1)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and state["pt"] is not None:   # Enter
            return state["pt"]
        if k in (ord("s"), ord("S")):
            return "skip"
        if k in (ord("q"), ord("Q")):
            return "quit"


# ----------------------------------------------------------------------------
# Video helpers
# ----------------------------------------------------------------------------

def open_video(path, start_s):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    return cap, fps, n


def apply_crop(frame, crop):
    if crop is None:
        return frame
    x, y, w, h = crop
    return frame[y:y + h, x:x + w]


def prune_session(session, current_idx, keep=64):
    """Drop stored frames / non-conditioning outputs older than `keep` processed frames.

    transformers' streaming session keeps every streamed frame and every per-frame output on the
    GPU, so without this VRAM grows linearly with the recording. SAM 3 only attends to the last
    num_maskmem (7) memories and max_object_pointers_in_encoder (16) pointers, so 64 is generous.
    Conditioning frames (your clicks) are kept.
    """
    cutoff = current_idx - keep
    if cutoff <= 0:
        return
    frames = session.processed_frames
    if frames:
        for k in [k for k in frames if k < cutoff]:
            del frames[k]
    for per_obj in session.output_dict_per_obj.values():
        nc = per_obj["non_cond_frame_outputs"]
        for k in [k for k in nc if k < cutoff]:
            del nc[k]


def mask_stats(mask_np):
    """Centroid and area of a boolean mask."""
    ys, xs = np.nonzero(mask_np)
    if len(xs) == 0:
        return None, None, 0
    return float(xs.mean()), float(ys.mean()), int(len(xs))


# ----------------------------------------------------------------------------
# Preflight: fail fast with a useful message instead of a stack trace an hour in
# ----------------------------------------------------------------------------

def fail(msg):
    sys.exit("\nERROR: " + msg.strip() + "\n")


def parse_crop(crop_str, frame_w, frame_h):
    """'x,y,w,h' -> (x, y, w, h), validated against the frame."""
    if not crop_str:
        return None
    try:
        x, y, w, h = (int(v) for v in crop_str.split(","))
    except ValueError:
        fail(f"--crop must be four integers x,y,w,h (got {crop_str!r})")
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        fail(f"--crop {crop_str} does not fit inside the {frame_w}x{frame_h} video. "
             f"x+w must be <= {frame_w} and y+h must be <= {frame_h}.")
    return x, y, w, h


def pick_device_and_dtype(args):
    if not torch.cuda.is_available():
        if not args.allow_cpu:
            fail("No CUDA GPU detected. SAM 3 on CPU is far too slow for real recordings.\n"
                 "  - If you have an NVIDIA GPU, install the CUDA build of torch, e.g.\n"
                 "      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
                 "  - To try a few seconds of video anyway, add --allow-cpu.")
        print("WARNING: running on CPU (--allow-cpu). Expect well under 0.1 fps.")
        return "cpu", torch.float32
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {name} ({vram:.0f} GB)")
    if args.bf16:
        # is_bf16_supported() is True on Turing because torch emulates it; only Ampere (sm_80) and
        # newer have bf16 tensor cores, so check the compute capability instead.
        if torch.cuda.get_device_capability(0)[0] < 8:
            print(f"WARNING: {name} has no native bfloat16; it will be emulated and may be SLOWER than fp32. "
                  "Use --fp16 instead on RTX 20-series / GTX cards.")
        return "cuda", torch.bfloat16
    if args.fp16:
        return "cuda", torch.float16
    print("NOTE: running in float32. Add --fp16 (RTX 20-series) or --bf16 (RTX 30-series and newer) "
          "for a 2-3x speedup.")
    return "cuda", torch.float32


def check_hf_access(model_id):
    """SAM 3 weights are gated: the user must accept the license on HF and log in."""
    if os.path.isdir(model_id):
        return  # local checkout of the weights
    try:
        from huggingface_hub import get_token, model_info
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return
    if get_token() is None:
        fail(f"Not logged in to Hugging Face, and {model_id} is a gated model.\n"
             f"  1. Visit https://huggingface.co/{model_id} and accept the license.\n"
             "  2. Run:  hf auth login   (or:  huggingface-cli login)\n"
             "  Then re-run this command.")
    try:
        model_info(model_id)
    except GatedRepoError:
        fail(f"Your Hugging Face account has not accepted the license for {model_id}.\n"
             f"  Visit https://huggingface.co/{model_id}, click 'Agree and access repository', then re-run.")
    except RepositoryNotFoundError:
        fail(f"Model {model_id!r} not found on Hugging Face (check --model).")
    except Exception as e:  # offline, DNS, etc. - let from_pretrained use the cache if it can
        print(f"NOTE: could not reach Hugging Face ({type(e).__name__}); trying the local cache.")


def check_ffmpeg(args):
    if args.no_clips or shutil.which("ffmpeg") is not None:
        return
    print("WARNING: ffmpeg is not on PATH, so ride clips will NOT be cut (track.csv and rides.csv still will).\n"
          "  Install it (Windows: winget install Gyan.FFmpeg; macOS: brew install ffmpeg; "
          "Linux: apt install ffmpeg)\n"
          "  and re-run with --redetect afterwards to cut the clips without re-tracking.")


def fmt_eta(seconds):
    seconds = int(max(0, seconds))
    h, m = divmod(seconds // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{seconds % 60:02d}s"


# ----------------------------------------------------------------------------
# Tracking
# ----------------------------------------------------------------------------

def run_tracking(args):
    if not os.path.isfile(args.video):
        fail(f"Video not found: {args.video}")
    device, dtype = pick_device_and_dtype(args)
    check_ffmpeg(args)
    check_hf_access(args.model)

    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

    print(f"Loading {args.model} on {device} ({str(dtype).replace('torch.', '')}) ... "
          "(first run downloads ~3.5 GB)")
    model = Sam3TrackerVideoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = Sam3TrackerVideoProcessor.from_pretrained(args.model)

    cap, fps, n_total = open_video(args.video, args.start)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop = parse_crop(args.crop, frame_w, frame_h)
    end_frame = int(args.end * fps) if args.end else n_total
    start_frame = int(args.start * fps)
    if start_frame >= n_total:
        fail(f"--start {args.start}s is past the end of the video ({n_total / fps:.0f}s).")
    n_to_process = max(1, (min(end_frame, n_total) - start_frame) // args.stride)
    print(f"Video: {frame_w}x{frame_h} @ {fps:.1f} fps, {n_total / fps / 60:.1f} min. "
          f"Processing {n_to_process} frames (every {args.stride} frame(s)) "
          f"from {args.start:.0f}s to {min(end_frame, n_total) / fps:.0f}s.")
    print("Speed is roughly 1-3 fps in fp16 on a mid-range GPU and about 0.7 fps in fp32; "
          "an ETA is printed every 100 frames.")
    os.makedirs(args.out, exist_ok=True)

    # streaming session: frames are fed one at a time, so memory stays bounded
    session = processor.init_video_session(inference_device=device, dtype=dtype)

    writer = None
    track_rows = []          # (session_idx, video_frame, t, cx, cy, area, score, brightness)
    trail = []               # recent centroids for drawing
    lost_run = 0
    need_click = True        # first frame always needs a click
    skipped = False          # S pressed: stay lost quietly until R or a confident re-acquisition
    last_fix = None          # (t, cx, cy) of the last accepted frame, for the re-acquisition gate
    lost_frames = max(1, int(round(args.lost_seconds * fps / args.stride)))
    sess_idx = 0
    t0 = time.time()
    win = "surf_tracker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    with torch.inference_mode():
        while True:
            vf = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if vf >= end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if (vf - int(args.start * fps)) % args.stride != 0:
                continue

            frame = apply_crop(frame, crop)
            t_sec = vf / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, device=device, return_tensors="pt").to(model.device)
            if dtype != torch.float32:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

            # --- (re)selection ---------------------------------------------------
            if need_click:
                res = pick_point(frame)
                if res == "quit":
                    break
                if res == "skip":
                    need_click = False
                    skipped = True
                    lost_run = 0
                else:
                    x, y = res
                    processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=sess_idx,
                        obj_ids=1,
                        input_points=[[[[x, y]]]],
                        input_labels=[[[1]]],
                        original_size=tuple(int(v) for v in inputs.original_sizes[0]),
                    )
                    need_click = False
                    skipped = False
                    last_fix = None
                    lost_run = 0
                    trail.clear()

            # --- track this frame -------------------------------------------------
            if session.get_obj_num() == 0:
                # skipped before the first click: nothing to track yet
                mask_np, score, cx, cy, area = None, 0.0, None, None, 0
            else:
                # frame_idx must be explicit: without it the library derives it from
                # len(session.processed_frames), which prune_session keeps constant, so the
                # index would stop advancing and cached features would be reused forever.
                out = model(inference_session=session, frame_idx=sess_idx, frame=inputs.pixel_values[0])
                masks = processor.post_process_masks(
                    [out.pred_masks], original_sizes=inputs.original_sizes, binarize=True
                )[0]
                mask_np = masks[0, 0].cpu().numpy().astype(bool)
                score = float(out.object_score_logits.flatten()[0]) if out.object_score_logits is not None else 0.0
                cx, cy, area = mask_stats(mask_np)
                prune_session(session, sess_idx)
            # Occlusion model: a wave can hide you for a second or two, but you stay roughly where
            # you were. Accept a mask only inside a radius around the last good fix that grows with
            # time lost; anything farther is another surfer.
            present = (score >= args.min_score) and area > 0
            gate_r = None
            if last_fix is not None:
                gate_r = args.reacquire_radius + args.max_speed * max(t_sec - last_fix[0], 0.0)
                if present and float(np.hypot(cx - last_fix[1], cy - last_fix[2])) > gate_r:
                    present = False
            if not present:
                cx, cy, area = None, None, 0

            if present:
                lost_run = 0
                skipped = False
                last_fix = (t_sec, cx, cy)
                r = 20
                patch = frame[max(0, int(cy) - r):int(cy) + r, max(0, int(cx) - r):int(cx) + r]
                brightness = float(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[..., 2].mean()) if patch.size else 0.0
                trail.append((int(cx), int(cy)))
                trail = trail[-int(3 * fps / args.stride):]
            else:
                lost_run += 1
                brightness = 0.0
                if lost_run >= lost_frames and not skipped:
                    print(f"[{t_sec:7.1f}s] lost track - asking for a re-click")
                    need_click = True
                    lost_run = 0

            track_rows.append((sess_idx, vf, t_sec, cx, cy, area, score, brightness))

            # --- draw ----------------------------------------------------------------
            disp = frame.copy()
            if present:
                cnts, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(disp, cnts, -1, (0, 255, 0), 2)
                for i in range(1, len(trail)):
                    cv2.line(disp, trail[i - 1], trail[i], (0, 200, 255), 2)
                cv2.circle(disp, (int(cx), int(cy)), 6, (0, 0, 255), -1)
            elif last_fix is not None and gate_r is not None:
                # where we expect you to reappear
                cv2.circle(disp, (int(last_fix[1]), int(last_fix[2])), int(min(gate_r, 400)), (0, 255, 255), 1)
            state = "TRACKING" if present else ("skipped" if skipped else "lost")
            status = f"t={t_sec:6.1f}s  score={score:+.1f}  {state}  [R=reclick Q=quit]"
            cv2.putText(disp, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(disp, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, disp)

            if not args.no_video:
                if writer is None:
                    h, w = disp.shape[:2]
                    writer = cv2.VideoWriter(os.path.join(args.out, "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (w, h))
                writer.write(disp)

            # no target yet (S on the first frame): slow the skim to ~1/4 real time so R can be hit in time
            k = cv2.waitKey(250 if session.get_obj_num() == 0 else 1) & 0xFF
            if k in (ord("r"), ord("R")):
                need_click = True
            elif k in (ord("q"), ord("Q")):
                break

            sess_idx += 1
            if sess_idx % 100 == 0:
                el = time.time() - t0
                rate = sess_idx / el
                eta = fmt_eta((n_to_process - sess_idx) / rate) if rate > 0 else "?"
                print(f"  {sess_idx}/{n_to_process} frames, {t_sec:.0f}s of video, {rate:.1f} fps, ETA {eta}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    track_path = os.path.join(args.out, "track.csv")
    with open(track_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_idx", "video_frame", "t", "cx", "cy", "area", "score", "brightness"])
        w.writerows(track_rows)
    print(f"Wrote {track_path} ({len(track_rows)} rows)")
    return track_rows, fps


# ----------------------------------------------------------------------------
# Ride detection
# ----------------------------------------------------------------------------

def detect_rides(rows, args):
    """Find intervals of sustained fast, shoreward motion. Returns list of (start_s, end_s, peak_speed)."""
    t = np.array([r[2] for r in rows], dtype=float)
    cx = np.array([np.nan if r[3] is None else r[3] for r in rows], dtype=float)
    cy = np.array([np.nan if r[4] is None else r[4] for r in rows], dtype=float)
    if len(t) < 5:
        return []

    # interpolate short gaps, then smooth (~0.5 s median)
    def fill(a):
        idx = np.arange(len(a))
        good = ~np.isnan(a)
        return np.interp(idx, idx[good], a[good]) if good.sum() >= 2 else a

    cx, cy = fill(cx), fill(cy)
    dt = np.median(np.diff(t))
    k = max(3, int(round(0.5 / dt)) | 1)
    pad = k // 2
    sm = lambda a: np.array([np.median(a[max(0, i - pad):i + pad + 1]) for i in range(len(a))])
    cx, cy = sm(cx), sm(cy)

    vx = np.gradient(cx, t)
    vy = np.gradient(cy, t)
    speed = np.hypot(vx, vy)

    shore_ok = {
        "down": vy > 0, "up": vy < 0, "right": vx > 0, "left": vx < 0,
        "any": np.ones_like(speed, dtype=bool),
    }[args.shore]
    riding = (speed > args.ride_speed) & shore_ok

    # group consecutive True runs
    segs = []
    i = 0
    while i < len(riding):
        if riding[i]:
            j = i
            while j + 1 < len(riding) and riding[j + 1]:
                j += 1
            segs.append([t[i], t[j], float(speed[i:j + 1].max())])
            i = j + 1
        else:
            i += 1

    # merge close segments, drop short ones
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < args.merge_gap:
            merged[-1][1] = s[1]
            merged[-1][2] = max(merged[-1][2], s[2])
        else:
            merged.append(s)
    return [tuple(s) for s in merged if s[1] - s[0] >= args.min_ride]


def write_rides(rides, args):
    path = os.path.join(args.out, "rides.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ride", "start_s", "end_s", "duration_s", "peak_speed_px_s"])
        for i, (s, e, v) in enumerate(rides, 1):
            w.writerow([i, f"{s:.2f}", f"{e:.2f}", f"{e - s:.2f}", f"{v:.1f}"])
    print(f"\n{len(rides)} rides detected -> {path}")
    for i, (s, e, v) in enumerate(rides, 1):
        print(f"  ride {i:2d}: {s:7.1f}s - {e:7.1f}s  ({e - s:4.1f}s, peak {v:.0f} px/s)")


def cut_clips(rides, args):
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH; skipping clips. Install it, then re-run with --redetect to cut them.")
        return
    clip_dir = os.path.join(args.out, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    for i, (s, e, _) in enumerate(rides, 1):
        a, b = max(0, s - args.pad), e + args.pad
        dst = os.path.join(clip_dir, f"ride_{i:02d}_{int(s)}s.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.2f}", "-to", f"{b:.2f}",
               "-i", args.video, "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an", dst]
        subprocess.run(cmd)
        print(f"  wrote {dst}")


# ----------------------------------------------------------------------------

def load_track(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            cx = None if r["cx"] in ("", "None") else float(r["cx"])
            cy = None if r["cy"] in ("", "None") else float(r["cy"])
            rows.append((int(r["session_idx"]), int(r["video_frame"]), float(r["t"]), cx, cy,
                         int(r["area"]), float(r["score"]), float(r["brightness"])))
    return rows


def main():
    args = parse_args()
    if args.redetect:
        track_path = os.path.join(args.out, "track.csv")
        if not os.path.isfile(track_path):
            fail(f"--redetect needs {track_path}, but it does not exist. Run tracking first "
                 f"(same --out), or point --out at a folder that has a track.csv.")
        rows = load_track(track_path)
    else:
        rows, fps = run_tracking(args)
    rides = detect_rides(rows, args)
    write_rides(rides, args)
    if rides and not args.no_clips:
        cut_clips(rides, args)
    print("\nDone. If rides look wrong, re-run detection only on track.csv by adjusting "
          "--ride-speed / --min-ride / --shore (see README).")


if __name__ == "__main__":
    main()
