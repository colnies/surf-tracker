"""Ride detection regression test. Needs no video and no GPU.

The fixture is a real track.csv from the Belmar cam (1196x384 recording, --stride 2, --crop to the
lineup). The surfer caught a wave at about 1:48, i.e. 108 s. The test guards two things:
  * the default thresholds find that ride on a fresh install;
  * the track.csv column set stays stable (the --redetect path reads it).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import surf_tracker  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "track_belmar_ride_at_108s.csv")
TRACK_COLUMNS = ["session_idx", "video_frame", "t", "cx", "cy", "area", "score", "brightness"]


def _args(*extra):
    return surf_tracker.parse_args(["unused.mp4", "--redetect", *extra])


def test_fixture_has_expected_columns():
    with open(FIXTURE) as f:
        header = f.readline().strip().split(",")
    assert header == TRACK_COLUMNS


def test_load_track_round_trips():
    rows = surf_tracker.load_track(FIXTURE)
    assert len(rows) > 500
    assert all(len(r) == len(TRACK_COLUMNS) for r in rows)


def test_default_thresholds_find_known_ride():
    rows = surf_tracker.load_track(FIXTURE)
    rides = surf_tracker.detect_rides(rows, _args())
    starts = [s for s, _, _ in rides]
    assert any(106.0 <= s <= 110.0 for s in starts), f"known ride at ~108 s not found; got {rides}"


def test_known_ride_has_plausible_duration():
    rows = surf_tracker.load_track(FIXTURE)
    rides = surf_tracker.detect_rides(rows, _args())
    ride = next((s, e) for s, e, _ in rides if 106.0 <= s <= 110.0)
    assert 2.0 <= ride[1] - ride[0] <= 15.0


def test_high_threshold_finds_nothing():
    rows = surf_tracker.load_track(FIXTURE)
    assert surf_tracker.detect_rides(rows, _args("--ride-speed", "200")) == []
