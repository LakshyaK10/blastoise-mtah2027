"""Frame-level ball possession from automatic tracking: an HMM over H / A / D with a minimum
stoppage duration and restart-aware transitions.

Pipeline
--------
1. Read the tactical ATD (players + ball) and the Smart Tagging.
2. Clean the ball track: link detections into physically possible trajectories, drop blips, ignore
   tracks that never touch the pitch (spare balls held by ball boys), interpolate short gaps.
3. Per frame, compute evidence:
     ownership  -- the nearest player's team, if that player is close enough to control the ball,
                   clearly closer than any opponent, and the ball is not in fast flight
     dead ball  -- collective player speed, ball crossing a line, ball lost after crossing,
                   stationary ball with slow players, kicked ball (evidence of live play)
4. Decode each graded half with Viterbi. States: H, A, and a chain D_1..D_K that forces every
   stoppage to last at least K frames. The cost of leaving a stoppage depends on which team
   controlled the dead ball just before the restart (the taker).
5. Frames outside the graded halves are written as X.

Every parameter is set from football/physical reasoning or from label-free measurements of the
tracking itself (documented in TECHNICAL.md); none is fitted to any ground truth. Emission scores
are rounded to integers before decoding, so the output is reproducible byte for byte.
"""

import io
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAMES, half_video_blocks, require, source_frames  # noqa: E402
from load_smart_tagging import load as load_smart_tagging  # noqa: E402

FPS = 25
H, A = 0, 1

PARAMS = dict(
    # --- ball track cleaning
    max_ball_speed=40.0,    # m/s. Faster than any football; a jump implying more is a glitch.
    max_link_gap=8,         # frames. Detections further apart start a new track.
    min_track_len=4,        # detections. Shorter tracks are treated as blips.
    off_pitch_margin=1.0,   # m. A track never inside the lines (by this much) is not the match ball.
    # --- ownership (H-over-A log-odds per frame)
    r_near=1.0,             # m. Within this, full control evidence (median control distance ~1.1 m).
    r_far=2.5,              # m. Beyond this, no control evidence.
    margin_full=1.0,        # m. Opponent this much further away -> uncontested.
    v_control=8.0,          # m/s. Up to this speed the ball can be at a player's feet.
    v_flight=12.0,          # m/s. Above this the ball is in flight (nearest player median 3-5 m).
    flight_weight=0.3,
    interp_weight=0.5,      # evidence weight on interpolated ball positions
    k_own=1.0,
    tag_rate=0.5,           # per second, Smart Tagging phase prior
    # --- dead ball (per second)
    s_mid=1.3,              # m/s. Mean player speed between stoppages (~0.5-1.1) and play (~2-3.5).
    a_speed=2.0,
    speed_clip=(-3.0, 1.5),
    smooth_speed=25,        # frames, +/- 1 s
    out_margin=0.5,         # m beyond a line
    out_rate=25.0,          # ball seen beyond a line on a track that came from the pitch
    lost_out_rate=2.5,      # ball lost after crossing a line, until it is seen on the pitch again
    exit_horizon=0.6,       # s. A track that ends heading out of play this close to a line has left it
    exit_min_speed=2.0,     # m/s
    kick_speed=8.0,
    kick_rate=-5.0,         # struck ball on the pitch: play is live
    still_speed=0.5,        # m/s
    still_rate=1.0,         # stationary ball and slow players: a restart is being set up
    unseen_rate=0.5,
    # --- decoding
    p_switch=8.0,           # H <-> A: ~0.3 s of clear control by the other team
    p_to_dead=6.0,
    p_from_dead=6.0,
    min_dead_s=2.5,         # a stoppage lasts at least this long (fetch, place, restart)
    taker_window=50,        # frames before a restart in which the taker's control is measured
    taker_cap=10.0,         # cap on the tracking part of the taker evidence
    taker_total_cap=12.0,
    c_taker=1.0,
    tag_taker=10.0,         # Smart Tagging set-piece team, as taker evidence (> p_switch: the
                            # restarting team always gets the ball first, as the laws require)
    pos_taker_rate=3.0,     # per second a player of only one team stands where restarts are taken
                            # (beyond a touchline: throw-in; at a corner flag: corner kick)
    pos_touchline=0.5,      # m beyond the touchline
    pos_corner=2.5,         # m from the corner flag
    tag_taker_window=(-50, 375),  # frames around the tag start where the restart can fall
    # --- kickoffs
    kickoff_search_after=75,
    kickoff_spot_tolerance=0.5,
    kickoff_radius=6.0,
    scale=1000,
)


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def team_of_each_track(meta_path):
    """0 for home slots, 1 for away. The first <Team> in the EPTS metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}".format(meta_path))
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([H if t == team_ids[0] else A for t in slots], dtype=np.int8)


def load_atd(raw_path, n_frames):
    """(players[n, slots, 3], ball[n, 3]) indexed by video frame; x, y normalised, speed m/s."""
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        float_precision="round_trip").to_numpy()
    frame_id = table[:, 0].astype(np.int64)
    n_slots = (table.shape[1] - 1 - 3) // 3
    players = np.full((n_frames, n_slots, 3), np.nan)
    ball = np.full((n_frames, 3), np.nan)
    rows = frame_id - 1  # ATD frame_id counts from 1, video frames from 0
    keep = (rows >= 0) & (rows < n_frames)
    players[rows[keep]] = table[keep, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball[rows[keep]] = table[keep, 1 + 3 * n_slots:1 + 3 * n_slots + 3]
    return players, ball


# ---------------------------------------------------------------------------------------------
# Ball track cleaning
# ---------------------------------------------------------------------------------------------
def ball_tracks(bx, by, p):
    """Split the detections into physically possible tracks: [array of frame indices]."""
    idx = np.where(np.isfinite(bx) & np.isfinite(by))[0]
    if len(idx) == 0:
        return []
    gap = np.diff(idx)
    jump = np.hypot(np.diff(bx[idx]), np.diff(by[idx]))
    linked = (gap <= p["max_link_gap"]) & (jump <= p["max_ball_speed"] * gap / FPS)
    cuts = np.where(~linked)[0] + 1
    return np.split(idx, cuts)


def clean_ball(bx, by, pitch, p):
    """Cleaned ball: positions (interpolated inside tracks), flags, and crossing information."""
    L, W = pitch
    n = len(bx)
    cx, cy = np.full(n, np.nan), np.full(n, np.nan)
    seen = np.zeros(n, bool)       # a real detection on a kept track
    interp = np.zeros(n, bool)     # an interpolated position inside a kept track
    out = np.zeros(n, bool)        # seen beyond a line, on a track that has been on the pitch
    lost_out = np.zeros(n, bool)   # after a track left the pitch, until the ball is seen on it again
    m_out, m_off = p["out_margin"], p["off_pitch_margin"]

    def inside(x, y, margin):
        return (x >= -margin) & (x <= L + margin) & (y >= -margin) & (y <= W + margin)

    kept = []
    for tr in ball_tracks(bx, by, p):
        if len(tr) < p["min_track_len"]:
            continue
        on_pitch = inside(bx[tr], by[tr], -m_off)  # at least m_off inside the lines
        if not on_pitch.any():
            continue  # never on the pitch: not the match ball
        kept.append(tr)
        a, b = tr[0], tr[-1]
        frames = np.arange(a, b + 1)
        cx[a:b + 1] = np.interp(frames, tr, bx[tr])
        cy[a:b + 1] = np.interp(frames, tr, by[tr])
        seen[tr] = True
        interp[a:b + 1] = True
        beyond = ~inside(bx[tr], by[tr], m_out)
        out[tr[beyond]] = True
    interp &= ~seen

    # Lost after crossing: from the end of a track whose last detection is beyond a line, to the
    # next detection on the pitch.
    on_pitch_seen = seen & inside(np.nan_to_num(cx, nan=-1e9), np.nan_to_num(cy, nan=-1e9), 0.0)
    next_on = np.full(n + 1, n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        next_on[i] = i if on_pitch_seen[i] else next_on[i + 1]
    n_heading_out = 0
    for tr in kept:
        e = tr[-1]
        if out[e]:
            lost_out[e + 1:next_on[e + 1]] = True
        elif len(tr) >= 3:
            # Track ends on the pitch: has the ball left it? Extrapolate its last velocity.
            j = tr[max(0, len(tr) - 5)]
            dt = (e - j) / FPS
            vx, vy = (bx[e] - bx[j]) / dt, (by[e] - by[j]) / dt
            if np.hypot(vx, vy) >= p["exit_min_speed"]:
                tx = (bx[e] / -vx) if vx < 0 else ((L - bx[e]) / vx if vx > 0 else np.inf)
                ty = (by[e] / -vy) if vy < 0 else ((W - by[e]) / vy if vy > 0 else np.inf)
                if min(tx, ty) <= p["exit_horizon"]:
                    lost_out[e + 1:next_on[e + 1]] = True
                    n_heading_out += 1
    return dict(bx=cx, by=cy, seen=seen, interp=interp, out=out, lost_out=lost_out,
                n_tracks=len(kept), n_heading_out=n_heading_out)


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------
def moving_nanmean(x, half_width):
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half_width)
    hi = np.minimum(len(x), i + half_width + 1)
    cnt = cn[hi] - cn[lo]
    out = np.full(len(x), np.nan)
    out[cnt > 0] = (cs[hi] - cs[lo])[cnt > 0] / cnt[cnt > 0]
    return out


def rolling_nanmedian(x, half_width):
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(x, half_width, constant_values=np.nan), 2 * half_width + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(windows, axis=1)


def compute_features(players, ball, team, pitch, p=PARAMS):
    L, W = pitch
    b = clean_ball(ball[:, 0] * L, ball[:, 1] * W, pitch, p)
    bx, by = b["bx"], b["by"]
    have = b["seen"] | b["interp"]
    px, py = players[:, :, 0] * L, players[:, :, 1] * W
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d_home = np.min(np.where(team[None, :] == H, dist, np.inf), axis=1)
    d_away = np.min(np.where(team[None, :] == A, dist, np.inf), axis=1)

    speed = np.full(len(bx), np.nan)
    ok = have[2:] & have[:-2]
    sp = np.hypot(bx[2:] - bx[:-2], by[2:] - by[:-2]) * FPS / 2.0
    speed[1:-1] = np.where(ok, sp, np.nan)
    ball_speed = rolling_nanmedian(speed, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_speed = np.nanmean(players[:, :, 2], axis=1)
    mean_speed_s = moving_nanmean(mean_speed, p["smooth_speed"])
    still = moving_nanmean(ball_speed, 12)  # +/- 0.5 s

    on_pitch = have & (bx >= 0) & (bx <= L) & (by >= 0) & (by <= W)

    # Restart positions: a player beyond a touchline (thrower) or at a corner flag (corner taker).
    with np.errstate(invalid="ignore"):
        beyond = ((py < -p["pos_touchline"]) | (py > W + p["pos_touchline"])) & (px > 0) & (px < L)
        flag = np.zeros_like(beyond)
        for cx_, cy_ in ((0.0, 0.0), (0.0, W), (L, 0.0), (L, W)):
            flag |= np.hypot(px - cx_, py - cy_) < p["pos_corner"]
    at_restart = beyond | flag
    home_there = (at_restart & (team[None, :] == H)).any(axis=1)
    away_there = (at_restart & (team[None, :] == A)).any(axis=1)
    restart_pos = home_there.astype(float) - away_there.astype(float)  # 0 when both or neither

    return dict(restart_pos=restart_pos, bx=bx, by=by, seen=b["seen"], interp=b["interp"], have=have, out=b["out"],
                lost_out=b["lost_out"], on_pitch=on_pitch, d_home=d_home, d_away=d_away,
                ball_speed=ball_speed, ball_still=still, mean_speed_s=mean_speed_s,
                pitch=pitch, n_tracks=b["n_tracks"], n_heading_out=b["n_heading_out"])


def ownership_evidence(f, p=PARAMS):
    """Per-frame log-odds for H over A. Positive = home control, negative = away."""
    d_h, d_a = f["d_home"], f["d_away"]
    d1 = np.minimum(d_h, d_a)
    with np.errstate(invalid="ignore"):
        margin = np.abs(d_h - d_a)
    margin = np.nan_to_num(margin, nan=0.0, posinf=1e9)
    sign = np.where(d_h < d_a, 1.0, -1.0)
    w_dist = np.clip((p["r_far"] - d1) / (p["r_far"] - p["r_near"]), 0.0, 1.0)
    w_margin = np.clip(margin / p["margin_full"], 0.0, 1.0)
    v = np.nan_to_num(f["ball_speed"], nan=0.0)
    w_speed = np.interp(v, [p["v_control"], p["v_flight"]], [1.0, p["flight_weight"]])
    w_src = np.where(f["seen"], 1.0, np.where(f["interp"], p["interp_weight"], 0.0))
    w = w_dist * w_margin * w_speed * w_src
    w = np.where(f["have"] & np.isfinite(d1), w, 0.0)
    return p["k_own"] * sign * w


def dead_evidence(f, p=PARAMS):
    """Per-frame log-odds for dead ball over live play."""
    lo, hi = p["speed_clip"]
    spd = f["mean_speed_s"]
    rate = np.where(np.isfinite(spd),
                    np.clip(p["a_speed"] * (p["s_mid"] - np.nan_to_num(spd)), lo, hi), 0.0)
    rate = rate + np.where(f["out"], p["out_rate"], 0.0)
    rate = rate + np.where(f["lost_out"] & ~f["have"], p["lost_out_rate"], 0.0)
    v = np.nan_to_num(f["ball_speed"], nan=0.0)
    kicked = f["seen"] & f["on_pitch"] & (v > p["kick_speed"])
    rate = rate + np.where(kicked, p["kick_rate"], 0.0)
    still = f["seen"] & f["on_pitch"] & (np.nan_to_num(f["ball_still"], nan=99.0) < p["still_speed"]) \
        & (np.nan_to_num(spd, nan=99.0) < p["s_mid"])
    rate = rate + np.where(still, p["still_rate"], 0.0)
    rate = rate + np.where(~f["have"], p["unseen_rate"], 0.0)
    return rate / FPS


# ---------------------------------------------------------------------------------------------
# Smart Tagging
# ---------------------------------------------------------------------------------------------
HOME_NAMES = {"barcelona"}
AWAY_NAMES = {"atletic club", "athletic club"}
IN_POSSESSION = {"build up", "progression", "creating chances", "attacking transition",
                 "long ball", "ball in final third", "ball in the box", "players in the box"}
OUT_OF_POSSESSION = {"defending in attacking third", "defending in middle third",
                     "defending in defensive third", "defensive transition"}


def tag_side(tags):
    team = str(tags.get("Team", "")).strip().lower()
    return +1 if team in HOME_NAMES else (-1 if team in AWAY_NAMES else 0)


def tag_ownership_prior(instances, n_frames, p=PARAMS):
    acc = np.zeros(n_frames + 1)
    for inst in instances:
        code, side = inst["code"].strip().lower(), tag_side(inst["tags"])
        if side == 0:
            continue
        sign = side if code in IN_POSSESSION else (-side if code in OUT_OF_POSSESSION else 0)
        a, b = max(0, inst["start_frame"]), min(n_frames, inst["end_frame"])
        if sign and b > a:
            acc[a] += sign
            acc[b] -= sign
    return np.sign(np.cumsum(acc)[:n_frames]) * p["tag_rate"] / FPS


def tag_taker_prior(instances, n_frames, p=PARAMS):
    """+1/-1 where a tagged set piece (corner, goal kick, free kick) names the restarting team."""
    out = np.zeros(n_frames)
    lo, hi = p["tag_taker_window"]
    for inst in instances:
        if inst["code"].strip().lower() != "set pieces":
            continue
        side = tag_side(inst["tags"])
        if side:
            a = max(0, inst["start_frame"] + lo)
            b = min(n_frames, inst["start_frame"] + hi)
            out[a:b] = side
    return out


def detect_kickoffs(instances, f, p=PARAMS):
    """[(kind, tag_start, kick_frame)]: the frame the ball leaves the centre spot."""
    L, W = f["pitch"]
    out = []
    for inst in instances:
        kind = str(inst["tags"].get("Type", "")).strip().lower()
        if inst["code"].strip().lower() != "set pieces" or not kind.startswith("kick off"):
            continue
        a, b = inst["start_frame"], inst["end_frame"] + p["kickoff_search_after"]
        idx = np.arange(a, min(b, len(f["seen"])))
        dist = np.hypot(f["bx"][idx] - L / 2, f["by"][idx] - W / 2)
        dist = np.where(f["seen"][idx], dist, np.inf)
        if np.min(dist) < p["kickoff_radius"]:
            i = int(np.argmin(dist))
            while i + 1 < len(idx) and dist[i + 1] <= dist.min() + p["kickoff_spot_tolerance"]:
                i += 1
            kick = int(idx[i])
        else:
            up = idx[np.nan_to_num(f["mean_speed_s"][idx], nan=0.0) > p["s_mid"]]
            kick = int(up[0]) if len(up) else (inst["start_frame"] + inst["end_frame"]) // 2
        out.append((kind, inst["start_frame"], kick))
    return out


# ---------------------------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------------------------
def viterbi_min_dead(e_h, e_a, e_d, exit_h, exit_a, p_switch, p_in, k):
    """Max-sum decode over H, A and a chain D_1..D_k (a stoppage lasts at least k frames).

    All inputs are integers, so the result is exact. Leaving the stoppage (D_k -> H or A) costs
    exit_h[t] / exit_a[t], which carry the taker's evidence. Returns codes 0=H, 1=A, 2=D.
    """
    n = len(e_h)
    neg = -(1 << 62)
    s_h, s_a = int(e_h[0]), int(e_a[0])
    s_d = np.full(k, int(e_d[0]), dtype=np.int64)  # a half may open mid-stoppage
    bp_h = np.zeros(n, np.int8)    # 0 from H, 1 from A, 2 from D_k
    bp_a = np.zeros(n, np.int8)    # 0 from A, 1 from H, 2 from D_k
    bp_d1 = np.zeros(n, np.int8)   # 0 from H, 1 from A
    bp_dk = np.zeros(n, np.int8)   # 0 from D_k (stay), 1 from D_{k-1}
    for t in range(1, n):
        dk = int(s_d[-1])
        # H
        c0, c1, c2 = s_h, s_a - p_switch, dk - int(exit_h[t])
        if c0 >= c1 and c0 >= c2:
            nh, bh = c0, 0
        elif c1 >= c2:
            nh, bh = c1, 1
        else:
            nh, bh = c2, 2
        # A
        c0, c1, c2 = s_a, s_h - p_switch, dk - int(exit_a[t])
        if c0 >= c1 and c0 >= c2:
            na, ba = c0, 0
        elif c1 >= c2:
            na, ba = c1, 1
        else:
            na, ba = c2, 2
        # D chain
        if s_h >= s_a:
            nd1, bd1 = s_h - p_in, 0
        else:
            nd1, bd1 = s_a - p_in, 1
        prev_last = int(s_d[-2]) if k > 1 else neg
        if dk >= prev_last:
            ndk, bdk = dk, 0
        else:
            ndk, bdk = prev_last, 1
        ed = int(e_d[t])
        s_d[1:] = s_d[:-1]
        s_d[0] = nd1
        s_d[-1] = ndk
        s_d += ed
        s_h, s_a = nh + int(e_h[t]), na + int(e_a[t])
        bp_h[t], bp_a[t], bp_d1[t], bp_dk[t] = bh, ba, bd1, bdk

    # backtrack; state encoding: -1 = H, -2 = A, i >= 0 = D_{i+1}
    finals = [s_h, s_a] + [int(x) for x in s_d]
    j = int(np.argmax(finals))
    state = -1 if j == 0 else (-2 if j == 1 else j - 2)
    path = np.zeros(n, np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = 0 if state == -1 else (1 if state == -2 else 2)
        if t == 0:
            break
        if state == -1:
            state = [-1, -2, k - 1][bp_h[t]]
        elif state == -2:
            state = [-2, -1, k - 1][bp_a[t]]
        elif state == 0:
            state = [-1, -2][bp_d1[t]]
        elif state == k - 1:
            state = k - 1 if bp_dk[t] == 0 else k - 2
        else:
            state = state - 1
    return path


def centre_switches(labels, own, a, b):
    """Move each H<->A switch that falls inside an evidence-free stretch to its midpoint.

    Inside such a stretch every switch position scores the same, so this does not change the
    Viterbi optimum; it only replaces an arbitrary tie-break with the one that minimises the
    worst-case boundary error.
    """
    lab = labels[a:b + 1].copy()
    free = np.abs(own[a:b + 1]) < 1e-12
    t = 1
    while t < len(lab):
        if lab[t] != lab[t - 1] and lab[t] in "HA" and lab[t - 1] in "HA" and free[t]:
            lo = t
            while lo - 1 >= 0 and free[lo - 1] and lab[lo - 1] == lab[t - 1]:
                lo -= 1
            hi = t
            while hi + 1 < len(lab) and free[hi + 1] and lab[hi + 1] == lab[t]:
                hi += 1
            mid = (lo + hi + 1) // 2
            old, new = lab[t - 1], lab[t]
            lab[lo:mid] = old
            lab[mid:hi + 1] = new
            t = hi + 1
        else:
            t += 1
    labels[a:b + 1] = lab


def predict(game="BAR-ATH", data_dir=None, p=PARAMS, use_tags=True, verbose=True, return_debug=False):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    team = team_of_each_track(meta_path)
    if verbose:
        print("reading tracking ...")
    players, ball = load_atd(raw_path, n_frames)
    f = compute_features(players, ball, team, pitch, p)
    instances = load_smart_tagging(game, data_dir)
    if verbose:
        print("ball: {} raw detections -> {} kept on {} tracks, {} interpolated frames, "
              "{} tracks end heading out of play".format(
                  int(np.isfinite(ball[:, 0]).sum()), int(f["seen"].sum()), f["n_tracks"],
                  int(f["interp"].sum()), f["n_heading_out"]))

    own_raw = ownership_evidence(f, p)
    own = own_raw + (tag_ownership_prior(instances, n_frames, p) if use_tags else 0.0)
    dead = dead_evidence(f, p)

    # Taker evidence: control of the dead ball in the window before a restart.
    cs = np.concatenate(([0.0], np.cumsum(own_raw)))
    i = np.arange(n_frames)
    lo = np.maximum(0, i - p["taker_window"])
    taker = np.clip(cs[i + 1] - cs[lo], -p["taker_cap"], p["taker_cap"])
    cp = np.concatenate(([0.0], np.cumsum(f["restart_pos"])))
    taker = taker + p["pos_taker_rate"] * (cp[i + 1] - cp[lo]) / FPS
    if use_tags:
        taker = taker + p["tag_taker"] * tag_taker_prior(instances, n_frames, p)
    taker = np.clip(taker, -p["taker_total_cap"], p["taker_total_cap"])

    kickoffs = detect_kickoffs(instances, f, p)
    blocks = half_video_blocks(game)
    forced_dead = np.zeros(n_frames, dtype=bool)
    for kind, tag_start, kick in kickoffs:
        if kind == "kick off (start)":
            start = next((a for _, a, b in blocks if a <= kick <= b), tag_start)
        else:
            start = tag_start
        forced_dead[start:kick] = True
    if verbose:
        for kk in kickoffs:
            print("kickoff: {:24s} tag starts {:6d}  kick detected at {:6d}".format(*kk))

    s = p["scale"]
    own_i = np.rint(own * s).astype(np.int64)
    e_h = np.minimum(0, own_i)          # the favoured team pays nothing, the other pays |own|
    e_a = np.minimum(0, -own_i)
    e_d = np.rint(dead * s).astype(np.int64)
    big = 10 ** 9
    e_h[forced_dead] -= big
    e_a[forced_dead] -= big
    exit_base = p["p_from_dead"] * s
    exit_h = np.rint(exit_base + p["c_taker"] * np.maximum(0.0, -taker) * s).astype(np.int64)
    exit_a = np.rint(exit_base + p["c_taker"] * np.maximum(0.0, taker) * s).astype(np.int64)
    k = int(round(p["min_dead_s"] * FPS))

    labels = np.full(n_frames, "X", dtype="<U1")
    names = np.array(["H", "A", "D"])
    for _, a, b in blocks:
        path = viterbi_min_dead(e_h[a:b + 1], e_a[a:b + 1], e_d[a:b + 1], exit_h[a:b + 1],
                                exit_a[a:b + 1], int(round(p["p_switch"] * s)),
                                int(round(p["p_to_dead"] * s)), k)
        labels[a:b + 1] = names[path]
        centre_switches(labels, own, a, b)
    if return_debug:
        return labels, dict(f=f, own=own, own_raw=own_raw, dead=dead, taker=taker)
    return labels


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path
