"""Perception, step 1: red / blue segmentation. Pure functions, no dependency on the simulator."""
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class PerceptionConfig:
    # OpenCV HSV ranges: H 0-179, S/V 0-255. Red hue wraps around 0.
    red_h_lo: int = 10        # red = H <= red_h_lo or H >= red_h_hi
    red_h_hi: int = 170
    blue_h: tuple = (100, 130)
    s_min: int = 100
    v_min: int = 60
    open_k: int = 2           # morphological opening kernel (noise removal)
    min_pixels: int = 15      # below this a colour counts as "not detected"
    # camera model (must match the real camera; FOV is still a placeholder)
    img_rows: int = 120
    img_cols: int = 60
    fov_deg: float = 80.0
    # blob classification
    min_blob_px: int = 12
    circle_diam: float = 0.20
    circle_tol: tuple = (0.6, 1.5)     # allowed equivalent-diameter ratio
    circle_max_aspect: float = 1.5     # minAreaRect long/short
    circle_fill: tuple = (0.6, 0.9)    # blob area / minAreaRect area (disc ~0.785, bar ~1.0)


def segment(img_rgb, cfg=PerceptionConfig()):
    """RGB uint8 image -> (red_mask, blue_mask), uint8 0/255, same size as the image."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sv = (s >= cfg.s_min) & (v >= cfg.v_min)
    red = sv & ((h <= cfg.red_h_lo) | (h >= cfg.red_h_hi))
    blue = sv & (h >= cfg.blue_h[0]) & (h <= cfg.blue_h[1])
    k = np.ones((cfg.open_k, cfg.open_k), np.uint8)
    out = []
    for m in (red, blue):
        out.append(cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, k))
    return out[0], out[1]


def mask_stats(mask, cfg=PerceptionConfig()):
    """-> dict(detected, area, centroid=(row, col) or None). Pixel coordinates, origin top-left."""
    area = int(np.count_nonzero(mask))
    if area < cfg.min_pixels:
        return dict(detected=False, area=area, centroid=None)
    r, c = np.nonzero(mask)
    return dict(detected=True, area=area, centroid=(float(r.mean()), float(c.mean())))


def focal(cfg):
    return (cfg.img_cols / 2) / np.tan(np.deg2rad(cfg.fov_deg) / 2)


def pix_to_body(rows, cols, alt, cfg):
    """Pixel (row, col) -> ground point in body frame (forward, right), metres."""
    f = focal(cfg)
    cu, cv_ = (cfg.img_cols - 1) / 2, (cfg.img_rows - 1) / 2
    h = max(alt, 0.03)
    return -(np.asarray(rows) - cv_) / f * h, (np.asarray(cols) - cu) / f * h


def body_to_world(fwd, right, pose):
    """Body (forward, right) -> NED (north, east) using pose = (x, y, z, yaw)."""
    x, y, _, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return x + fwd * c - right * s, y + fwd * s + right * c


def analyze(img, pose, cfg=PerceptionConfig()):
    """Full per-frame perception. Returns dict:
       blobs      : list of red blobs (area, length/width [m], is_circle, touches_border, centroid body+world)
       circle     : best circle blob or None
       line_world : (N,2) world (north, east) coords of red pixels NOT classified as circle (sub-sampled)
       blue_area  : blue pixel count"""
    red, blue = segment(img, cfg)
    alt = -pose[2]
    scale = max(alt, 0.03) / focal(cfg)          # metres per pixel
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    blobs, lr, lc = [], [], []
    line_mask = np.zeros(red.shape, bool)
    for i in range(1, nlab):
        x0, y0, w, h, area = [int(v) for v in stats[i]]
        if area < cfg.min_blob_px:
            continue
        ys, xs = np.nonzero(lab[y0:y0 + h, x0:x0 + w] == i)
        ys, xs = ys + y0, xs + x0
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        long_, short_ = max(rw, rh) + 1, min(rw, rh) + 1
        aspect, fill = long_ / short_, area / (long_ * short_)
        eq_d = np.sqrt(4 * area / np.pi) * scale
        touches = x0 == 0 or y0 == 0 or x0 + w >= cfg.img_cols or y0 + h >= cfg.img_rows
        is_circle = (not touches and aspect <= cfg.circle_max_aspect
                     and cfg.circle_fill[0] <= fill <= cfg.circle_fill[1]
                     and cfg.circle_tol[0] * cfg.circle_diam <= eq_d <= cfg.circle_tol[1] * cfg.circle_diam)
        fwd, right = pix_to_body(ys.mean(), xs.mean(), alt, cfg)
        N, E = body_to_world(fwd, right, pose)
        blobs.append(dict(area=area, is_circle=bool(is_circle), touches=bool(touches), length=long_ * scale,
                          width=short_ * scale, eq_d=eq_d, fwd=float(fwd), right=float(right), N=float(N), E=float(E)))
        if not is_circle:
            lr.append(ys[::2]); lc.append(xs[::2]); line_mask[ys, xs] = True
    circles = [b for b in blobs if b["is_circle"]]
    if lr:
        f_, r_ = pix_to_body(np.concatenate(lr), np.concatenate(lc), alt, cfg)
        N_, E_ = body_to_world(f_, r_, pose)
        line_world = np.column_stack([N_, E_])
    else:
        line_world = np.zeros((0, 2))
    return dict(blobs=blobs, circle=max(circles, key=lambda b: b["area"]) if circles else None,
                line_world=line_world, line_mask=line_mask, blue_area=int(np.count_nonzero(blue)), red=red, blue=blue)