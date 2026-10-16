#!/usr/bin/env python3
"""
Standalone mini-drone arena simulator.

Window 1: 3D world (ground, blue start square, red line, red circle, drone cuboid).
Window 2: what the downward camera (at drone centre) sees.

Frame convention: NED.  x = North, y = East, z = Down (altitude = -z), ground at z = 0.
Yaw is clockwise from North (positive about Down).  Units: metres, radians.
Body frame: x forward, y right, z down.
  forward = (cos yaw, sin yaw) in (N, E);  right = (-sin yaw, cos yaw) in (N, E)
Image: top of image = drone forward, right of image = drone right.
The 3D plot is drawn as a normal map view (plot x = East, plot y = North, plot z = Up).

Controls (click a window first):
  W/S forward/back   A/D left/right   Q/E yaw left/right   R/F up/down
  [ / ]  camera FOV -/+ 5 deg (compare the camera window with the real Simulink camera)
  M toggle auto-wander   N new random arena   SPACE reset drone to blue square
"""
from dataclasses import dataclass
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

IN = 0.0254  # inch -> metre


@dataclass
class Config:
    seed: int = 0
    # arena (NED): north extent x east extent, centred on origin
    arena_n: float = 4.0
    arena_e: float = 4.0
    res: float = 0.005              # ground raster metres/pixel
    start_square: float = 0.40      # blue square side (UNKNOWN)
    square_clearance: float = 0.25  # min gap between blue square and line/circle
    line_width: float = 0.10
    n_segments: tuple = (3, 5)
    seg_len: tuple = (0.4, 0.8)
    turn_deg: tuple = (30, 330)     # heading change between segments
    circle_diameter: float = 0.20
    circle_gap: float = 0.25        # line endpoint -> circle centre
    # camera (UNKNOWN: rows/cols order and FOV)
    img_rows: int = 120
    img_cols: int = 60
    fov_deg: float = 80.0           # PLACEHOLDER, horizontal FOV across image columns
    noise_sigma: float = 4.0
    brightness: float = 1.0
    # drone
    drone_dims_in: tuple = (5.2, 5.2, 1.6)
    start_alt: float = 1.0          # initial hover height above the blue square
    # colours (RGB)
    floor: tuple = (190, 190, 190)
    red: tuple = (220, 30, 30)
    blue: tuple = (30, 30, 220)


def fov_from_measurement(px_width, altitude_m, real_width_m, cols):
    """Calibrate FOV: measure a known-width object (e.g. the 10 cm line) in pixels at a known
    altitude in the real Simulink camera image. Returns horizontal FOV in degrees."""
    f = px_width * altitude_m / real_width_m
    return float(np.rad2deg(2 * np.arctan(cols / (2 * f))))


class Arena:
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.xmin, self.xmax = -cfg.arena_n / 2, cfg.arena_n / 2   # North
        self.ymin, self.ymax = -cfg.arena_e / 2, cfg.arena_e / 2   # East
        self.rng = np.random.default_rng(seed)
        self._generate()
        self._rasterize()

    @staticmethod
    def _samples(a, b, step=0.02):
        n = max(2, int(np.linalg.norm(b - a) / step))
        return a + (b - a) * np.linspace(0, 1, n)[:, None]

    def _inside(self, p, m):
        return self.xmin + m < p[0] < self.xmax - m and self.ymin + m < p[1] < self.ymax - m

    def _valid(self, pts, circ):
        c = self.cfg
        r = c.circle_diameter / 2
        if not all(self._inside(p, 0.3) for p in pts):
            return False
        if not all(self._inside(circ + d, 0.2) for d in ([r, 0], [-r, 0], [0, r], [0, -r])):
            return False
        segs = [self._samples(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        for i in range(len(segs)):
            for j in range(i + 2, len(segs)):
                d = np.linalg.norm(segs[i][:, None] - segs[j][None], axis=2)
                if d.min() < c.line_width + 0.2:
                    return False
        for s in segs[:-1]:
            if np.linalg.norm(s - circ, axis=1).min() < r + c.line_width / 2 + 0.15:
                return False
        self._line_samples = np.vstack(segs)
        return True

    def _place_square(self, circ):
        c = self.cfg
        hs = c.start_square / 2
        reach = hs * np.sqrt(2) + c.square_clearance
        lo = np.array([self.xmin, self.ymin]) + hs + 0.2
        hi = np.array([self.xmax, self.ymax]) - hs - 0.2
        for _ in range(500):
            s = self.rng.uniform(lo, hi)
            if (np.linalg.norm(self._line_samples - s, axis=1).min() > reach + c.line_width / 2 and
                    np.linalg.norm(circ - s) > reach + c.circle_diameter / 2):
                return s
        return None

    def _generate(self):
        c = self.cfg
        for _ in range(5000):
            p0 = self.rng.uniform([self.xmin + 0.3, self.ymin + 0.3], [self.xmax - 0.3, self.ymax - 0.3])
            h = self.rng.uniform(0, 2 * np.pi)          # heading, clockwise from North
            pts = [p0]
            for i in range(self.rng.integers(c.n_segments[0], c.n_segments[1] + 1)):
                if i > 0:
                    h += np.deg2rad(self.rng.uniform(*c.turn_deg))
                pts.append(pts[-1] + self.rng.uniform(*c.seg_len) * np.array([np.cos(h), np.sin(h)]))
            circ = pts[-1] + c.circle_gap * np.array([np.cos(h), np.sin(h)])
            if not self._valid(pts, circ):
                continue
            sq = self._place_square(circ)
            if sq is None:
                continue
            self.pts, self.circ, self.square = np.array(pts), circ, sq
            return
        raise RuntimeError("could not generate arena; relax config")

    def _px(self, p):  # (north, east) -> (col, row); image up = north, right = east
        return (int(round((p[1] - self.ymin) / self.cfg.res)),
                int(round((self.xmax - p[0]) / self.cfg.res)))

    def _rasterize(self):
        c = self.cfg
        Hp, Wp = int(c.arena_n / c.res), int(c.arena_e / c.res)
        g = np.full((Hp, Wp, 3), c.floor, np.uint8)
        hs = c.start_square / 2
        sn, se = self.square
        cv2.rectangle(g, self._px((sn + hs, se - hs)), self._px((sn - hs, se + hs)), c.blue, -1)
        t = max(1, int(round(c.line_width / c.res)))
        for a, b in zip(self.pts[:-1], self.pts[1:]):
            cv2.line(g, self._px(a), self._px(b), c.red, t, cv2.LINE_AA)
        for p in self.pts:
            cv2.circle(g, self._px(p), t // 2, c.red, -1, cv2.LINE_AA)
        cv2.circle(g, self._px(self.circ), int(c.circle_diameter / 2 / c.res), c.red, -1, cv2.LINE_AA)
        self.ground = g


class Camera:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(1)

    @property
    def f(self):  # focal length in pixels
        return (self.cfg.img_cols / 2) / np.tan(np.deg2rad(self.cfg.fov_deg) / 2)

    def pixel_to_world(self, pose, u, v):
        """Pixel -> ground (north, east). pose = (x, y, z, yaw) in NED."""
        x, y, z, yaw = pose
        h = max(-z, 0.03)  # altitude above ground
        cu, cv_ = (self.cfg.img_cols - 1) / 2, (self.cfg.img_rows - 1) / 2
        right = (np.asarray(u) - cu) / self.f * h
        fwd = -(np.asarray(v) - cv_) / self.f * h
        c, s = np.cos(yaw), np.sin(yaw)
        return x + fwd * c - right * s, y + fwd * s + right * c

    def render(self, arena, pose):
        cfg = self.cfg
        u, v = np.meshgrid(np.arange(cfg.img_cols), np.arange(cfg.img_rows))
        N, E = self.pixel_to_world(pose, u, v)
        mx = ((E - arena.ymin) / cfg.res).astype(np.float32)
        my = ((arena.xmax - N) / cfg.res).astype(np.float32)
        img = cv2.remap(arena.ground, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(60, 60, 60))
        img = img.astype(np.float32) * cfg.brightness
        img += self.rng.normal(0, cfg.noise_sigma, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)  # RGB uint8 (rows, cols, 3)

    def footprint(self, pose):
        c = self.cfg
        us = [-0.5, c.img_cols - 0.5, c.img_cols - 0.5, -0.5, -0.5]
        vs = [-0.5, -0.5, c.img_rows - 0.5, c.img_rows - 0.5, -0.5]
        return self.pixel_to_world(pose, us, vs)


class Drone:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dims = np.array(cfg.drone_dims_in) * IN
        self.p = np.zeros(3); self.yaw = 0.0; self.vel = np.zeros(4)

    def reset(self, arena):
        self.p = np.array([arena.square[0], arena.square[1], -self.cfg.start_alt])
        self.yaw = 0.0
        self.vel = np.zeros(4)  # body vx(fwd), vy(right), vz(down), yaw rate (cw +)

    def step(self, cmd, dt, arena, tau=0.25):
        self.vel += (np.asarray(cmd, float) - self.vel) * min(1.0, dt / tau)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.p[0] += (self.vel[0] * c - self.vel[1] * s) * dt
        self.p[1] += (self.vel[0] * s + self.vel[1] * c) * dt
        self.p[2] += self.vel[2] * dt
        self.yaw = (self.yaw + self.vel[3] * dt + np.pi) % (2 * np.pi) - np.pi
        self.p[0] = np.clip(self.p[0], arena.xmin, arena.xmax)
        self.p[1] = np.clip(self.p[1], arena.ymin, arena.ymax)
        self.p[2] = min(self.p[2], -self.dims[2] / 2)  # ground at z = 0 (z is down)

    @property
    def pose(self):
        return (*self.p, self.yaw)

    def faces(self):
        """Cuboid faces in NED; face index 1 is the forward (+x body) face."""
        hx, hy, hz = self.dims / 2
        c = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        cs, sn = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
        v = c @ R.T + self.p
        idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
        return [v[i] for i in idx]


def plot_xyz(n, e, d=0.0):
    """NED -> plot coordinates (x=East, y=North, z=Up) for a non-mirrored map view."""
    return np.asarray(e), np.asarray(n), -np.asarray(d)


class Viewer:
    DT = 0.03

    def __init__(self, cfg):
        for k in list(plt.rcParams):
            if k.startswith("keymap."):
                plt.rcParams[k] = []
        self.cfg, self.seed = cfg, cfg.seed
        self.arena = Arena(cfg, self.seed)
        self.cam, self.drone = Camera(cfg), Drone(cfg)
        self.drone.reset(self.arena)
        self.held, self.auto, self.wp, self.trail = set(), False, None, []
        self.rng = np.random.default_rng(123)

        self.fig3 = plt.figure("World (3D, NED)", figsize=(7, 6))
        self.ax = self.fig3.add_subplot(111, projection="3d")
        self.figc = plt.figure("Camera view", figsize=(3.5, 6))
        self.axc = self.figc.add_subplot(111)
        self.im = self.axc.imshow(np.zeros((cfg.img_rows, cfg.img_cols, 3), np.uint8),
                                  interpolation="nearest")
        self.axc.plot([(cfg.img_cols - 1) / 2], [(cfg.img_rows - 1) / 2], "g+", ms=12)
        self.axc.set_xticks([]); self.axc.set_yticks([])

        self.static = []
        self._draw_arena()
        self.body = Poly3DCollection(self.drone.faces(), edgecolor="k", linewidth=0.5)
        self.ax.add_collection3d(self.body)
        self.mark, = self.ax.plot([], [], [], "ko", ms=4)
        self.drop, = self.ax.plot([], [], [], "k:", lw=0.8)
        self.foot, = self.ax.plot([], [], [], "g-", lw=1.2)
        self.tr, = self.ax.plot([], [], [], "b-", lw=0.6, alpha=0.6)

        for f in (self.fig3, self.figc):
            f.canvas.mpl_connect("key_press_event", self._kp)
            f.canvas.mpl_connect("key_release_event", self._kr)
        self.timer = self.fig3.canvas.new_timer(interval=int(self.DT * 1000))
        self.timer.add_callback(self.tick)

    def _draw_arena(self):
        for a in self.static:
            a.remove()
        self.static = []
        a, ax, c = self.arena, self.ax, self.cfg
        bx = [a.xmin, a.xmax, a.xmax, a.xmin, a.xmin]
        by = [a.ymin, a.ymin, a.ymax, a.ymax, a.ymin]
        self.static += ax.plot(*plot_xyz(bx, by, np.zeros(5)), color="gray")
        hs, (sn, se) = c.start_square / 2, a.square
        corners = [(sn + hs, se - hs), (sn + hs, se + hs), (sn - hs, se + hs), (sn - hs, se - hs)]
        sq = Poly3DCollection([[tuple(plot_xyz(n, e, 0)) for n, e in corners]],
                              facecolor="blue", alpha=0.8)
        ax.add_collection3d(sq); self.static.append(sq)
        self.static += ax.plot(*plot_xyz(a.pts[:, 0], a.pts[:, 1], np.full(len(a.pts), -0.001)),
                               "r-", lw=max(1.5, c.line_width * 100 * 0.5))
        t = np.linspace(0, 2 * np.pi, 60)
        r = c.circle_diameter / 2
        self.static += ax.plot(*plot_xyz(a.circ[0] + r * np.cos(t), a.circ[1] + r * np.sin(t),
                                         np.full(60, -0.001)), "r-", lw=3)
        ax.set_xlim(a.ymin, a.ymax); ax.set_ylim(a.xmin, a.xmax); ax.set_zlim(0, 2.5)
        ax.set_box_aspect((c.arena_e, c.arena_n, 2.5))
        ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_zlabel("Up = -z [m]")

    def _kp(self, e):
        k = (e.key or "").lower()
        if k == "m":
            self.auto = not self.auto; self.wp = None
        elif k == "n":
            self.seed += 1
            self.arena = Arena(self.cfg, self.seed); self._draw_arena()
            self.drone.reset(self.arena); self.trail.clear()
        elif k == " ":
            self.drone.reset(self.arena); self.trail.clear()
        elif k == "]":
            self.cfg.fov_deg = min(150.0, self.cfg.fov_deg + 5)
        elif k == "[":
            self.cfg.fov_deg = max(20.0, self.cfg.fov_deg - 5)
        else:
            self.held.add(k)

    def _kr(self, e):
        self.held.discard((e.key or "").lower())

    def _manual_cmd(self):
        h = self.held
        ax = lambda p, n: (p in h) - (n in h)
        # [vx fwd, vy right, vz down, yaw rate clockwise]
        return [0.6 * ax("w", "s"), 0.6 * ax("d", "a"), 0.4 * ax("f", "r"), 1.0 * ax("e", "q")]

    def _auto_cmd(self):
        d, a = self.drone, self.arena
        if self.wp is None or np.linalg.norm(self.wp[:3] - d.p) < 0.15:
            self.wp = np.array([self.rng.uniform(a.xmin + .3, a.xmax - .3),
                                self.rng.uniform(a.ymin + .3, a.ymax - .3),
                                self.rng.uniform(-1.8, -0.4), self.rng.uniform(-np.pi, np.pi)])
        vw = np.clip(1.0 * (self.wp[:3] - d.p), -0.7, 0.7)   # NED velocity
        c, s = np.cos(d.yaw), np.sin(d.yaw)
        ey = (self.wp[3] - d.yaw + np.pi) % (2 * np.pi) - np.pi
        return [vw[0] * c + vw[1] * s, -vw[0] * s + vw[1] * c, vw[2], np.clip(1.5 * ey, -1, 1)]

    def tick(self):
        d = self.drone
        d.step(self._auto_cmd() if self.auto else self._manual_cmd(), self.DT, self.arena)
        img = self.cam.render(self.arena, d.pose)
        self.im.set_data(img)
        fn, fe = self.cam.footprint(d.pose)
        self.foot.set_data_3d(*plot_xyz(fn, fe, np.zeros(5)))
        self.body.set_verts([np.c_[f[:, 1], f[:, 0], -f[:, 2]] for f in d.faces()])
        self.body.set_facecolor(["gray", "orange"] + ["lightgray"] * 4)
        self.mark.set_data_3d(*[[v] for v in plot_xyz(d.p[0], d.p[1], d.p[2])])
        self.drop.set_data_3d(*plot_xyz([d.p[0]] * 2, [d.p[1]] * 2, [0, d.p[2]]))
        self.trail.append(d.p.copy()); self.trail = self.trail[-600:]
        T = np.array(self.trail)
        self.tr.set_data_3d(*plot_xyz(T[:, 0], T[:, 1], T[:, 2]))
        self.ax.set_title(f"N={d.p[0]:+.2f} E={d.p[1]:+.2f} D={d.p[2]:+.2f} "
                          f"yaw={np.rad2deg(d.yaw):+.0f}deg  {'AUTO' if self.auto else 'MANUAL'}  seed={self.seed}",
                          fontsize=9)
        wm = 2 * max(-d.p[2], .03) * np.tan(np.deg2rad(self.cfg.fov_deg) / 2)
        self.axc.set_title(f"FOV {self.cfg.fov_deg:.0f} deg | footprint {wm:.2f} x "
                           f"{wm * self.cfg.img_rows / self.cfg.img_cols:.2f} m", fontsize=9)
        self.fig3.canvas.draw_idle(); self.figc.canvas.draw_idle()
        self.last_img = img  # don't return it: matplotlib timers inspect the return value

    def run(self):
        self.timer.start()
        plt.show()


if __name__ == "__main__":
    print(__doc__)
    Viewer(Config()).run()