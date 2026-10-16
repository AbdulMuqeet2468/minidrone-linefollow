"""Closed-loop run: simulator camera -> Mission FSM -> reference setpoint -> flight-controller model -> drone.
  python run_mission.py                 single run (seed 0) with full report
  python run_mission.py --seed 5        single run, other arena
  python run_mission.py --batch 20      many random arenas / start yaws, summary table
  python run_mission.py --live          watch it in the 3D + camera windows (N = new arena)
"""
import sys
import numpy as np
import drone_sim as ds
from mission import Mission, MissionConfig, wrap


def controller(drone, sp, kp=2.0, kz=1.5, kyaw=2.0, vmax=0.8, vzmax=0.6, wmax=1.5):
    """Stand-in for the (already working) flight controller: position/yaw reference -> body velocity."""
    c, s = np.cos(drone.yaw), np.sin(drone.yaw)
    vn, ve = np.clip(kp * (sp[0] - drone.p[0]), -vmax, vmax), np.clip(kp * (sp[1] - drone.p[1]), -vmax, vmax)
    return [c * vn + s * ve, -s * vn + c * ve, np.clip(kz * (sp[2] - drone.p[2]), -vzmax, vzmax),
            np.clip(kyaw * wrap(sp[3] - drone.yaw), -wmax, wmax)]


def dist_to_polyline(p, pts):
    best = 1e9
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a; u = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
        best = min(best, np.linalg.norm(p - (a + u * ab)))
    return best


class Recorder:
    def __init__(self, arena):
        self.arena, self.err, self.on_line, self.t0 = arena, [], False, 0.0

    def update(self, mission, pose):
        if mission.state == "FOLLOW":
            e = dist_to_polyline(np.array(pose[:2]), self.arena.pts)
            self.on_line = self.on_line or e < 0.15          # ignore the initial approach to the line
            if self.on_line and not mission.lost_active:
                self.err.append(e)

    def summary(self, mission, drone, t_end):
        tr = mission.transitions
        dur, last = {}, 0.0
        for (tt, a, b, _) in tr:
            dur[a] = dur.get(a, 0.0) + tt - last; last = tt
        land_err = float(np.linalg.norm(drone.p[:2] - self.arena.circ))
        e = np.array(self.err) if self.err else np.array([np.nan])
        ok = mission.success and land_err <= self.arena.cfg.circle_diameter / 2
        return dict(success=ok, reason=mission.fail_reason, t_total=t_end, durations=dur,
                    xtrack_mean=float(np.nanmean(e)), xtrack_rms=float(np.sqrt(np.nanmean(e ** 2))),
                    xtrack_max=float(np.nanmax(e)), n_lost=mission.n_lost, reversals=mission.reversals,
                    restarts=mission.restarts, land_err=land_err, transitions=tr)


def report_text(s, seed=None):
    L = [f"=== MISSION REPORT{'' if seed is None else f' (arena seed {seed})'} ===",
         f"Result            : {'SUCCESS - landed on circle' if s['success'] else 'FAILED ' + s['reason']}",
         f"Total time        : {s['t_total']:.1f} s"]
    L += [f"  {k:9s}: {v:5.1f} s" for k, v in s["durations"].items() if k not in ("DONE", "ABORTED")]
    L += [f"Path error (FOLLOW, after reaching the line): mean {s['xtrack_mean'] * 100:.1f} cm, "
          f"rms {s['xtrack_rms'] * 100:.1f} cm, max {s['xtrack_max'] * 100:.1f} cm",
          f"Line-loss events  : {s['n_lost']}   reversals: {s['reversals']}   search restarts: {s['restarts']}",
          f"Landing error     : {s['land_err'] * 100:.1f} cm from circle centre (circle radius 10 cm)",
          "Transitions:"]
    L += [f"  {t:6.1f}s  {a:8s} -> {b:8s} {r}" for (t, a, b, r) in s["transitions"]]
    return "\n".join(L)


def run(seed=0, cfg=None, mcfg=None, dt=0.05, t_max=300.0, yaw0=0.0, sim_fov=None):
    cfg = cfg or ds.Config(seed=seed)
    if sim_fov:
        cfg.fov_deg = sim_fov                     # reality differs from the FOV the mission believes
    arena, cam, drone = ds.Arena(cfg, seed), ds.Camera(cfg), ds.Drone(cfg)
    drone.reset(arena); drone.p[2] = -drone.dims[2] / 2; drone.yaw = yaw0   # sitting on the blue square
    m, rec, t = Mission(mcfg or MissionConfig()), None, 0.0
    rec = Recorder(arena)
    while t < t_max and not m.finished:
        sp, _ = m.step(cam.render(arena, drone.pose), drone.pose, t)
        drone.step(controller(drone, sp), dt, arena)
        rec.update(m, drone.pose); t += dt
    return rec.summary(m, drone, t)


def batch(n):
    rng = np.random.default_rng(7)
    print("seed  result  total  search  follow  xtrk_mean  xtrk_max  lost  land_err_cm")
    res = []
    for s in range(n):
        r = run(s, yaw0=rng.uniform(-np.pi, np.pi)); res.append(r)
        d = r["durations"]
        print(f"{s:4d}  {'OK  ' if r['success'] else 'FAIL'}  {r['t_total']:6.1f} {d.get('SEARCH', 0):6.1f} "
              f"{d.get('FOLLOW', 0):7.1f}  {r['xtrack_mean'] * 100:8.1f}  {r['xtrack_max'] * 100:8.1f}  "
              f"{r['n_lost']:4d}  {r['land_err'] * 100:8.1f}   {r['reason']}")
    ok = [r for r in res if r["success"]]
    print(f"\nsuccess {len(ok)}/{n}   mean time {np.mean([r['t_total'] for r in ok]):.1f} s   "
          f"mean landing error {np.mean([r['land_err'] for r in ok]) * 100:.1f} cm")


def live(seed):
    class V(ds.Viewer):
        def __init__(self, cfg):
            super().__init__(cfg); self._restart()

        def _restart(self):
            self.mission, self.t, self.rec, self._arena, self._done = Mission(), 0.0, Recorder(self.arena), self.arena, False
            self.drone.p[2] = -self.drone.dims[2] / 2; self.cmd = [0, 0, 0, 0]

        def _manual_cmd(self):
            if self.arena is not self._arena:
                self._restart()
            if not self.mission.finished:
                sp, _ = self.mission.step(self.cam.render(self.arena, self.drone.pose), self.drone.pose, self.t)
                self.cmd = controller(self.drone, sp); self.t += self.DT
                self.rec.update(self.mission, self.drone.pose)
            elif not self._done:
                self._done = True
                print(report_text(self.rec.summary(self.mission, self.drone, self.t), self.seed))
            return self.cmd

        def tick(self):
            super().tick()
            self.ax.set_title(f"{self.mission.state}  t={self.t:.1f}s  seed={self.seed}  (N = new arena)", fontsize=10)
    V(ds.Config(seed=seed)).run()


if __name__ == "__main__":
    a = sys.argv
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 0
    if "--batch" in a:
        batch(int(a[a.index("--batch") + 1]))
    elif "--live" in a:
        live(seed)
    else:
        print(report_text(run(seed), seed))