"""Mission finite state machine + guidance. Inputs: RGB image, pose (x,y,z,yaw) NED, time.
Output: reference setpoint (north, east, down, yaw) -- same interface as the Simulink reference.
States: INIT -> TAKEOFF -> SEARCH -> FOLLOW -> APPROACH -> ALIGN -> LAND -> DONE ;  any -> ABORT -> ABORTED"""
from dataclasses import dataclass, field
import numpy as np
import cv2
from perception import PerceptionConfig, analyze, focal, pix_to_body, body_to_world


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class MissionConfig:
    alt: float = 1.0                  # cruise altitude
    align_alt: float = 0.8
    v_search: float = 0.5
    v_follow: float = 0.35
    v_approach: float = 0.4
    v_climb: float = 0.4
    v_descend: float = 0.2
    yaw_rate: float = 1.0             # max reference yaw rate [rad/s]
    max_lead: float = 0.4             # reference may lead the vehicle by at most this [m]
    search_bounds: tuple = (-1.6, 1.6, -1.6, 1.6)   # N_min, N_max, E_min, E_max
    lane_overlap: float = 0.7         # lane spacing = overlap * footprint width
    search_confirm_s: float = 0.2     # line must be seen this long to leave SEARCH
    line_min_len: float = 0.30        # [m] min blob length to count as "line" (rejects partial circle)
    r_min: float = 0.25               # follow look-ahead distance [m]
    band: float = 0.12                # look-ahead band thickness [m]
    r_max: float = 0.7                # line farther than this does not count as 'ahead' (once on the line)
    branch_gap: float = 0.5           # [rad] angular gap that separates two line branches
    visit_r: float = 0.15             # line within this distance ALONG the line of the vehicle counts as followed
    end_hold_s: float = 0.6           # no unvisited line for this long => end of line
    max_reversals: int = 1
    circle_confirm: int = 4           # consecutive frames
    circle_near: float = 0.8          # circle must be within this of the line end
    lost_hold_s: float = 1.0
    lost_giveup_s: float = 5.0
    lost_alt: float = 1.6             # climb here while the line is lost
    max_restarts: int = 2             # SEARCH restarts before ABORT
    align_tol: float = 0.05
    align_hold_s: float = 1.0
    land_err_max: float = 0.08        # descend only while centre error below this
    blind_alt: float = 0.15           # below this circle may not fit the image: descend blind
    landed_alt: float = 0.045
    init_blue_s: float = 0.5
    takeoff_tol: float = 0.05
    takeoff_hold_s: float = 1.0
    timeouts: dict = field(default_factory=lambda: dict(
        INIT=5, TAKEOFF=15, SEARCH=120, FOLLOW=150, APPROACH=25, ALIGN=25, LAND=30, ABORT=30))
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)


def geodesic(mask, seed, max_iter):
    """Distance (in pixel steps) along the line mask from the seed pixels; -1 = unreached."""
    dist = np.full(mask.shape, -1, int)
    reached = seed & mask
    dist[reached] = 0
    k = np.ones((3, 3), np.uint8)
    for i in range(1, max_iter + 1):
        new = cv2.dilate(reached.astype(np.uint8), k).astype(bool) & mask & ~reached
        if not new.any():
            break
        dist[new] = i
        reached |= new
    return dist


class Visited:
    """Coarse world grid marking line pixels that were already followed."""
    def __init__(self, bounds, res=0.02, margin=1.0):
        self.n0, self.e0, self.res = bounds[0] - margin, bounds[2] - margin, res
        self.m = np.zeros((int((bounds[1] - bounds[0] + 2 * margin) / res),
                           int((bounds[3] - bounds[2] + 2 * margin) / res)), np.uint8)

    def _idx(self, P):
        return ((P[:, 0] - self.n0) / self.res).astype(int), ((P[:, 1] - self.e0) / self.res).astype(int)

    def clear(self):
        self.m[:] = 0

    def mark_disc(self, p, r):
        cv2.circle(self.m, (int((p[1] - self.e0) / self.res), int((p[0] - self.n0) / self.res)),
                   int(r / self.res), 1, -1)

    def mark_points(self, P, r=0.04):
        if len(P) == 0:
            return
        i, j = self._idx(P)
        ok = (i >= 0) & (j >= 0) & (i < self.m.shape[0]) & (j < self.m.shape[1])
        tmp = np.zeros_like(self.m)
        tmp[i[ok], j[ok]] = 1
        k = int(r / self.res) * 2 + 1
        self.m |= cv2.dilate(tmp, np.ones((k, k), np.uint8))   # dilate only the new points

    def check(self, P):
        i, j = self._idx(P)
        ok = (i >= 0) & (j >= 0) & (i < self.m.shape[0]) & (j < self.m.shape[1])
        out = np.zeros(len(P), bool)
        out[ok] = self.m[i[ok], j[ok]] > 0
        return out


class Mission:
    FINAL = ("DONE", "ABORTED")

    def __init__(self, cfg=MissionConfig()):
        self.c, self.pc = cfg, cfg.perception
        self.state, self.ref = "INIT", None
        self.visited = Visited(cfg.search_bounds)
        self.transitions, self.n_lost, self.reversals, self.restarts = [], 0, 0, 0
        self.circ_est, self.circ_hits, self.circ_confirmed, self.circ_meas = None, 0, False, None
        self.wps, self.wpi, self.move_dir = None, 0, np.array([1.0, 0.0])
        self.home, self.path = None, []
        self.success, self.fail_reason = False, ""
        self._reset_state_vars()

    # ---------------------------------------------------------------- plumbing
    @property
    def finished(self):
        return self.state in self.FINAL

    def _reset_state_vars(self):
        self.line_t = self.ok_t = self.no_cand_t = self.lost_t = self.no_meas_t = 0.0
        self.lost_active = False

    def _enter(self, new, t, reason=""):
        self.transitions.append((t, self.state, new, reason))
        self.state, self.t_enter = new, t
        self._reset_state_vars()
        if new == "SEARCH":
            self.wps = None
        elif new == "FOLLOW":
            self._acquire()
        elif new == "ABORT":
            self.fail_reason = reason

    def _goto(self, n, e, d, psi, vxy, vz, scale=1.0):
        """Rate-limited reference toward a goal (the Simulink 'reference position / yaw' output)."""
        c, r, dt = self.c, self.ref, self.dt
        v = np.array([n - r[0], e - r[1]]); dist = np.linalg.norm(v); step = vxy * scale * dt
        if dist > 1e-9:
            r[:2] += v if dist <= step else v / dist * step
        r[2] += np.clip(d - r[2], -vz * dt, vz * dt)
        r[3] = wrap(r[3] + np.clip(wrap(psi - r[3]), -c.yaw_rate * dt, c.yaw_rate * dt))
        lead = r[:2] - np.array(self.pose[:2]); ln = np.linalg.norm(lead)
        if ln > c.max_lead:
            r[:2] = np.array(self.pose[:2]) + lead / ln * c.max_lead

    def _hold(self):
        self._goto(self.ref[0], self.ref[1], self.ref[2], self.ref[3], 0, 0)

    def _recover_or_abort(self, reason, t):
        self.restarts += 1
        if self.restarts > self.c.max_restarts:
            self._enter("ABORT", t, reason + " (restarts exhausted)")
        else:
            self._enter("SEARCH", t, reason + " -> restart search")

    # ---------------------------------------------------------------- main step
    def step(self, img, pose, t):
        c = self.c
        if self.ref is None:
            self.ref = np.array(pose, float); self.t_prev = t; self.t_enter = t
        self.dt = max(t - self.t_prev, 1e-3); self.t_prev = t
        self.pose = pose
        pos, alt = np.array(pose[:2]), -pose[2]
        self.fr = fr = analyze(img, pose, self.pc)
        self._update_circle(fr, alt)
        if not self.path or np.linalg.norm(pos - self.path[-1]) > 0.04:
            self.path = (self.path + [pos.copy()])[-40:]
        to = c.timeouts.get(self.state)
        if to and t - self.t_enter > to:
            if self.state in ("FOLLOW", "APPROACH", "ALIGN"):
                self._recover_or_abort(self.state + " timeout", t)
            else:
                self._enter("ABORT", t, self.state + " timeout")
        getattr(self, "_s_" + self.state.lower())(fr, pose, pos, alt, t)
        return self.ref.copy(), dict(state=self.state, circle_hits=self.circ_hits, n_lost=self.n_lost)

    # ---------------------------------------------------------------- circle tracking
    def _update_circle(self, fr, alt):
        meas = fr["circle"]
        if meas is None and alt < 0.5 and self.circ_est is not None:      # low: relaxed (circle fills the view)
            near = [b for b in fr["blobs"] if not b["touches"] and
                    np.hypot(b["N"] - self.circ_est[0], b["E"] - self.circ_est[1]) < 0.2]
            meas = max(near, key=lambda b: b["area"]) if near else None
        self.circ_meas = meas
        if alt < self.c.blind_alt:                                         # frozen estimate when too close
            return
        if meas is not None:
            p = np.array([meas["N"], meas["E"]])
            if self.circ_est is not None and np.linalg.norm(p - self.circ_est) < 0.25:
                self.circ_hits += 1
                self.circ_est = 0.7 * self.circ_est + 0.3 * p
            else:
                self.circ_est, self.circ_hits, self.circ_confirmed = p, 1, False
            if self.circ_hits >= self.c.circle_confirm:
                self.circ_confirmed = True
        else:
            self.circ_hits = 0

    # ---------------------------------------------------------------- states
    def _s_init(self, fr, pose, pos, alt, t):
        self._hold()
        self.line_t = self.line_t + self.dt if fr["blue_area"] >= self.pc.min_pixels else 0.0
        if self.line_t >= self.c.init_blue_s:
            self.home = pos.copy()
            self._enter("TAKEOFF", t, "on blue square")

    def _s_takeoff(self, fr, pose, pos, alt, t):
        c = self.c
        self._goto(self.home[0], self.home[1], -c.alt, self.ref[3], 0.2, c.v_climb)
        self.ok_t = self.ok_t + self.dt if abs(alt - c.alt) < c.takeoff_tol else 0.0
        if self.ok_t >= c.takeoff_hold_s:
            self._enter("SEARCH", t, "altitude reached")

    def _build_search(self, pos):
        c = self.c
        nmin, nmax, emin, emax = c.search_bounds
        foot_w = 2 * c.alt * np.tan(np.deg2rad(self.pc.fov_deg) / 2)
        nl = max(2, int(np.ceil((emax - emin) / (c.lane_overlap * foot_w))) + 1)
        w = []
        for k, e in enumerate(np.linspace(emin, emax, nl)):
            a, b = (nmin, nmax) if k % 2 == 0 else (nmax, nmin)
            w += [(a, e), (b, e)]
        w = np.array(w); w2 = w[::-1]
        self.wps = w if np.linalg.norm(w[0] - pos) <= np.linalg.norm(w2[0] - pos) else w2
        self.wpi = 0

    def _s_search(self, fr, pose, pos, alt, t):
        c = self.c
        if self.wps is None:
            self._build_search(pos)
        wp = self.wps[self.wpi]
        if np.linalg.norm(wp - pos) < 0.15:
            self.wpi = (self.wpi + 1) % len(self.wps)
            wp = self.wps[self.wpi]
        d = wp - pos
        if np.linalg.norm(d) > 1e-6:
            self.move_dir = d / np.linalg.norm(d)
        self._goto(wp[0], wp[1], -c.alt, 0.0, c.v_search, c.v_climb)
        hit = abs(alt - c.alt) < 0.2 and any((not b["is_circle"]) and b["length"] >= c.line_min_len
                                               for b in fr["blobs"])
        self.line_t = self.line_t + self.dt if hit else 0.0
        if self.line_t >= c.search_confirm_s:
            self._enter("FOLLOW", t, "line confirmed")

    def _line_geo(self, max_d):
        """Line pixels reachable ALONG the line from the pixels under the vehicle.
        Returns world (N,2) coords and distance-along-line [m] from the vehicle, plus nearest-pixel distance."""
        mask, pose = self.fr["line_mask"], self.pose
        if not mask.any():
            return np.zeros((0, 2)), np.zeros(0), np.inf
        pc = self.pc
        scale = max(-pose[2], 0.03) / focal(pc)
        rr, cc = np.nonzero(mask)
        dc = np.hypot(rr - (pc.img_rows - 1) / 2, cc - (pc.img_cols - 1) / 2) * scale
        dmin = dc.min()
        seed = np.zeros_like(mask)
        sel = dc <= max(0.10, dmin + 2 * scale)
        seed[rr[sel], cc[sel]] = True
        gd = geodesic(mask, seed, int(max(max_d - dmin, 0) / scale) + 1)
        ok = gd[rr, cc] >= 0
        rr, cc = rr[ok], cc[ok]
        d = dmin + gd[rr, cc] * scale
        f_, r_ = pix_to_body(rr, cc, -pose[2], pc)
        N_, E_ = body_to_world(f_, r_, pose)
        return np.column_stack([N_, E_]), d, dmin

    def _acquire(self):
        """Pick the line direction at first contact; mark the part of the line behind as already followed."""
        pos = np.array(self.pose[:2])
        self.visited.clear()
        P, d, _ = self._line_geo(1.2)
        if len(P) == 0:
            return
        A = P[d.argmin()]
        near = P[np.linalg.norm(P - A, axis=1) < 0.5]
        u = self.move_dir
        if len(near) >= 5:
            q = near - near.mean(0)
            u = np.linalg.eigh(q.T @ q)[1][:, 1]
            if u @ self.move_dir < 0:
                u = -u
        self.visited.mark_points(P[(P - A) @ u < -0.05])

    def _s_follow(self, fr, pose, pos, alt, t):
        c = self.c
        if not fr["line_mask"].any():                         # ---- line lost: recovery
            if not self.lost_active:
                self.lost_active, self.n_lost = True, self.n_lost + 1
            self.lost_t += self.dt
            if self.lost_t < c.lost_hold_s:
                self._hold()
            elif self.lost_t < c.lost_giveup_s:
                self._goto(self.ref[0], self.ref[1], -c.lost_alt, self.ref[3], 0, c.v_climb)
            else:
                self._recover_or_abort("line lost", t)
            return
        self.lost_active, self.lost_t = False, 0.0
        P, d, dmin = self._line_geo(c.r_max)
        if dmin > 0.3 and len(P):                             # far from the line: fly to its nearest point first
            tgt = P[d.argmin()]
            self._goto(tgt[0], tgt[1], -c.alt, self.ref[3], c.v_follow, c.v_climb)
            return
        self.visited.mark_points(P[d <= c.visit_r], r=0.03)    # followed = near us ALONG the line (not as the crow flies)
        unv = ~self.visited.check(P)
        cand, d = P[unv], d[unv]
        keep = d <= c.r_max
        cand, d = cand[keep], d[keep]
        if len(cand) == 0:                                    # ---- nothing new ahead: end of line?
            self.no_cand_t += self.dt
            self._goto(self.ref[0], self.ref[1], -c.alt, self.ref[3], 0, c.v_climb)
            if self.no_cand_t >= c.end_hold_s:
                self._end_of_line(fr, pos, t)
            return
        self.no_cand_t = 0.0
        d0 = d[d >= c.r_min].min() if (d >= c.r_min).any() else d.max()
        sel = cand[(d >= d0) & (d <= d0 + c.band)]
        rel = wrap(np.arctan2(sel[:, 1] - pos[1], sel[:, 0] - pos[0]) - self.ref[3])
        order = np.argsort(rel); rel_s = rel[order]
        cuts = np.nonzero(np.diff(rel_s) > c.branch_gap)[0] + 1   # split branches (e.g. at a junction/corner)
        groups = np.split(order, cuts)
        g = min(groups, key=lambda ix: abs(rel[ix].mean()))       # keep the branch best aligned with our heading
        tgt = sel[g].mean(0)
        self.visited.mark_points(cand[d < d0], r=0.03)           # line between us and the target is passed over
        psi = np.arctan2(tgt[1] - pos[1], tgt[0] - pos[0]) if np.linalg.norm(tgt - pos) > 0.05 else self.ref[3]
        scale = float(np.clip(np.cos(wrap(psi - pose[3])), 0.15, 1.0))
        self._goto(tgt[0], tgt[1], -c.alt, psi, c.v_follow, c.v_climb, scale)

    def _end_of_line(self, fr, pos, t):
        c = self.c
        if self.circ_confirmed and self.circ_est is not None and np.linalg.norm(self.circ_est - pos) < c.circle_near:
            self._enter("APPROACH", t, "line end + circle confirmed")
        elif self.reversals < c.max_reversals:                # wrong end: go back along the line
            self.reversals += 1
            back = self.path[0] - pos
            u = back / np.linalg.norm(back) if np.linalg.norm(back) > 1e-6 else -self.move_dir
            self.visited.clear()
            P, _, _ = self._line_geo(1.2)
            self.visited.mark_points(P[(P - pos) @ u < -0.05])
            self.transitions.append((t, "FOLLOW", "FOLLOW", "end without circle -> reverse"))
            self.no_cand_t = 0.0
        else:
            self._recover_or_abort("end without circle", t)

    def _s_approach(self, fr, pose, pos, alt, t):
        c = self.c
        if self.circ_est is None:
            self._recover_or_abort("circle lost", t); return
        self.no_meas_t = 0.0 if self.circ_meas is not None else self.no_meas_t + self.dt
        if self.no_meas_t > 3.0:
            self._recover_or_abort("circle lost", t); return
        e = self.circ_est
        self._goto(e[0], e[1], -c.alt, self.ref[3], c.v_approach, c.v_climb)
        if np.linalg.norm(e - pos) < 0.15 and self.circ_hits >= c.circle_confirm:
            self._enter("ALIGN", t, "above circle")

    def _s_align(self, fr, pose, pos, alt, t):
        c, m = self.c, self.circ_meas
        if m is None or self.circ_est is None:
            self.no_meas_t += self.dt; self.ok_t = 0.0; self._hold()
            if self.no_meas_t > 3.0:
                self._recover_or_abort("circle lost in align", t)
            return
        self.no_meas_t = 0.0
        e = self.circ_est
        self._goto(e[0], e[1], -c.align_alt, self.ref[3], 0.25, c.v_climb)
        err = np.hypot(m["fwd"], m["right"])                  # camera is at the vehicle centre
        self.ok_t = self.ok_t + self.dt if (err < c.align_tol and abs(alt - c.align_alt) < 0.05) else 0.0
        if self.ok_t >= c.align_hold_s:
            self._enter("LAND", t, "aligned")

    def _s_land(self, fr, pose, pos, alt, t):
        c, m = self.c, self.circ_meas
        if alt <= c.landed_alt:
            self.success = True
            self._enter("DONE", t, "touchdown"); return
        e = self.circ_est if self.circ_est is not None else pos
        err = np.hypot(m["fwd"], m["right"]) if m is not None else None
        blind = alt < c.blind_alt
        if blind or (err is not None and err < c.land_err_max):
            d_goal = 0.1                                        # below ground -> keep descending
            self.no_meas_t = 0.0
        else:
            d_goal = self.ref[2]                                # pause descent until re-centred
            self.no_meas_t = self.no_meas_t + self.dt if err is None else 0.0
            if self.no_meas_t > 4.0:
                self._recover_or_abort("circle lost during landing", t); return
        self._goto(e[0], e[1], d_goal, self.ref[3], 0.25, c.v_descend)

    def _s_abort(self, fr, pose, pos, alt, t):
        self._goto(self.ref[0], self.ref[1], 0.1, self.ref[3], 0, self.c.v_descend)
        if alt <= self.c.landed_alt:
            self._enter("ABORTED", t, "landed in place")

    def _s_done(self, *a): self._hold()
    def _s_aborted(self, *a): self._hold()