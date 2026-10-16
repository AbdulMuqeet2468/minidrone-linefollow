"""Deterministic tests of perception.segment on simulator frames.  Run: python test_perception.py [--live]"""
import sys
import numpy as np
import drone_sim as ds
from perception import segment, mask_stats


def frame(cam, arena, n, e, alt, yaw=0.0):
    return cam.render(arena, (n, e, -alt, yaw))


def run_tests():
    for sigma, bright in [(4, 1.0), (15, 0.6), (15, 1.3)]:
        cfg = ds.Config(noise_sigma=sigma, brightness=bright)
        arena, cam = ds.Arena(cfg, 0), ds.Camera(cfg)
        p0, p1 = arena.pts[0], arena.pts[1]
        mid = (p0 + p1) / 2
        yaw = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        cases = {  # name: (frame, expect red, expect blue)
            "square": (frame(cam, arena, *arena.square, 0.15), False, True),
            "line":   (frame(cam, arena, *mid, 0.15, yaw), True, False),
            "circle": (frame(cam, arena, *arena.circ, 0.15), True, False),
        }
        print(f"noise={sigma} brightness={bright}")
        for name, (img, er, eb) in cases.items():
            r, b = mask_stats(segment(img)[0]), mask_stats(segment(img)[1])
            ok = (r["detected"] == er) and (b["detected"] == eb)
            print(f"  {name:6s} red={r['area']:4d} blue={b['area']:4d}  {'PASS' if ok else 'FAIL'}")
            assert ok


def live():
    import matplotlib.pyplot as plt

    class V(ds.Viewer):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.figm = plt.figure("Masks (red | blue)", figsize=(5, 6))
            self.axm = self.figm.add_subplot(111)
            self.imm = self.axm.imshow(np.zeros((cfg.img_rows, 2 * cfg.img_cols + 2, 3)), interpolation="nearest")
            self.axm.set_xticks([]); self.axm.set_yticks([])

        def tick(self):
            super().tick()
            red, blue = segment(self.last_img)
            sep = np.full((red.shape[0], 2), 128, np.uint8)
            m = np.hstack([red, sep, blue])
            self.imm.set_data(np.dstack([m] * 3))
            rs, bs = mask_stats(red), mask_stats(blue)
            self.axm.set_title(f"red {rs['area']} px | blue {bs['area']} px", fontsize=9)
            self.figm.canvas.draw_idle()

    V(ds.Config()).run()


if __name__ == "__main__":
    live() if "--live" in sys.argv else run_tests()