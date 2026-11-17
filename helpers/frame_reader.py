"""Reading video frames and finding ball candidates in them.

A candidate is a small whitish blob on the pitch. No player is erased from the image (a goalkeeper may hold the ball,
and boots can be white): each blob is kept and described by 13 features, including whether it lies inside a player's
body hull and how close it is to a player's foot, so the classifier can learn to distrust boots.
Filter thresholds: brightness > 115, saturation < 110, area 4-220 px, elongation <= 3.5, fill >= 0.30.
"""
import cv2
import numpy as np

from .camera import Camera
from .settings import Settings


class FrameReader:
    MAX_CANDIDATES = 150
    MARGIN_M = 4.0            # pitch polygon is grown by this margin (metres)
    V_MIN, S_MAX = 115, 110   # whitish: bright, low saturation
    AREA_MIN, AREA_MAX, SIDE_MAX = 4, 220, 26
    AR_MAX, FILL_MIN = 3.5, 0.30

    def __init__(self, match):
        self.m = match
        self.cap = cv2.VideoCapture(match.video_path)

    def read_image(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, img = self.cap.read()
        return img if ok else None

    def read_window(self, start, count, step=1, keep_imgs=False):
        """Read `count` consecutive frames from `start`; candidates are computed on every `step`-th frame
        that has an ATD ball (the others cannot be used and are skipped to save time).
        Returns (frames, candidates per frame, images)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        frames, cands, imgs = [], [], []
        for k in range(count):
            ok, img = self.cap.read()
            if not ok:
                break
            if k % step == 0:
                f = start + k
                frames.append(f)
                cands.append(self.find_candidates(img, f) if self.m.has_ball[f] else [])
                if keep_imgs:
                    imgs.append(img)
        return frames, cands, imgs

    def read_candidates(self, start, count):
        """Candidates for `count` consecutive frames from `start`, whether or not the ATD has a ball there."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        out = []
        for k in range(count):
            ok, img = self.cap.read()
            if not ok:
                break
            out.append(self.find_candidates(img, start + k))
        return out

    def find_candidates(self, img, f):
        """List of dicts: x, y (pixels), px, py (pitch metres, ground-plane unprojection), slot (nearest
        player slot or None), fv (13 features), area."""
        m = self.m
        L, W = Settings.PITCH_L, Settings.PITCH_W
        h, w = img.shape[:2]
        Hn = m.H[f]
        mg = self.MARGIN_M
        corners = np.array([Camera.to_pixel(Hn, u, v) for u, v in ((-mg / L, -mg / W), (1 + mg / L, -mg / W),
                                                                   (1 + mg / L, 1 + mg / W), (-mg / L, 1 + mg / W))], np.int32)
        pitch = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(pitch, corners, 255)
        linem = np.zeros((h, w), np.uint8)
        for pl in Camera.pitch_lines():
            cv2.polylines(linem, [np.array([Camera.to_pixel(Hn, x / L, y / W) for x, y in pl], np.int32)], False, 255, 9)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        grass = (hsv[..., 0] >= 28) & (hsv[..., 0] <= 95) & (hsv[..., 1] > 45)
        mask = ((hsv[..., 2] > self.V_MIN) & (hsv[..., 1] < self.S_MAX) & (~grass) & (pitch > 0) & (linem == 0)).astype(np.uint8)

        cam = Camera(Hn)
        hulls, feet = [], []
        for i in range(46):
            if not np.isfinite(m.X[f, i]):
                continue
            c = [(m.X[f, i] + dx, m.Y[f, i] + dy, z) for dx in (-0.45, 0.45) for dy in (-0.45, 0.45) for z in (0, 1.9)]
            hulls.append((i, cv2.convexHull(cam.project(c).astype(np.int32))))
            feet.append((i, Camera.to_pixel(Hn, m.X[f, i] / L, m.Y[f, i] / W)))

        n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
        V = hsv[..., 2].astype(np.float32)
        S = hsv[..., 1].astype(np.float32)
        out = []
        for k in range(1, n):
            x0, y0, bw, bh, area = st[k]
            if area < self.AREA_MIN or area > self.AREA_MAX or bw > self.SIDE_MAX or bh > self.SIDE_MAX:
                continue
            ar = max(bw, bh) / max(1, min(bw, bh))
            fill = area / float(bw * bh)
            if ar > self.AR_MAX or fill < self.FILL_MIN:
                continue
            cxk, cyk = cen[k]
            sub = (slice(max(0, y0 - 10), y0 + bh + 10), slice(max(0, x0 - 10), x0 + bw + 10))
            comp = lab[sub] == k
            ring = ~cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            ring_green = grass[sub][ring].mean() if ring.any() else 0
            ring_v = V[sub][ring].mean() if ring.any() else 0
            vin, sin = V[sub][comp], S[sub][comp]
            slot = None
            for i, hu in hulls:
                if cv2.pointPolygonTest(hu, (float(cxk), float(cyk)), False) >= 0:
                    slot = i
                    break
            d_foot = min([np.hypot(cxk - fx, cyk - fy) for i, (fx, fy) in feet]) if feet else 999
            if slot is None and feet:
                j = int(np.argmin([np.hypot(cxk - fx, cyk - fy) for i, (fx, fy) in feet]))
                if np.hypot(cxk - feet[j][1][0], cyk - feet[j][1][1]) < 35:
                    slot = feet[j][0]
            px, py = Camera.to_pitch(Hn, cxk, cyk)
            fv = [area, bw, bh, fill, ar, vin.mean(), sin.mean(), vin.std(), ring_green, vin.mean() - ring_v,
                  float(slot is not None), d_foot, cyk / 1080.0]
            out.append(dict(x=cxk, y=cyk, px=px, py=py, slot=slot, fv=fv, area=int(area)))
        return out[:self.MAX_CANDIDATES]
