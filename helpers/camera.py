"""Camera model recovered from the per-frame homography, conversions pitch <-> pixel, and the pitch markings.

The homography maps the pitch plane (z = 0) to the image. Assuming square pixels, no lens distortion and the principal
point at the image centre (960, 540), the focal length and the camera pose can be recovered from it, which lets us project
a point *above* the ground (x, y, z) into the image. That is what makes the "vertical line above the ATD point" possible:
the ATD ball position is the ball's ground position, so the real (airborne) ball lies somewhere along that vertical line.
"""
import numpy as np

from .settings import Settings


class Camera:
    CX0, CY0 = 960.0, 540.0
    _lines = None

    def __init__(self, Hn):
        """Normalised homography -> K, R, t, up. `up` is +1/-1 so that z points towards the camera."""
        Hm = Hn @ np.diag([1 / Settings.PITCH_L, 1 / Settings.PITCH_W, 1.0])
        h1, h2, h3 = Hm[:, 0], Hm[:, 1], Hm[:, 2]
        a1, a2 = self._centred(h1), self._centred(h2)
        f = np.sqrt(abs(-(a1 @ a2) / (h1[2] * h2[2])))
        K = np.array([[f, 0, self.CX0], [0, f, self.CY0], [0, 0, 1]])
        Ki = np.linalg.inv(K)
        s = np.linalg.norm(Ki @ h1)
        r1, r2, t = Ki @ h1 / s, Ki @ h2 / s, Ki @ h3 / s
        if t[2] < 0:
            r1, r2, t = -r1, -r2, -t
        R = np.stack([r1, r2, np.cross(r1, r2)], 1)
        C = -R.T @ t
        self.K, self.R, self.t = K, R, t
        self.up = 1.0 if C[2] > 0 else -1.0

    @classmethod
    def _centred(cls, h):
        return np.array([h[0] - cls.CX0 * h[2], h[1] - cls.CY0 * h[2]])

    def project(self, points):
        """(n, 3) points (x, y metres; z metres above ground) -> image pixels (n, 2)."""
        P = np.array(points, float).copy()
        P[:, 2] *= self.up
        v = (self.K @ (self.R @ P.T + self.t[:, None])).T
        return v[:, :2] / v[:, 2:3]

    def ground_from_pixel(self, u, v, z=0.0):
        """Image pixel -> pitch position in metres of the point at height z above the ground."""
        M = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.R[:, 2] * z * self.up + self.t])
        g = np.linalg.solve(M, np.array([u, v, 1.0]))
        return g[0] / g[2], g[1] / g[2]

    @staticmethod
    def to_pixel(Hn, x, y):
        """Normalised pitch coordinates (0-1) -> image pixels, with the plain homography."""
        d = Hn[2, 0] * x + Hn[2, 1] * y + Hn[2, 2]
        return (Hn[0, 0] * x + Hn[0, 1] * y + Hn[0, 2]) / d, (Hn[1, 0] * x + Hn[1, 1] * y + Hn[1, 2]) / d

    @staticmethod
    def to_pitch(Hn, px, py):
        """Image pixels -> pitch metres, with the plain homography."""
        v = np.linalg.inv(Hn) @ np.array([px, py, 1.0])
        return v[0] / v[2] * Settings.PITCH_L, v[1] / v[2] * Settings.PITCH_W

    @classmethod
    def pitch_lines(cls):
        """Pitch markings as polylines in metres (used to mask white lines out of the ball candidates)."""
        if cls._lines is None:
            L, W = Settings.PITCH_L, Settings.PITCH_W
            lines = [[(0, 0), (L, 0), (L, W), (0, W), (0, 0)], [(L / 2, 0), (L / 2, W)]]
            for x0, sgn, dep, wid in ((0, 1, 16.5, 40.32), (L, -1, 16.5, 40.32), (0, 1, 5.5, 18.32), (L, -1, 5.5, 18.32)):
                y0 = W / 2 - wid / 2
                lines.append([(x0, y0), (x0 + sgn * dep, y0), (x0 + sgn * dep, y0 + wid), (x0, y0 + wid)])
            theta = np.linspace(0, 2 * np.pi, 90)
            lines.append([(L / 2 + 9.15 * np.cos(a), W / 2 + 9.15 * np.sin(a)) for a in theta])
            for x0, sgn in ((11, 1), (L - 11, -1)):
                lines.append([(x0 + 9.15 * sgn * np.cos(a), W / 2 + 9.15 * np.sin(a)) for a in np.linspace(-0.93, 0.93, 40)])
            cls._lines = lines
        return cls._lines
