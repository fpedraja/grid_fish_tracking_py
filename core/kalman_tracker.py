"""Kalman tracker with amplitude-aware association for electric fish.

State vector: [px, py, vx, vy, f]
  px, py  – position in cm
  vx, vy  – velocity in cm/s  (assumed small over 60-s steps)
  f       – discharge frequency in Hz

Improvements over the MATLAB version:
  - Class-based (no 'persistent' variables / global state)
  - scipy.optimize.linear_sum_assignment (always Hungarian, no greedy fallback)
  - Amplitude EMA update actually implemented
  - Joseph-form covariance update for numerical stability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# Track dataclass
# ---------------------------------------------------------------------------

@dataclass
class Track:
    id:       int
    x:        np.ndarray          # state (5,): [px, py, vx, vy, f]
    P:        np.ndarray          # covariance (5, 5)
    miss:     int = 0             # consecutive missed frames
    age:      int = 0             # total frames alive
    last_det: Optional[np.ndarray] = None   # last accepted measurement (3,)
    sig:      Optional[np.ndarray] = None   # amplitude fingerprint (C,)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class KalmanFishTracker:
    """Kalman tracker with amplitude-aware data association.

    Parameters
    ----------
    meas_sig_pos  : measurement noise std-dev for position (cm)
    meas_sig_f    : measurement noise std-dev for frequency (Hz)
    wander_1min   : expected position wander RMS over 1 minute (cm)
    freq_drift_1m : expected frequency drift RMS over 1 minute (Hz)
    xlim, ylim    : tank boundaries (cm) – used to clamp predictions
    max_miss      : prune tracks that miss this many consecutive frames
    sig_gate      : cosine-distance threshold; reject if cosDist > sig_gate
    sigma_sig     : cosine-distance scale for cost term
    w_sig         : weight of amplitude term relative to Mahalanobis^2
    beta_sig      : EMA rate for fingerprint update (0 = freeze, 1 = replace)
    hard_gate_pos : hard position gate (cm) – skip costly Mahalanobis if exceeded
    hard_gate_f   : hard frequency gate (Hz)
    """

    def __init__(self,
                 meas_sig_pos:  float = 6.0,
                 meas_sig_f:    float = 0.15,
                 wander_1min:   float = 40.0,
                 freq_drift_1m: float = 0.3,
                 xlim: tuple = (0, 80),
                 ylim: tuple = (0, 120),
                 max_miss:      int   = 10,
                 sig_gate:      float = 0.35,
                 sigma_sig:     float = 0.20,
                 w_sig:         float = 6.0,
                 beta_sig:      float = 0.2,
                 hard_gate_pos: float = 60.0,
                 hard_gate_f:   float = 2.0):

        self.meas_sig_pos  = meas_sig_pos
        self.meas_sig_f    = meas_sig_f
        self.wander        = wander_1min
        self.freq_drift    = freq_drift_1m
        self.xlim          = xlim
        self.ylim          = ylim
        self.max_miss      = max_miss
        self.sig_gate      = sig_gate
        self.sigma_sig     = sigma_sig
        self.w_sig         = w_sig
        self.beta_sig      = beta_sig
        self.hard_gate_pos = hard_gate_pos
        self.hard_gate_f   = hard_gate_f

        try:
            self._chi2_gate = float(chi2.ppf(0.997, df=3))
        except Exception:
            self._chi2_gate = 14.16  # fallback if scipy.stats unavailable

        self.tracks:  list[Track] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self,
               centroids: np.ndarray,
               freqs:     np.ndarray,
               amp_sigs:  list,
               dt:        float = 60.0) -> list[Track]:
        """Predict, associate, and update tracks with new detections.

        Args:
            centroids: (N, 2) array of [x, y] in cm
            freqs:     (N,) discharge frequencies in Hz
            amp_sigs:  list of N fingerprint arrays (C,) or None
            dt:        time elapsed since the previous call (seconds)

        Returns:
            Current list of all live Track objects.
        """
        N = len(centroids)
        F = self._F(dt)
        Q = self._Q(dt)
        H = self._H()
        R = self._R()

        self.last_log: list[str] = []   # diagnostic lines, readable by caller

        # ---- Predict ----
        for t in self.tracks:
            t.x = F @ t.x
            t.P = F @ t.P @ F.T + Q
            t.x = self._clamp(t.x)
            if t.miss > 0:
                t.x[2:4] *= 0.9          # damp velocity when track is coasting

        # ---- Build cost matrix ----
        nT = len(self.tracks)
        cost     = np.full((nT, N), np.inf)
        # Keep per-(track,det) rejection reason for logging
        _reasons = [[None]*N for _ in range(nT)]

        for i, t in enumerate(self.tracks):
            for j in range(N):
                dp = np.hypot(centroids[j, 0] - t.x[0], centroids[j, 1] - t.x[1])
                df = abs(freqs[j] - t.x[4])
                if dp > self.hard_gate_pos or df > self.hard_gate_f:
                    _reasons[i][j] = f"hard gate (dp={dp:.1f}cm df={df:.2f}Hz)"
                    continue

                z = np.array([centroids[j, 0], centroids[j, 1], freqs[j]])
                S = H @ t.P @ H.T + R
                v = z - H @ t.x
                try:
                    d2 = float(v @ np.linalg.solve(S, v))
                except np.linalg.LinAlgError:
                    _reasons[i][j] = "singular S"
                    continue
                if d2 >= self._chi2_gate:
                    _reasons[i][j] = f"Mahal gate (d²={d2:.1f})"
                    continue

                add_cost = self._amplitude_cost(t.sig, amp_sigs[j])
                if add_cost is None:
                    _reasons[i][j] = "amplitude gate"
                    continue

                cost[i, j] = d2 + add_cost
                _reasons[i][j] = f"OK cost={cost[i,j]:.2f}"

        # ---- Hungarian assignment ----
        pairs: list[tuple] = []
        assigned_tracks: set = set()
        assigned_dets:   set = set()

        if nT > 0 and N > 0 and np.any(np.isfinite(cost)):
            row_ind, col_ind = linear_sum_assignment(
                np.where(np.isfinite(cost), cost, 1e9)
            )
            for r, c in zip(row_ind, col_ind):
                if np.isfinite(cost[r, c]):
                    pairs.append((r, c))
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

        # ---- Kalman update for matched tracks ----
        for i, j in pairs:
            t  = self.tracks[i]
            z  = np.array([centroids[j, 0], centroids[j, 1], freqs[j]])
            S  = H @ t.P @ H.T + R
            K  = np.linalg.solve(S, H @ t.P).T        # K = P H' S^{-1}
            v  = z - H @ t.x
            t.x = t.x + K @ v
            I_KH = np.eye(5) - K @ H
            t.P  = I_KH @ t.P @ I_KH.T + K @ R @ K.T  # Joseph form (numerically stable)
            t.x  = self._clamp(t.x)
            t.miss    = 0
            t.age    += 1
            t.last_det = z.copy()
            self.last_log.append(
                f"  MATCHED  det({centroids[j,0]:.0f},{centroids[j,1]:.0f} "
                f"{freqs[j]:.1f}Hz) → track#{t.id} "
                f"(cost={cost[i,j]:.2f})"
            )
            # EMA update of amplitude fingerprint
            sig_j = amp_sigs[j]
            if sig_j is not None:
                if t.sig is None:
                    t.sig = sig_j.copy()
                else:
                    t.sig = (1.0 - self.beta_sig) * t.sig + self.beta_sig * sig_j
                    nrm = np.linalg.norm(t.sig)
                    if nrm > 0:
                        t.sig /= nrm

        # ---- Handle missed tracks ----
        for i in range(nT):
            if i not in assigned_tracks:
                t = self.tracks[i]
                t.miss += 1
                t.age  += 1
                # Log rejection reasons for every unmatched detection
                if N > 0:
                    for j in range(N):
                        reason = _reasons[i][j] or "unknown"
                        self.last_log.append(
                            f"  MISSED   track#{t.id} "
                            f"({t.x[0]:.0f},{t.x[1]:.0f} {t.x[4]:.1f}Hz) "
                            f"vs det{j} ({centroids[j,0]:.0f},{centroids[j,1]:.0f} "
                            f"{freqs[j]:.1f}Hz): {reason}"
                        )
                else:
                    self.last_log.append(
                        f"  COASTING track#{t.id} ({t.x[0]:.0f},{t.x[1]:.0f} "
                        f"{t.x[4]:.1f}Hz) miss={t.miss}"
                    )

        # ---- Initialise new tracks ----
        for j in range(N):
            if j not in assigned_dets:
                x0 = np.array([centroids[j, 0], centroids[j, 1],
                                0.0, 0.0, freqs[j]], dtype=float)
                p_pos = (self.meas_sig_pos * 2.0) ** 2
                p_vel = self.meas_sig_pos ** 2
                p_f   = max(self.meas_sig_f, self.freq_drift) ** 2
                P0 = np.diag([p_pos, p_pos, p_vel, p_vel, p_f])
                s0 = amp_sigs[j].copy() if amp_sigs[j] is not None else None
                self.tracks.append(Track(
                    id=self._next_id, x=x0, P=P0, miss=0, age=1,
                    last_det=np.array([centroids[j, 0], centroids[j, 1], freqs[j]]),
                    sig=s0,
                ))
                self.last_log.append(
                    f"  NEW      track#{self._next_id} "
                    f"({centroids[j,0]:.0f},{centroids[j,1]:.0f} {freqs[j]:.1f}Hz)"
                )
                self._next_id += 1

        # ---- Prune stale tracks ----
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]

        return self.tracks

    def reset(self) -> None:
        """Clear all tracks and reset ID counter."""
        self.tracks   = []
        self._next_id = 1

    @property
    def matched_tracks(self) -> list[Track]:
        """Tracks matched in the most recent update call (miss == 0)."""
        return [t for t in self.tracks if t.miss == 0]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _F(self, dt: float) -> np.ndarray:
        return np.array([
            [1, 0, dt, 0,  0],
            [0, 1, 0,  dt, 0],
            [0, 0, 1,  0,  0],
            [0, 0, 0,  1,  0],
            [0, 0, 0,  0,  1],
        ], dtype=float)

    def _H(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ], dtype=float)

    def _Q(self, dt: float) -> np.ndarray:
        q_acc = 3.0 * self.wander**2 / dt**3
        Qcv   = np.array([[dt**3 / 3, dt**2 / 2],
                           [dt**2 / 2, dt       ]]) * q_acc
        return block_diag(Qcv, Qcv, np.array([[self.freq_drift**2]]))

    def _R(self) -> np.ndarray:
        return np.diag([self.meas_sig_pos**2,
                        self.meas_sig_pos**2,
                        self.meas_sig_f**2])

    def _clamp(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        if x[0] < self.xlim[0]: x[0] = self.xlim[0]; x[2] = 0.0
        if x[0] > self.xlim[1]: x[0] = self.xlim[1]; x[2] = 0.0
        if x[1] < self.ylim[0]: x[1] = self.ylim[0]; x[3] = 0.0
        if x[1] > self.ylim[1]: x[1] = self.ylim[1]; x[3] = 0.0
        return x

    def _amplitude_cost(self,
                        sig_track: Optional[np.ndarray],
                        sig_det:   Optional[np.ndarray]) -> Optional[float]:
        """Compute amplitude penalty term.  Returns None if gate is exceeded."""
        if sig_track is None or sig_det is None:
            return 0.0
        n1 = np.linalg.norm(sig_track)
        n2 = np.linalg.norm(sig_det)
        if n1 == 0 or n2 == 0:
            return 0.0
        cos_sim  = np.dot(sig_track, sig_det) / (n1 * n2)
        cos_dist = max(0.0, 1.0 - cos_sim)
        if cos_dist > self.sig_gate:
            return None                         # gate exceeded – reject pair
        return self.w_sig * (cos_dist / self.sigma_sig) ** 2
