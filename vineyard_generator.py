"""vineyard_generator.py  --  TopVine vine stand generator for MuJoCo.

Generates bare-wood vine stands (dormant season: no leaves, no grapes)
following the TopVine structural model (Louarn et al. 2008):
  * Genotype -> generate_rameau_moyen -> shoot topology
  * gen_shoot_param -> spur positions + shoot geometry
  * Topiary_2023 -> 3D shoot path with curvature

Vine geometry is randomised each call via the Thomas cluster process,
providing structural diversity across training episodes.

Coordinate convention (TopVine):
  * Spur positions in CM, converted to M by x0.01
  * Path: Xii = cos(courbure)*Lin*cos(azimuth) + Xi
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# VSP-2W scene geometry  (metres)
# ---------------------------------------------------------------------------
CORDON_Z     = 0.60
WIRE1_Z      = 1.20
WIRE2_Z      = 1.80
CORDON_Y     = 0.00
ROW_X_MIN    = -2.60
ROW_X_MAX    =  2.60

# ---------------------------------------------------------------------------
# Leaf MuJoCo physics
# ---------------------------------------------------------------------------
LEAF_PETIOLE_VIS = 0.05  # visual petiole in XML
STIFF_STEM = 0.001;  DAMP_STEM = 0.0005
STIFF_BODY = 0.010;  DAMP_BODY = 0.005
STIFF_RZ   = 0.005;  DAMP_RZ   = 0.003

# ---------------------------------------------------------------------------
# Grape geometry
# ---------------------------------------------------------------------------
GRAPE_STEM_LEN  = 0.03
GRAPE_Z_DROP    = 0.124
STIFF_GRAPE   = 0.005;  DAMP_GRAPE   = 0.008
STIFF_GRAPE_Z = 0.004;  DAMP_GRAPE_Z = 0.006


# ═══════════════════════════════════════════════════════════════════════════
# TopVine Genotype  (from genodata.py / generate_rameau_moyen.py)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Genotype:
    name: str = "Grenache"
    NFI_mean: float = 24.38;  NFI_sd: float = 2.33
    SF_max_mean: float = 279.0;  SF_max_sd: float = 58.5
    IN_max_mean: float = 11.71;  IN_max_sd: float = 0.91
    max_normalized_rank_SF: float = 0.34
    intercept_0_SF: float = 0.26;  intercept_1_SF: float = 0.207
    max_normalized_rank_IN: float = 0.46
    intercept_0_IN: float = 0.10;  intercept_1_IN: float = 0.46
    slope_NFII_SFII: float = 87.63;  slope_sd_NFII_SFII: float = 2.57
    size_r_binorm: float = 1.68;  mu_r_binorm: float = 1.27

GENOTYPES = {
    "Grenache": Genotype(),
    "Carignan": Genotype(
        name="Carignan", NFI_mean=21.75, NFI_sd=2.95,
        SF_max_mean=266, SF_max_sd=45.2, IN_max_mean=8.14, IN_max_sd=0.65,
        max_normalized_rank_SF=0.34, intercept_0_SF=0.31, intercept_1_SF=0.126,
        max_normalized_rank_IN=0.35, intercept_0_IN=0.0, intercept_1_IN=0.44,
        slope_NFII_SFII=56.34, slope_sd_NFII_SFII=5.96,
        size_r_binorm=100.0, mu_r_binorm=0.14),
    "Chasselas": Genotype(
        name="Chasselas", NFI_mean=23.56, NFI_sd=4.13,
        SF_max_mean=141, SF_max_sd=39.2, IN_max_mean=7.09, IN_max_sd=0.46,
        max_normalized_rank_SF=0.26, intercept_0_SF=0.22, intercept_1_SF=0.179,
        max_normalized_rank_IN=0.25, intercept_0_IN=0.0, intercept_1_IN=0.43,
        slope_NFII_SFII=37.39, slope_sd_NFII_SFII=1.02,
        size_r_binorm=6.83, mu_r_binorm=1.59),
    "Marselan": Genotype(
        name="Marselan", NFI_mean=24.63, NFI_sd=2.45,
        SF_max_mean=155, SF_max_sd=26.7, IN_max_mean=7.41, IN_max_sd=0.34,
        max_normalized_rank_SF=0.28, intercept_0_SF=0.14, intercept_1_SF=0.291,
        max_normalized_rank_IN=0.26, intercept_0_IN=0.0, intercept_1_IN=0.35,
        slope_NFII_SFII=45.26, slope_sd_NFII_SFII=0.82,
        size_r_binorm=2.96, mu_r_binorm=1.63),
    "Mourvèdre": Genotype(
        name="Mourvèdre", NFI_mean=16.94, NFI_sd=1.90,
        SF_max_mean=136, SF_max_sd=27.4, IN_max_mean=6.43, IN_max_sd=0.62,
        max_normalized_rank_SF=0.39, intercept_0_SF=0.16, intercept_1_SF=0.247,
        max_normalized_rank_IN=0.38, intercept_0_IN=0.0, intercept_1_IN=0.53,
        slope_NFII_SFII=44.05, slope_sd_NFII_SFII=2.15,
        size_r_binorm=4.21, mu_r_binorm=0.73),
    "Petit Verdot": Genotype(
        name="Petit Verdot", NFI_mean=20.50, NFI_sd=3.05,
        SF_max_mean=117, SF_max_sd=15.7, IN_max_mean=5.53, IN_max_sd=0.65,
        max_normalized_rank_SF=0.33, intercept_0_SF=0.23, intercept_1_SF=0.213,
        max_normalized_rank_IN=0.32, intercept_0_IN=0.1, intercept_1_IN=0.48,
        slope_NFII_SFII=27.82, slope_sd_NFII_SFII=1.73,
        size_r_binorm=3.20, mu_r_binorm=0.66),
    "Vermentino": Genotype(
        name="Vermentino", NFI_mean=24.38, NFI_sd=2.33,
        SF_max_mean=279, SF_max_sd=58.5, IN_max_mean=11.71, IN_max_sd=0.91,
        max_normalized_rank_SF=0.34, intercept_0_SF=0.26, intercept_1_SF=0.207,
        max_normalized_rank_IN=0.46, intercept_0_IN=0.1, intercept_1_IN=0.46,
        slope_NFII_SFII=87.63, slope_sd_NFII_SFII=2.57,
        size_r_binorm=1.68, mu_r_binorm=1.27),
}

# ═══════════════════════════════════════════════════════════════════════════
# Measured Grenache parameters  (from CSV data files)
# ═══════════════════════════════════════════════════════════════════════════

# Allometry  (allo_Grenache.csv):  L_mm = a·N + b
ALLO_I  = (65.0, -243.8)     # primary petiole
ALLO_II = (36.8, -52.4)      # secondary petiole

# Leaf angle distributions  (Law-leaf-2W-Grenache.csv)
#   [label, type(1=gauss), mean_deg, sd_deg]
LAWF = [
    ["elvS", 1, 33.265, 19.68],
    ["elvN", 1, 33.265, 19.68],
    ["aziS", 1, 181.58, 91.23],
    ["aziN", 1, -10.62, 56.28],
]

# Shoot parameter distributions  (2W_VSP_GRE_ramd.csv)
# Spur positions along cordon (cm): [(x_mu, x_sd), (y_mu, y_sd), (z_mu, z_sd)]
SPURS0_CM = [
    [(42.97, 9.16),  (5.11, 13.12), (59.14, 6.26)],   # spur 1
    [(29.40, 12.90), (6.33, 14.50), (59.14, 6.26)],   # spur 2
    [(13.40, 11.34), (4.90, 12.80), (59.14, 6.26)],   # spur 3
    [(-11.81, 10.33),(4.31, 14.50), (59.14, 6.26)],   # spur 4
    [(-29.94, 11.04),(4.11, 12.52), (59.14, 6.26)],   # spur 5
    [(-46.30, 11.05),(2.23, 11.90), (59.14, 6.26)],   # spur 6
]
DSPURS_CM = [(-0.05, 1.96), (-0.16, 1.68), (-1.67, 1.18)]

# Azimuth frequency bins  + multivariate means [alpha°, phi°, Ls, MX]
F_AZI = [0.296, 0.107, 0.200, 0.397]
SHOOT_STATS = [
    # (mean_4d, covariance_4x4)  for each azimuth range
    # Range 340-20  (front-facing)
    (np.array([59.98, -19.06, 1.078, 0.509]),
     np.array([[236.50, -152.21, -1.12, -0.14],
               [-152.21, 830.87, -5.68, -0.78],
               [-1.12, -5.68, 0.111, 0.013],
               [-0.14, -0.78, 0.013, 0.017]])),
    # Range 20-160
    (np.array([64.61, -7.61, 0.848, 0.525]),
     np.array([[628.52, -623.20, 0.58, 1.41],
               [-623.20, 1472.43, -5.51, -1.00],
               [0.58, -5.51, 0.069, 0.011],
               [1.41, -1.00, 0.011, 0.019]])),
    # Range 160-200
    (np.array([61.72, -17.02, 1.046, 0.552]),
     np.array([[373.25, -395.13, -1.97, 1.13],
               [-395.13, 933.17, -1.27, -2.09],
               [-1.97, -1.27, 0.085, -0.002],
               [1.13, -2.09, -0.002, 0.024]])),
    # Range 200-340
    (np.array([74.06, -28.26, 1.030, 0.618]),
     np.array([[97.70, -96.11, -0.021, 0.246],
               [-96.11, 763.06, -3.12, -0.993],
               [-0.021, -3.12, 0.063, 0.003],
               [0.246, -0.993, 0.003, 0.031]])),
]


# ═══════════════════════════════════════════════════════════════════════════
# Thomas cluster process  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ThomasParams:
    lambda_p: float; mu_c: float
    sigma_x: float;  sigma_z: float
    z_mu: float;     z_std: float

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "ThomasParams":
        return cls(
            lambda_p=rng.uniform(1.5, 3.0), mu_c=rng.uniform(1.5, 3.5),
            sigma_x=rng.uniform(0.06, 0.18), sigma_z=rng.uniform(0.04, 0.12),
            z_mu=rng.uniform(0.15, 0.40),    z_std=rng.uniform(0.04, 0.10))


# ═══════════════════════════════════════════════════════════════════════════
# TopVine core functions  (faithful port)
# ═══════════════════════════════════════════════════════════════════════════

def _normalised_profile(i0: float, i1: float, mx: float, n: int) -> np.ndarray:
    """Piecewise-linear normalised profile  (TopVine get_normalized_value)."""
    nr = np.arange(1, n + 1) / n
    out = np.empty(n)
    for k in range(n):
        if nr[k] < mx:
            out[k] = (1.0 - i0) / mx * nr[k] + i0
        else:
            out[k] = (i1 - 1.0) / (1.0 - mx) * (nr[k] - mx) + 1.0
    return out


def _generate_rameau_moyen(g: Genotype, rng: np.random.Generator):
    """Generate topology from genotype  (TopVine generate_rammoy_topvine).

    Returns: sf_I (cm²), sf_II_mean (cm²), in_profile (cm), n_sec
    """
    nfi = max(5, int(round(rng.normal(g.NFI_mean, g.NFI_sd))))
    sf_max = -1.0
    in_max = -1.0
    while sf_max <= 0 or in_max <= 0:
        sf_max = rng.normal(g.SF_max_mean, g.SF_max_sd)
        in_max = rng.normal(g.IN_max_mean, g.IN_max_sd)

    sf_profile = _normalised_profile(
        g.intercept_0_SF, g.intercept_1_SF, g.max_normalized_rank_SF, nfi) * sf_max
    in_profile = _normalised_profile(
        g.intercept_0_IN, g.intercept_1_IN, g.max_normalized_rank_IN, nfi) * in_max

    # Secondary leaf count: negative binomial, last 6 phytomers = 0
    p_nb = g.size_r_binorm / (g.size_r_binorm + g.mu_r_binorm)
    n_sec = np.zeros(nfi, dtype=int)
    if nfi > 7:
        n_sec[:nfi - 6] = rng.negative_binomial(
            max(0.01, g.size_r_binorm), np.clip(p_nb, 0.001, 0.999),
            size=nfi - 6)

    # Surface areas (TopVine /1.04 convention)
    sf_I = sf_profile / 1.04
    sf_II_mean = np.zeros(nfi)
    for i in range(nfi):
        if n_sec[i] > 0:
            slope = max(0.1, rng.normal(g.slope_NFII_SFII, g.slope_sd_NFII_SFII))
            sf_II_mean[i] = (n_sec[i] * slope) / (1.04 * n_sec[i])

    return sf_I, sf_II_mean, in_profile, n_sec


# ═══════════════════════════════════════════════════════════════════════════
# TopVine coordinate utilities  (exact port of coor3D.py)
# ═══════════════════════════════════════════════════════════════════════════

def _xyz_to_pol(c):
    x, y, z = float(c[0]), float(c[1]), float(c[2])
    r = math.sqrt(x*x + y*y + z*z)
    incli = math.asin(z / r) if r > 1e-12 else 0.0
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        azi = 0.0
    elif y >= 0:
        azi = math.acos(max(-1, min(1, x / math.sqrt(x*x + y*y))))
    else:
        azi = -math.acos(max(-1, min(1, x / math.sqrt(x*x + y*y))))
    return np.array([r, azi, incli])


def _pol_to_xyz(c):
    r, azi, incli = float(c[0]), float(c[1]), float(c[2])
    z = r * math.sin(incli)
    l2 = max(0, r*r - z*z)
    ll = math.sqrt(l2)
    x = ll * math.cos(azi)
    y = ll * math.sin(azi)
    return np.array([x, y, z])


def _rotate_axis(coord, r_azi, r_incli):
    """Exact port of TopVine coor3D.RotateAxis."""
    # inclination first: rotation around Y
    pol1 = _xyz_to_pol(np.array([coord[0], coord[2], -coord[1]]))
    xyz1 = _pol_to_xyz(np.array([pol1[0], pol1[1] + r_incli, pol1[2]]))
    # azimuth second: rotation around Z
    pol2 = _xyz_to_pol(np.array([xyz1[0], -xyz1[2], xyz1[1]]))
    xyz2 = _pol_to_xyz(np.array([pol2[0], pol2[1] + r_azi, pol2[2]]))
    return xyz2


# ═══════════════════════════════════════════════════════════════════════════
# TopVine leaf placement  (exact port)
# ═══════════════════════════════════════════════════════════════════════════

def _random_cyl(rng, r=1.0, h=1.0):
    """Random point inside a cylinder  (TopVine leaf.random_cyl)."""
    while True:
        rx = rng.uniform(-r, r)
        ry = rng.uniform(-r, r)
        if rx*rx + ry*ry <= r*r:
            return np.array([rx, ry, rng.uniform(0, h)])


def _set_coord0(cyl_coord, Lin, Lb, omega=math.pi/4):
    """Map normalised cyl coord to SOR parallelogram  (TopVine leaf.set_coord0).

    Returns coord in local internode frame (metres).
    """
    c = np.array([
        cyl_coord[0] * Lb * math.cos(omega),
        cyl_coord[1] * Lb * math.cos(omega),
        cyl_coord[2] * Lin,
    ])
    d = math.sqrt(c[0]**2 + c[1]**2)
    return np.array([c[0], c[1], c[2] + math.tan(omega) * d])


def _set_coordF(coord, origin, r_azi, r_incli):
    """Rotate + translate to world frame  (TopVine Topiary.set_coordF)."""
    rotated = _rotate_axis(coord, r_azi, r_incli)
    return rotated + origin


def _sample_leaf_angles_normalised(lawf, rng):
    """Draw normalised (0,1) angles at leaf construction  (TopVine Leaf.__init__)."""
    # Elevation: always Gaussian normalised draw
    elv_norm = rng.normal(0, 1)
    # Azimuth: check type
    if lawf[2][1] == 2:  # uniform
        azi_val = rng.uniform(-180, 180)
        azi_type = 2
    else:
        azi_val = rng.normal(0, 1)
        azi_type = 1
    return elv_norm, azi_val, azi_type


def _set_anglesF(elv_norm, azi_val, azi_type, lawf, is_north):
    """Scale normalised angles to actual degrees  (TopVine leaf.set_anglesF)."""
    if is_north:
        melv, sdelv = lawf[1][2], lawf[1][3]
        mazi, sdazi = lawf[3][2], lawf[3][3]
    else:
        melv, sdelv = lawf[0][2], lawf[0][3]
        mazi, sdazi = lawf[2][2], lawf[2][3]

    elv_deg = melv + elv_norm * sdelv
    if azi_type == 1:
        azi_deg = mazi + azi_val * sdazi
    else:
        azi_deg = azi_val  # already in degrees

    return elv_deg, azi_deg


def _allo_LN(allo, N):
    """Allometric petiole length  (TopVine Topiary.allo_LN).

    Returns length in metres.
    """
    L = allo[0] * N + allo[1]
    if L <= 0.005:
        L = 0.005
    return L * 0.001


# ═══════════════════════════════════════════════════════════════════════════
# Shoot parameter generation  (port of gen_shoot_param.py)
# ═══════════════════════════════════════════════════════════════════════════

def _gen_spur_positions(n_shoots, rng):
    """Generate spur (x,y,z) positions in CM along the cordon.

    Faithful port of gen_shoot_param.gen_spurs.
    """
    positions = []
    # First 6 shoots: one per spur rank, random order
    ranks = list(range(min(6, n_shoots)))
    rng.shuffle(ranks)
    for rank in ranks[:n_shoots]:
        sp = SPURS0_CM[rank]
        x = rng.normal(sp[0][0], sp[0][1])
        y = rng.normal(sp[1][0], sp[1][1])
        z = rng.normal(sp[2][0], sp[2][1])
        positions.append(np.array([x, y, z]))

    # Shoots 7-12: offset from first 6
    if n_shoots > 6:
        for i in range(6, min(12, n_shoots)):
            ref = positions[i % 6]
            dx = rng.normal(DSPURS_CM[0][0], DSPURS_CM[0][1])
            dy = rng.normal(DSPURS_CM[1][0], DSPURS_CM[1][1])
            dz = rng.normal(DSPURS_CM[2][0], DSPURS_CM[2][1])
            positions.append(ref + np.array([dx, dy, dz]))

    # Beyond 12: random spur rank
    for i in range(12, n_shoots):
        rank = rng.integers(0, 6)
        sp = SPURS0_CM[rank]
        x = rng.normal(sp[0][0], sp[0][1])
        y = rng.normal(sp[1][0], sp[1][1])
        z = rng.normal(sp[2][0], sp[2][1])
        positions.append(np.array([x, y, z]))

    # Sort by x (left to right along cordon)
    positions.sort(key=lambda p: p[0])
    return positions


def _gen_shoot_geom(rng):
    """Sample shoot geometry from multivariate distribution.

    Returns (azimuth_deg, alpha_deg, phi_deg, Ls, MX).
    Faithful port of gen_shoot_param.gen_shoot, with VSP corrections:
      - azimuth forced to trellis plane (along X ± small spread)
      - phi zeroed (was causing systematic drone-side lean)
      - Ls capped at 1.3 for realistic VSP height (~1.8-2.0m total)
    """
    r = rng.random()
    b = np.cumsum(F_AZI)
    if r <= b[0]:
        mean, cov = SHOOT_STATS[0]
    elif r <= b[1]:
        mean, cov = SHOOT_STATS[1]
    elif r <= b[2]:
        mean, cov = SHOOT_STATS[2]
    else:
        mean, cov = SHOOT_STATS[3]

    # Ensure covariance is positive semi-definite
    cov_safe = (cov + cov.T) / 2
    eigvals = np.linalg.eigvalsh(cov_safe)
    if eigvals.min() < 0:
        cov_safe += (-eigvals.min() + 1e-6) * np.eye(4)

    sh = rng.multivariate_normal(mean, cov_safe)
    # sh = [alpha, phi, Ls, MX]
    alpha = float(np.clip(sh[0], 45, 85))   # upright VSP shoots; was 20° min causing horizontal escapes
    phi   = 0.0   # zeroed: was causing systematic lean toward drone (-y)
    Ls    = float(np.clip(sh[2], 0.3, 1.3))   # capped for VSP ~1.8-2.0m total height
    MX    = float(np.clip(sh[3], 0.2, 0.9))

    # VSP: shoots grow along the trellis plane (X axis), ±15° spread,
    # ±8° out-of-plane wobble symmetric so neither side dominates
    along_row = rng.choice([0.0, 180.0])
    azi = float(along_row + rng.uniform(-15, 15) + rng.uniform(-8, 8))

    return azi, alpha, phi, Ls, MX


# ═══════════════════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════════════════

class VineyardGenerator:

    def __init__(
        self,
        assets_dir: str = "assets",
        genotype: str | Genotype = "Grenache",
        rng: Optional[np.random.Generator] = None,
    ):
        self.assets_dir = assets_dir
        self._rng       = rng
        if isinstance(genotype, str):
            self.genotype = GENOTYPES.get(genotype, Genotype())
        else:
            self.genotype = genotype

    def generate(
        self,
        path: Optional[str] = None,
        thomas: Optional[ThomasParams] = None,
        seed: Optional[int] = None,
        genotype: Optional[str | Genotype] = None,
    ) -> str:
        """Generate a bare-wood vineyard XML string.

        Vine structure is randomised via the Thomas cluster process each call.
        No leaves or grapes are included -- this models the dormant season
        used for phenotyping and active reconstruction.
        """
        rng = (np.random.default_rng(seed) if seed is not None
               else (self._rng or np.random.default_rng()))
        geno = self.genotype
        if genotype is not None:
            geno = GENOTYPES[genotype] if isinstance(genotype, str) else genotype

        tp     = thomas or ThomasParams.sample(rng)
        result = self._build_stand(geno, rng)

        self.last_trunk_segs = result["trunk_segs"]
        self.last_shoot_segs = result["segments"]
        self.last_n_shoots   = result["n_shoots"]

        xml = self._render_xml(result["segments"], result["trunk_segs"], tp)
        if path:
            Path(path).write_text(xml)
        return xml

    # ------------------------------------------------------------------
    # Build the whole stand  (port of topvine_2023.topvine)
    # ------------------------------------------------------------------
    def _build_stand(self, geno: Genotype, rng: np.random.Generator) -> dict:
        all_segments = []
        all_leaves   = []
        all_trunk_segs = []   # list of ((x0,y0,z0),(x1,y1,z1), radius)
        all_grape_sites = []
        total_shoots = 0  # accumulate across all plants

        # Place plants along the row  (carto: 110cm spacing)
        plant_spacing_cm = 110.0
        row_len_cm = (ROW_X_MAX - ROW_X_MIN) * 100
        n_plants = max(2, int(row_len_cm / plant_spacing_cm))
        plant_xs_cm = np.linspace(
            ROW_X_MIN * 100 + 30, ROW_X_MAX * 100 - 30, n_plants)

        for pi, px_cm in enumerate(plant_xs_cm):
            plant_origin_cm = np.array([px_cm, 0.0, 0.0])
            n_shoots = max(6, int(round(rng.normal(13, 1.5))))
            total_shoots += n_shoots
            px_m = px_cm * 0.01

            # ── Build multi-segment trunk ──
            # Curved trunk from ground to cordon with 5-6 segments
            # Slight lean and S-curve like a real vine trunk
            n_trunk_segs = rng.integers(5, 8)
            lean_x = float(rng.normal(0, 0.02))  # slight lean
            lean_y = float(rng.normal(0, 0.015))
            trunk_top = np.array([px_m + lean_x, CORDON_Y + lean_y, CORDON_Z])
            trunk_base = np.array([px_m, CORDON_Y, 0.02])

            prev = trunk_base.copy()
            for ts in range(n_trunk_segs):
                t = (ts + 1) / n_trunk_segs
                # Interpolate with some wobble
                wobble_x = rng.normal(0, 0.008) * math.sin(t * math.pi)
                wobble_y = rng.normal(0, 0.006) * math.sin(t * math.pi)
                pt = trunk_base + (trunk_top - trunk_base) * t
                pt[0] += wobble_x
                pt[1] += wobble_y
                # Taper: thicker at base (0.035), thinner at top (0.020)
                radius = 0.035 - 0.015 * t
                all_trunk_segs.append((
                    (float(prev[0]), float(prev[1]), float(prev[2])),
                    (float(pt[0]), float(pt[1]), float(pt[2])),
                    float(radius)))
                prev = pt.copy()

            # ── Build cordon arms + spur positions ──
            # Generate spur X-offsets from statistical distribution (in CM)
            spur_positions_cm = _gen_spur_positions(n_shoots, rng)

            # Sort by X so we can connect them along the cordon
            spur_positions_cm.sort(key=lambda p: p[0])

            # Convert spur positions to metres, clamping X to a randomised
            # reach from trunk (varies per arm so cordons aren't all identical)
            cordon_reach = float(rng.uniform(0.30, 0.50))
            spur_points_m = []
            for sp_cm in spur_positions_cm:
                sx = float(np.clip(px_m + sp_cm[0] * 0.01,
                                   px_m - cordon_reach, px_m + cordon_reach))
                sy = trunk_top[1] + sp_cm[1] * 0.001
                sz = float(trunk_top[2])   # keep exactly at cordon height
                spur_points_m.append(np.array([sx, sy, sz]))

            # Build cordon arm segments: trunk_top → through each spur
            # Multi-segment with slight natural waviness between waypoints
            cordon_points = sorted(spur_points_m, key=lambda p: p[0])
            left_spurs  = [p for p in cordon_points if p[0] < trunk_top[0]]
            right_spurs = [p for p in cordon_points if p[0] >= trunk_top[0]]

            # Cordon radius tapers from trunk outward
            CORDON_RAD_BASE = 0.022
            CORDON_RAD_TIP  = 0.013

            for arm_spurs in [list(reversed(left_spurs)), right_spurs]:
                if not arm_spurs:
                    continue
                # Build waypoints: trunk_top → each spur → slight extension
                waypoints = [trunk_top.copy()]
                for sp in arm_spurs:
                    waypoints.append(sp.copy())
                # Extend a little past the last spur
                if len(waypoints) >= 2:
                    direction = waypoints[-1] - waypoints[-2]
                    direction[2] = 0.0  # keep horizontal
                    norm = np.linalg.norm(direction)
                    if norm > 0.001:
                        waypoints.append(waypoints[-1] + direction / norm * 0.03)

                # Between each pair of waypoints, add 2-3 intermediate
                # points with slight y/z perturbation for natural look
                n_wp = len(waypoints)
                for wi in range(n_wp - 1):
                    p_start = waypoints[wi]
                    p_end   = waypoints[wi + 1]
                    n_sub = rng.integers(2, 4)  # 2-3 sub-segments
                    t_frac = float(wi) / max(1, n_wp - 2)  # 0→1 along arm
                    rad = CORDON_RAD_BASE + (CORDON_RAD_TIP - CORDON_RAD_BASE) * t_frac

                    prev_pt = p_start.copy()
                    for si in range(n_sub):
                        t = (si + 1) / n_sub
                        pt = p_start + (p_end - p_start) * t
                        # Add slight droop and sideways wobble (not on endpoints)
                        if si < n_sub - 1:
                            pt[1] += rng.normal(0, 0.004)  # y wobble
                            pt[2] += rng.normal(0, 0.003)  # z wobble (slight droop)
                        all_trunk_segs.append((
                            (float(prev_pt[0]), float(prev_pt[1]), float(prev_pt[2])),
                            (float(pt[0]),      float(pt[1]),      float(pt[2])),
                            float(rad)))
                        prev_pt = pt.copy()

            # ── Build shoots starting at each spur point ──
            for si in range(n_shoots):
                # Spur position in CM (for Topiary_2023 which expects CM input)
                spur_m = spur_points_m[si]
                spur_cm = spur_m * 100.0  # back to CM for Topiary

                # Generate shoot geometry
                azi_deg, alpha_deg, phi_deg, Ls, MX = _gen_shoot_geom(rng)

                # Generate topology from genotype
                sf_I, sf_II, in_profile_cm, n_sec = _generate_rameau_moyen(
                    geno, rng)
                n_phyto = len(sf_I)

                # Build topology lists
                topo = []
                lins = []
                for k in range(n_phyto):
                    phyto_leaves = [sf_I[k]]
                    for _ in range(n_sec[k]):
                        phyto_leaves.append(sf_II[k])
                    topo.append(phyto_leaves)
                    lins.append([in_profile_cm[k]])

                # Run Topiary_2023 algorithm — shoot starts at spur point
                segs, lvs, grape_nodes = self._topiary_2023(
                    spur_cm, azi_deg, alpha_deg, Ls, MX,
                    topo, lins, n_phyto, rng)
                all_segments.extend(segs)
                all_leaves.extend(lvs)
                all_grape_sites.extend(grape_nodes)

        return {"segments": all_segments, "leaves": all_leaves,
                "trunk_segs": all_trunk_segs, "grape_sites": all_grape_sites,
                "n_shoots": total_shoots}

    # ------------------------------------------------------------------
    # Faithful port of Topiary_2023.__init__
    # ------------------------------------------------------------------
    def _topiary_2023(
        self,
        spur_pos_cm,     # [x, y, z] in CM
        azimuth_deg,     # shoot azimuth in horizontal plane
        alpha_deg,       # initial elevation (= CourbureActu start)
        Ls,              # length factor
        MX,              # inflexion proportion
        topo,            # topo[phyto] = [sf_I, sf_II_1, sf_II_2, ...]
        lins,            # lins[phyto] = [internode_length_cm]
        n_phyto,
        rng,
    ):
        segments    = []
        leaves      = []
        grape_nodes = []  # (x, y, z) positions for grape attachment

        # LongRamMoy = allo_LN(ALLO_I, NombrePhyto)  → metres
        LongRamMoy = _allo_LN(ALLO_I, n_phyto)

        # Shoot length and per-phytomer length
        LongRam   = Ls * LongRamMoy
        LongPhyto = LongRam / n_phyto  # fallback if lin is None

        # Curvature  (exact TopVine: starts at alpha, adds alpha/(2·N))
        CourbureActu = alpha_deg
        NumInflexion = max(1, math.ceil(MX * n_phyto))

        # Starting position  (CM → M)
        Xi = spur_pos_cm[0] * 0.01
        Yi = spur_pos_cm[1] * 0.01
        Zi = spur_pos_cm[2] * 0.01

        azi_rad = azimuth_deg * math.pi / 180.0
        omega   = 45 * math.pi / 180.0
        LongPetiole = 0.12  # constant for primary leaves (TopVine default)

        for phyto in range(n_phyto):
            # Internode length: use individual value from topology
            Lin = lins[phyto][0] * 0.01  # cm → m
            if Lin <= 0 or Lin is None:
                Lin = LongPhyto

            # Curvature step  (TopVine Topiary_2023)
            # Before inflexion: angle increases (shoot straightens toward vertical)
            # After inflexion: angle DECREASES (shoot arches over toward horizontal)
            if phyto < NumInflexion:
                dCourbure = alpha_deg / 2.0 / NumInflexion
            else:
                dCourbure = -alpha_deg / 2.0 / max(1, n_phyto - NumInflexion)

            # Soft wire cap: accelerate arching when approaching top wire
            # instead of hard-clamping Z
            if Zi > WIRE2_Z - 0.10 and CourbureActu > 20:
                dCourbure = -abs(dCourbure) - 5.0  # force strong arch-over

            # Next node position  (exact TopVine eq.)
            courb_rad = CourbureActu * math.pi / 180.0
            Xii = math.cos(courb_rad) * Lin * math.cos(azi_rad) + Xi
            Yii = math.cos(courb_rad) * Lin * math.sin(azi_rad) + Yi
            Zii = math.sin(courb_rad) * Lin + Zi
            Zii = max(CORDON_Z, Zii)   # floor at cordon height, allow natural arch

            # Branch segment
            segments.append(((Xi, Yi, Zi), (Xii, Yii, Zii)))

            # ── Primary leaf (Leaf I) ──
            # Normalised angles (drawn at leaf construction)
            elv_norm, azi_val, azi_type = _sample_leaf_angles_normalised(
                LAWF, rng)
            # Random cyl coord for insertion point
            cyl = _random_cyl(rng)
            # SOR parallelogram mapping
            Lb_I = LongPetiole
            coord = _set_coord0(cyl, Lin, Lb_I, omega)
            # Rotate to shoot frame + translate to world
            r_azi_frame   = azimuth_deg * math.pi / 180.0
            r_incli_frame = (-90 + CourbureActu) * math.pi / 180.0
            coordF = _set_coordF(coord, np.array([Xi, Yi, Zi]),
                                 r_azi_frame, r_incli_frame)
            # N/S status
            is_north = coordF[1] < 0
            elv_deg, _ = _set_anglesF(
                elv_norm, azi_val, azi_type, LAWF, is_north)
            tilt = float(np.clip(-(90.0 - elv_deg), -65, 30))
            # Face outward from canopy center: -y side → 180°, +y side → 0°
            azi_mj = float((180.0 if coordF[1] < CORDON_Y else 0.0) + rng.uniform(-60, 60))
            leaves.append({
                "x": float(coordF[0]),
                "y": float(coordF[1]),
                "z": float(max(0.05, coordF[2])),
                "euler_x": float(rng.uniform(-10, 10)),
                "euler_y": float(tilt + rng.uniform(-10, 10)),
                "euler_z": float(azi_mj),
                "sf_cm2": float(topo[phyto][0]),
            })

            # ── Secondary leaves (Leaf II) ──
            n_sec = len(topo[phyto]) - 1
            if n_sec > 0:
                Lb_II = _allo_LN(ALLO_II, n_sec)
                for j in range(1, len(topo[phyto])):
                    elv_n2, azi_v2, azi_t2 = _sample_leaf_angles_normalised(
                        LAWF, rng)
                    cyl2 = _random_cyl(rng)
                    coord2 = _set_coord0(cyl2, Lin, Lb_II, omega)
                    coordF2 = _set_coordF(coord2, np.array([Xi, Yi, Zi]),
                                          r_azi_frame, r_incli_frame)
                    is_n2 = coordF2[1] < 0
                    elv_d2, _ = _set_anglesF(
                        elv_n2, azi_v2, azi_t2, LAWF, is_n2)
                    tilt2 = float(np.clip(-(90.0 - elv_d2), -65, 30))
                    azi_mj2 = float((180.0 if coordF2[1] < CORDON_Y else 0.0) + rng.uniform(-60, 60))
                    leaves.append({
                        "x": float(coordF2[0]),
                        "y": float(coordF2[1]),
                        "z": float(max(0.05, coordF2[2])),
                        "euler_x": float(rng.uniform(-10, 10)),
                        "euler_y": float(tilt2 + rng.uniform(-10, 10)),
                        "euler_z": float(azi_mj2),
                        "sf_cm2": float(topo[phyto][j]),
                    })

            # Grape attachment: phytomers 3-8 (fruiting zone, between
            # cordon ~0.6m and wire1 ~1.2m)
            if 2 <= phyto <= 7 and CORDON_Z < Zii < WIRE1_Z + 0.1:
                grape_nodes.append({
                    "x": float(Xii), "y": float(Yii), "z": float(Zii),
                })

            # Advance along shoot
            Xi = Xii
            Yi = Yii
            Zi = Zii
            CourbureActu += dCourbure

        return segments, leaves, grape_nodes

    # ------------------------------------------------------------------
    # Grape placement on shoot nodes  (attached to lower phytomers)
    # ------------------------------------------------------------------
    def _place_grapes_on_shoots(self, grape_sites, rng: np.random.Generator):
        """Place grape clusters hanging from shoot nodes.

        Real grape clusters hang ~10-15cm below the attachment node on a
        short peduncle. We pick a subset of basal phytomer nodes and place
        the grape mesh at the hanging position.
        """
        if not grape_sites:
            return []

        rng.shuffle(grape_sites)
        # ~1 in 3 eligible nodes gets a cluster
        n_grapes = min(20, max(5, len(grape_sites) // 3))

        grapes = []
        for site in grape_sites[:n_grapes]:
            # Grape hangs directly below the node with small xy scatter
            gx = float(site["x"] + rng.normal(0, 0.01))
            gy = float(site["y"] + rng.normal(0, 0.01))
            # Hang point: the grape_xml template already drops by GRAPE_Z_DROP
            # from the stem pos, so place stem AT the node
            gz = float(max(site["z"], CORDON_Z + GRAPE_Z_DROP + 0.05))
            grapes.append({"x": gx, "y": gy, "z": gz})

        grapes = _thin_points(grapes, 0.08, rng)
        return grapes[:20]

    # ------------------------------------------------------------------
    # XML rendering
    # ------------------------------------------------------------------
    def _render_xml(self, segments, trunk_segs, tp) -> str:
        ad = self.assets_dir
        parts = []

        parts.append(textwrap.dedent(f"""\
            <!-- AUTO-GENERATED  TopVine vine stand  (Louarn et al. 2008)      -->
            <!-- Genotype: {self.genotype.name}  -- bare wood (dormant season) -->
            <!-- Thomas: lp={tp.lambda_p:.2f} mc={tp.mu_c:.2f} sx={tp.sigma_x:.3f} sz={tp.sigma_z:.3f} -->
            <mujoco model="vineyard_topvine">

              <visual>
                <headlight ambient="0.7 0.7 0.7" diffuse="0.8 0.8 0.8"/>
                <quality shadowsize="0"/>
                <global offwidth="1920" offheight="1080"/>
              </visual>

              <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"
                      wind="0 0 0" density="1.225" viscosity="0.00002"/>

              <default>
                <joint armature="0.00005" frictionloss="0"/>
                <geom contype="0" conaffinity="0"/>
                <default class="x2">
                  <geom mass="0"/>
                  <motor ctrlrange="0 13"/>
                  <mesh scale="0.01 0.01 0.01"/>
                  <default class="x2_visual">
                    <geom group="2" type="mesh" contype="0" conaffinity="0"/>
                  </default>
                  <default class="x2_collision">
                    <geom group="3" type="box"/>
                    <default class="x2_rotor">
                      <geom type="ellipsoid" size=".13 .13 .01"/>
                    </default>
                  </default>
                  <site group="5"/>
                </default>
              </default>

              <asset>
                <texture name="x2_tex" type="2d"
                         file="{ad}/skydio_x2/X2_lowpoly_texture_SpinningProps_1024.png"/>
                <material name="x2_phong3SG" texture="x2_tex"/>
                <material name="x2_invisible" rgba="0 0 0 0"/>
                <mesh name="x2_mesh" file="{ad}/skydio_x2/X2_lowpoly.obj" class="x2"/>
              </asset>

              <worldbody>
                <light name="sun" directional="true" pos="0 -2 4"
                       dir="0.15 0.4 -1" diffuse="0.85 0.85 0.80" ambient="0.1 0.1 0.1"/>
                <light name="fill" directional="true" pos="0 3 3"
                       dir="-0.1 -0.3 -0.7" diffuse="0.25 0.28 0.32" ambient="0 0 0"/>
                <geom name="floor" type="plane" size="10 10 0.1"
                      contype="1" conaffinity="1"/>
            """))

        # Trunks + cordon arms  (multi-segment, connected)
        parts.append("\n            <!-- Trunk and cordon segments -->")
        for ti, (p0, p1, rad) in enumerate(trunk_segs):
            x0, y0, z0 = p0
            x1, y1, z1 = p1
            d = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
            if d < 0.001:
                continue
            parts.append(f"""\
                <body name="trunk_{ti}" pos="0 0 0">
                  <geom type="capsule" fromto="{x0:.4f} {y0:.4f} {z0:.4f}  {x1:.4f} {y1:.4f} {z1:.4f}"
                        size="{rad:.4f}" rgba="0.35 0.22 0.10 1"
                        mass="0.005" contype="0" conaffinity="0"/>
                </body>""")

        # Shoot branches (internode segments from Topiary_2023)
        parts.append("\n            <!-- Shoot internodes -->")
        br_idx = 0
        for (p0, p1) in segments:
            x0, y0, z0 = p0
            x1, y1, z1 = p1
            # Skip degenerate segments
            d = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
            if d < 0.001:
                continue
            parts.append(f"""\
                <body name="br_{br_idx}" pos="0 0 0">
                  <geom type="capsule" fromto="{x0:.4f} {y0:.4f} {z0:.4f}  {x1:.4f} {y1:.4f} {z1:.4f}"
                        size="0.008" rgba="0.40 0.28 0.12 1"
                        mass="0.0005" contype="0" conaffinity="0"/>
                </body>""")
            br_idx += 1

        # Drone
        parts.append(textwrap.dedent(f"""\

                <!-- Skydio X2 -->
                <light name="x2_spot" mode="targetbodycom" target="x2"
                       pos="0 -1 2.5" diffuse="0.3 0.3 0.3" ambient="0 0 0"/>
                <camera name="overview" pos="0.0 -4.5 1.4"
                        xyaxes="1 0 0  0 0.25 0.97" fovy="52"/>

                <body name="x2" pos="0.0 -0.9 1.2" childclass="x2">
                  <freejoint/>
                  <camera name="drone_cam" pos="0.0 0.25 0.05"
                          xyaxes="0.000 1.000 0.000  -0.500 0.000 0.866"/>
                  <site name="imu" pos="0 0 .02"/>
                  <geom material="x2_phong3SG" mesh="x2_mesh" class="x2_visual" quat="0 0 1 1"/>
                  <geom class="x2_collision" size=".06 .027 .02" pos=".04 0 .02"/>
                  <geom class="x2_collision" size=".06 .027 .02" pos=".04 0 .06"/>
                  <geom class="x2_collision" size=".05 .027 .02" pos="-.07 0 .065"/>
                  <geom class="x2_collision" size=".023 .017 .01" pos="-.137 .008 .065" quat="1 0 0 1"/>
                  <geom name="rotor1" class="x2_rotor" pos="-.14 -.18 .05" mass=".25"/>
                  <geom name="rotor2" class="x2_rotor" pos="-.14 .18 .05" mass=".25"/>
                  <geom name="rotor3" class="x2_rotor" pos=".14 .18 .08" mass=".25"/>
                  <geom name="rotor4" class="x2_rotor" pos=".14 -.18 .08" mass=".25"/>
                  <geom size=".16 .04 .02" pos="0 0 0.02" type="ellipsoid" mass=".325"
                        class="x2_visual" material="x2_invisible"/>
                  <site name="thrust1" pos="-.14 -.18 .05"/>
                  <site name="thrust2" pos="-.14 .18 .05"/>
                  <site name="thrust3" pos=".14 .18 .08"/>
                  <site name="thrust4" pos=".14 -.18 .08"/>
                </body>

              </worldbody>

              <actuator>
                <motor class="x2" name="thrust1" site="thrust1" gear="0 0 1 0 0 -.0201"/>
                <motor class="x2" name="thrust2" site="thrust2" gear="0 0 1 0 0  .0201"/>
                <motor class="x2" name="thrust3" site="thrust3" gear="0 0 1 0 0 -.0201"/>
                <motor class="x2" name="thrust4" site="thrust4" gear="0 0 1 0 0  .0201"/>
              </actuator>

              <sensor>
                <gyro          name="body_gyro"   site="imu"/>
                <accelerometer name="body_linacc"  site="imu"/>
                <framequat     name="body_quat"   objtype="site" objname="imu"/>
              </sensor>

            </mujoco>
            """))

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------
def _leaf_xml(i: int, lf: dict) -> str:
    name = f"lf{i}"
    x, y, z = lf["x"], lf["y"], lf["z"]
    ex, ey, ez = lf["euler_x"], lf["euler_y"], lf["euler_z"]
    pl = LEAF_PETIOLE_VIS
    return f"""\
            <body name="{name}_stem" pos="{x:.4f} {y:.4f} {z:.4f}" euler="0 0 {ez:.1f}">
              <joint name="{name}_stem_bend" type="hinge" axis="1 0 0"
                     stiffness="{STIFF_STEM}" damping="{DAMP_STEM}"/>
              <joint name="{name}_stem_side" type="hinge" axis="0 0 1"
                     stiffness="{STIFF_STEM}" damping="{DAMP_STEM}"/>
              <geom type="capsule" fromto="0 0 0  0 {pl:.3f} 0" size="0.003"
                    rgba="0.35 0.22 0.10 0" mass="0.00005"
                    contype="0" conaffinity="0" fluidshape="none"/>
              <body name="{name}" pos="0 {pl:.3f} 0" euler="{ex:.1f} {ey:.1f} 0">
                <joint name="{name}_rx" type="hinge" axis="1 0 0"
                       stiffness="{STIFF_BODY}" damping="{DAMP_BODY}"/>
                <joint name="{name}_ry" type="hinge" axis="0 1 0"
                       stiffness="{STIFF_BODY}" damping="{DAMP_BODY}"/>
                <joint name="{name}_rz" type="hinge" axis="0 0 1"
                       stiffness="{STIFF_RZ}" damping="{DAMP_RZ}"/>
                <geom name="{name}_vis" type="mesh" mesh="leaf_mesh" material="leaf_mat"
                      contype="0" conaffinity="0" mass="0.002"
                      fluidshape="ellipsoid" fluidcoef="0.5 0.1 0.25 0.8 0.8"/>
              </body>
            </body>"""


def _grape_xml(ii: int, gr: dict) -> str:
    name = f"grape_{ii:02d}"
    x, y, z = gr["x"], gr["y"], gr["z"]
    sl, zd = GRAPE_STEM_LEN, GRAPE_Z_DROP
    return f"""\
            <body name="{name}_stem" pos="{x:.4f} {y:.4f} {z:.4f}">
              <joint name="{name}_sway_x" type="hinge" axis="1 0 0"
                     range="-40 40" stiffness="{STIFF_GRAPE}" damping="{DAMP_GRAPE}"/>
              <joint name="{name}_sway_z" type="hinge" axis="0 0 1"
                     range="-20 20" stiffness="{STIFF_GRAPE_Z}" damping="{DAMP_GRAPE_Z}"/>
              <geom type="capsule" fromto="0 0 0  0 0 -{sl:.3f}" size="0.006"
                    rgba="0.35 0.20 0.08 1" mass="0"
                    contype="0" conaffinity="0"/>
              <body name="{name}" pos="0 0 -{zd:.3f}">
                <geom name="{name}_vis" type="mesh" mesh="grape_mesh" material="grape_mat"
                      contype="0" conaffinity="0" mass="0.08"
                      fluidshape="ellipsoid" fluidcoef="0.5 0.1 0.5 0.9 0.9"/>
              </body>
            </body>"""


def _thin_points(points, min_dist, rng):
    if not points:
        return points
    order = rng.permutation(len(points))
    kept, kept_xz = [], []
    for idx in order:
        p = points[idx]
        if not kept_xz:
            kept.append(p); kept_xz.append((p["x"], p["z"]))
        else:
            xs, zs = zip(*kept_xz)
            dists = np.sqrt((np.array(xs)-p["x"])**2 + (np.array(zs)-p["z"])**2)
            if dists.min() >= min_dist:
                kept.append(p); kept_xz.append((p["x"], p["z"]))
    return kept


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, re
    out = sys.argv[1] if len(sys.argv) > 1 else "vineyard_topvine.xml"
    geno_name = sys.argv[2] if len(sys.argv) > 2 else "Grenache"
    gen = VineyardGenerator(genotype=geno_name)
    xml = gen.generate(path=out, seed=42)
    br  = re.findall(r'name="(br_\d+)"', xml)
    tr  = re.findall(r'name="(trunk_\d+)"', xml)
    print(f"Generated: {out}  (genotype: {geno_name})")
    print(f"  Trunk segments : {len(tr)}   Shoot segments : {len(br)}")
    print(f"  Available genotypes: {list(GENOTYPES.keys())}")
