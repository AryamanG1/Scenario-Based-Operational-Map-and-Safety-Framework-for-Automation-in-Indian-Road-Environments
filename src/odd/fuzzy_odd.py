"""Stage 4: Scenario Classification -- fuzzy micro-ODD (uODD) partitioning.

Implements Salvi et al. 2022's ("Fuzzy Interpretation of Operational Design
Domains in Autonomous Driving", IEEE Intelligent Vehicles Symposium)
fuzzy uODD machinery: the triangular/trapezoidal membership functions
(their Appendix I), the Precipitation (P) and Precipitation-Deposits (PD)
linguistic variables (their Def. 3.2), and the NoRain/LowRain/HeavyRain
fuzzy rules (their Table I) -- combined with the Stage 3 traffic-density
category into a composite scenario label, per the capstone proposal's
Stage 4.

IDD-Lite has no precipitation-rate sensor, so the P (Precipitation)
variable is not observable here; PD (surface wetness/puddling) is
approximated from the image-derived 'wetness' feature (pixel-intensity
std, a reflective-surface proxy already computed in feature_extraction.py).
Table I's rules are OR-combinations of P and PD (e.g. "LowRain: P is Low
OR PD is Wet"), so evaluating them using only the PD term is a legitimate,
explicitly documented partial application of the same rules, not a
different model. Membership breakpoints are calibrated from the observed
wetness distribution's quartiles (as with traffic_density.py), since the
paper's own numeric breakpoints are in physical sensor units we don't have.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.odd.traffic_density import (
    DENSITY_LEVELS,
    TrafficDensityThresholds,
    classify_dataframe as classify_traffic_density,
)

NO_RAIN = "NoRain"
LOW_RAIN = "LowRain"
HEAVY_RAIN = "HeavyRain"
MU_ODD_STATES = (NO_RAIN, LOW_RAIN, HEAVY_RAIN)


def triangular_membership(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function Tr(x; a, b, c) (Salvi et al., Appendix I).

    Args:
        x: The value to evaluate membership for.
        a: Left foot (membership 0).
        b: Peak (membership 1).
        c: Right foot (membership 0).

    Returns:
        Membership degree in [0.0, 1.0].
    """
    left = (x - a) / (b - a) if b != a else 1.0
    right = (c - x) / (c - b) if c != b else 1.0
    return max(min(left, right), 0.0)


def trapezoidal_membership(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership function Tp(x; a, b, c, d) (Salvi et al., Appendix I).

    Args:
        x: The value to evaluate membership for.
        a: Left foot (membership 0).
        b: Left shoulder (membership reaches 1).
        c: Right shoulder (membership starts leaving 1).
        d: Right foot (membership 0).

    Returns:
        Membership degree in [0.0, 1.0].
    """
    left = (x - a) / (b - a) if b != a else 1.0
    right = (d - x) / (d - c) if d != c else 1.0
    return max(min(left, 1.0, right), 0.0)


@dataclass
class PDBreakpoints:
    """Quartile-calibrated breakpoints for the Precipitation-Deposits (PD) proxy.

    Attributes:
        w_min: Minimum observed wetness value.
        p25: 25th percentile of observed wetness.
        p50: Median observed wetness.
        p75: 75th percentile of observed wetness.
        w_max: Maximum observed wetness value.
    """

    w_min: float
    p25: float
    p50: float
    p75: float
    w_max: float

    def save(self, path: str) -> None:
        """Serializes the breakpoints to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PDBreakpoints":
        """Loads previously calibrated breakpoints from a JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


def calibrate_pd_breakpoints(wetness_values: List[float]) -> PDBreakpoints:
    """Calibrates PD membership breakpoints from an observed wetness distribution.

    Args:
        wetness_values: Observed 'wetness' feature values.

    Returns:
        A PDBreakpoints instance.
    """
    p25, p50, p75 = np.percentile(wetness_values, [25, 50, 75])
    return PDBreakpoints(
        w_min=float(np.min(wetness_values)),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        w_max=float(np.max(wetness_values)),
    )


def pd_dry_membership(wetness: float, bp: PDBreakpoints) -> float:
    """Membership degree of 'PD is Dry' (Salvi et al. Table I / Def. 3.2)."""
    return trapezoidal_membership(wetness, bp.w_min, bp.w_min, bp.p25, bp.p50)


def pd_wet_membership(wetness: float, bp: PDBreakpoints) -> float:
    """Membership degree of 'PD is Wet' (Salvi et al. Table I / Def. 3.2)."""
    return triangular_membership(wetness, bp.p25, bp.p50, bp.p75)


def pd_puddles_membership(wetness: float, bp: PDBreakpoints) -> float:
    """Membership degree of 'PD is Puddles' (Salvi et al. Table I / Def. 3.2)."""
    return trapezoidal_membership(wetness, bp.p50, bp.p75, bp.w_max, bp.w_max)


def classify_mu_odd(wetness: float, bp: PDBreakpoints) -> Dict[str, float]:
    """Computes fuzzy uODD membership degrees for one scene (Salvi et al. Table I).

    Rules (evaluated using the PD term only -- see module docstring):
        NoRain:    PD is Dry
        LowRain:   PD is Wet
        HeavyRain: PD is Puddles

    Args:
        wetness: The scene's 'wetness' feature value.
        bp: Calibrated PD breakpoints.

    Returns:
        A dict mapping each of MU_ODD_STATES to its membership degree.
    """
    return {
        NO_RAIN: pd_dry_membership(wetness, bp),
        LOW_RAIN: pd_wet_membership(wetness, bp),
        HEAVY_RAIN: pd_puddles_membership(wetness, bp),
    }


def defuzzify_mu_odd(memberships: Dict[str, float]) -> str:
    """Picks the crisp uODD label with the highest membership degree.

    Args:
        memberships: Output of classify_mu_odd().

    Returns:
        One of MU_ODD_STATES.
    """
    return max(memberships, key=memberships.get)


def compose_scenario_label(mu_odd_label: str, traffic_level: str) -> str:
    """Combines the weather micro-ODD and traffic-density category into one label.

    Args:
        mu_odd_label: One of MU_ODD_STATES.
        traffic_level: One of traffic_density.DENSITY_LEVELS.

    Returns:
        A composite scenario label, e.g. "HeavyRain-Congested".
    """
    return f"{mu_odd_label}-{traffic_level}"


def classify_dataframe(
    df: pd.DataFrame,
    pd_breakpoints: PDBreakpoints = None,
    density_thresholds: TrafficDensityThresholds = None,
):
    """Adds fuzzy uODD and composite scenario columns to a features frame.

    Args:
        df: A features DataFrame with 'wetness', 'vehicle_count',
            'num_two_wheelers', and 'num_autorickshaws' columns.
        pd_breakpoints: Pre-calibrated PD breakpoints to reuse; if None,
            calibrated fresh from this dataframe's own 'wetness' column.
        density_thresholds: Pre-calibrated traffic-density thresholds to
            reuse; if None, calibrated fresh from this dataframe.

    Returns:
        A tuple (df_with_scenario_columns, pd_breakpoints_used,
        density_thresholds_used).
    """
    if pd_breakpoints is None:
        pd_breakpoints = calibrate_pd_breakpoints(df["wetness"].tolist())

    df_out, density_thresholds = classify_traffic_density(df, density_thresholds)

    memberships = df_out["wetness"].apply(lambda w: classify_mu_odd(w, pd_breakpoints))
    df_out["mu_odd_no_rain"] = memberships.apply(lambda m: m[NO_RAIN])
    df_out["mu_odd_low_rain"] = memberships.apply(lambda m: m[LOW_RAIN])
    df_out["mu_odd_heavy_rain"] = memberships.apply(lambda m: m[HEAVY_RAIN])
    df_out["mu_odd_label"] = memberships.apply(defuzzify_mu_odd)
    df_out["scenario_label"] = df_out.apply(
        lambda row: compose_scenario_label(row["mu_odd_label"], row["traffic_density_level"]), axis=1
    )

    return df_out, pd_breakpoints, density_thresholds


if __name__ == "__main__":
    from src.common.paths import FEATURES_CSV, FUZZY_PD_BREAKPOINTS_JSON, ensure_output_dirs

    ensure_output_dirs()
    features_path = FEATURES_CSV
    pd_breakpoints_path = FUZZY_PD_BREAKPOINTS_JSON

    features_df = pd.read_csv(features_path)
    labeled_df, bp, thresholds = classify_dataframe(features_df)
    bp.save(pd_breakpoints_path)

    print(f"PD breakpoints: {bp}")
    print("Fuzzy uODD label distribution:")
    print(labeled_df["mu_odd_label"].value_counts().reindex(MU_ODD_STATES))
    print("\nComposite scenario label distribution (top 10):")
    print(labeled_df["scenario_label"].value_counts().head(10))
