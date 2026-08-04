"""Stage 5: ODD Mapping -- static rare-event failure probability estimation.

Implements the structure of Jiang et al. 2024's ("Enhancing Autonomous
Vehicle Safety Based on Operational Design Domain Definition, Monitoring,
and Functional Degradation", IEEE TIV) Kriging-based Subset Simulation
(Eqs. 16-19):

    P_f = P(X in F_fail) = integral I_F(X) * phi_ODD(X) dX
    P_f_hat = P(F_m) * prod_{i=2}^{m} P(F_i | F_{i-1})
    P(F_m) = (sum I_F(X_i)) / N_m

i.e. decomposing a rare-event probability into a product of more-common
nested conditional probabilities, each easier to estimate than the rare
event itself. The paper's version uses a Kriging surrogate model plus
Metropolis-Hastings resampling to generate new candidate points at each
level from a live driving simulator -- this project has no simulator (see
the no-closed-loop-simulation scope decision), so this module instead
applies the same nested-threshold decomposition directly to a STATIC
population of risk scores (e.g. this project's fitted ODD copula density,
converted to a severity score via odd_boundary.py). Each level's threshold
is set as an empirical quantile of the current surviving population,
targeting a fixed conditional probability per level -- the same adaptive-
threshold idea as the paper, without the Kriging/MCMC machinery that a
live simulator would require.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.odd.odd_boundary import DEFAULT_ODD_VARIABLES, ODDCopulaModel, fit_odd_copula, odd_density


@dataclass
class SubsetSimulationResult:
    """Result of a staged, nested-conditional-probability failure estimate.

    Attributes:
        failure_probability: The final estimated P_f.
        level_thresholds: The severity threshold used at each level.
        level_conditional_probabilities: The conditional probability
            realized at each level (their product is failure_probability).
        num_levels_used: How many levels the decomposition actually used.
    """

    failure_probability: float
    level_thresholds: List[float]
    level_conditional_probabilities: List[float]
    num_levels_used: int


def estimate_failure_probability(
    population: np.ndarray,
    failure_threshold: float,
    conditional_prob_target: float = 0.1,
    max_levels: int = 10,
) -> SubsetSimulationResult:
    """Estimates P(X >= failure_threshold) via nested conditional probabilities.

    At each level, the threshold is set to the (1 - conditional_prob_target)
    quantile of the current surviving population, so each level retains
    approximately conditional_prob_target of the prior level's mass by
    construction (Jiang et al. Eqs. 17-19's adaptive-threshold idea). Once a
    level's candidate threshold reaches or exceeds failure_threshold, the
    exact conditional probability of failure within that level's surviving
    population is computed directly and the decomposition stops.

    Args:
        population: Continuous risk-score observations; larger values are
            more severe/less safe.
        failure_threshold: The score at/above which a scene counts as a
            failure (F_fail).
        conditional_prob_target: Target conditional probability retained
            per level.
        max_levels: Safety cap on the number of levels, in case
            failure_threshold is never reached (e.g. it exceeds the
            population's maximum).

    Returns:
        A SubsetSimulationResult.
    """
    current_population = np.asarray(population, dtype=float)
    level_thresholds: List[float] = []
    level_probs: List[float] = []

    for _ in range(max_levels):
        if len(current_population) < 2:
            break

        candidate_threshold = float(np.quantile(current_population, 1 - conditional_prob_target))
        if candidate_threshold >= failure_threshold:
            break

        level_thresholds.append(candidate_threshold)
        level_probs.append(conditional_prob_target)
        current_population = current_population[current_population >= candidate_threshold]

    # A final exact conditional probability within whatever population
    # survives every prior level -- computed unconditionally, regardless of
    # why the loop above stopped (threshold reached, population exhausted,
    # or max_levels hit). This correctly evaluates to 0 when
    # failure_threshold turns out to be unreachable from the population
    # (e.g. it exceeds the population's maximum), instead of silently
    # decaying to a small-but-nonzero product from unfinished levels.
    p_final = float(np.mean(current_population >= failure_threshold)) if len(current_population) else 0.0
    level_thresholds.append(failure_threshold)
    level_probs.append(p_final)

    failure_probability = float(np.prod(level_probs)) if level_probs else 0.0
    return SubsetSimulationResult(
        failure_probability=failure_probability,
        level_thresholds=level_thresholds,
        level_conditional_probabilities=level_probs,
        num_levels_used=len(level_probs),
    )


def density_to_severity(density_percentiles: np.ndarray, density_value: float) -> float:
    """Converts an ODD-copula density into a severity score in [0.0, 1.0].

    Severity is 1 minus the density's percentile rank within a reference
    (training) population: a scene sitting in the sparsest tail of the
    fitted ODD space (lowest density) gets severity close to 1.0; a scene
    in the densest, most typical region gets severity close to 0.0.

    Args:
        density_percentiles: A sorted reference array of density values
            (ODDCopulaModel.density_percentiles).
        density_value: The density value to convert.

    Returns:
        A severity score in [0.0, 1.0].
    """
    rank = np.searchsorted(density_percentiles, density_value, side="right")
    percentile_rank = rank / len(density_percentiles)
    return float(1.0 - percentile_rank)


def estimate_odd_failure_probability(
    copula_model: ODDCopulaModel,
    df: pd.DataFrame,
    severity_threshold: float = 0.95,
    conditional_prob_target: float = 0.1,
) -> SubsetSimulationResult:
    """Estimates the probability of landing in the extreme tail of the ODD space.

    Args:
        copula_model: A fitted ODDCopulaModel (see odd_boundary.py).
        df: A features DataFrame containing copula_model.variables.
        severity_threshold: The severity score (see density_to_severity)
            at/above which a scene counts as an ODD-space failure. 0.95
            means "the worst 5% of scenes by ODD density."
        conditional_prob_target: Target conditional probability per
            subset-simulation level.

    Returns:
        A SubsetSimulationResult over the dataset's severity scores.
    """
    densities = df[copula_model.variables].apply(
        lambda row: odd_density(copula_model, dict(zip(copula_model.variables, row))), axis=1
    )
    severities = densities.apply(
        lambda d: density_to_severity(copula_model.density_percentiles, d)
    )
    return estimate_failure_probability(
        severities.to_numpy(), severity_threshold, conditional_prob_target
    )


if __name__ == "__main__":
    from src.common.paths import FEATURES_CSV

    features_path = FEATURES_CSV

    features_df = pd.read_csv(features_path)
    copula = fit_odd_copula(features_df, DEFAULT_ODD_VARIABLES)

    for threshold in (0.90, 0.95, 0.99):
        result = estimate_odd_failure_probability(copula, features_df, severity_threshold=threshold)
        print(
            f"P(severity >= {threshold}) ~= {result.failure_probability:.6f} "
            f"over {result.num_levels_used} levels {result.level_conditional_probabilities}"
        )
