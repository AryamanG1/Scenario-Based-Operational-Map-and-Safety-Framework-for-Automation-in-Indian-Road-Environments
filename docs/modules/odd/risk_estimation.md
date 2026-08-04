# `risk_estimation.py`

**Stage:** 5
**Package:** `src.odd.risk_estimation`

## Purpose

This module implements the remaining piece of Stage 5, "ODD Mapping": estimating the probability that a scene lands in the rare, extreme tail of the ODD space — i.e. a "failure" event, in the sense of being far outside the region the system was characterized on. Directly estimating a very small probability by naive Monte Carlo sampling is statistically inefficient (you'd need enormous sample counts to observe enough rare events to estimate their rate accurately). This module instead decomposes the rare-event probability into a product of several more-common, nested conditional probabilities, each individually easy to estimate — the same statistical idea used by the source paper's Subset Simulation method, adapted to work over a static population of scores rather than a live simulator.

In this pipeline, the "risk score" fed into this module is a severity score derived from `odd_boundary.py`'s fitted Gaussian-copula ODD density: scenes sitting in the sparsest tail of the fitted ODD space get high severity, and this module estimates how probable it is for a scene to reach a given severity level.

## Paper / Formula Provenance

Jiang, Pan, Liu, Han, Pan, Li, Pan, "Enhancing Autonomous Vehicle Safety Based on Operational Design Domain Definition, Monitoring, and Functional Degradation," IEEE TIV, Oct 2024 — Kriging-based Subset Simulation, Eqs. 16-19:

```
P_f     = P(X in F_fail) = integral I_F(X) * phi_ODD(X) dX
P_f_hat = P(F_m) * prod_{i=2}^{m} P(F_i | F_{i-1})
P(F_m)  = (sum I_F(X_i)) / N_m
```

i.e. decomposing a rare-event probability `P_f` into a product of nested conditional probabilities `P(F_i | F_{i-1})`, each estimated over the population that survived the previous (less strict) level's threshold.

**What is simplified vs. the paper:** the paper's version uses a Kriging surrogate model plus Metropolis-Hastings resampling to generate new candidate points at each level, drawn from a live driving simulator. This project has no simulator (the same no-closed-loop-simulation scope decision documented for `rss_safety.py`), so this module instead applies the *same nested-threshold decomposition* directly to a **static population** of risk scores (here, `odd_boundary.py`'s fitted copula density, converted to a severity score). Each level's threshold is set as an empirical quantile of the current surviving population, targeting a fixed conditional probability per level — the same adaptive-threshold idea as the paper, but without the Kriging surrogate or MCMC resampling machinery that a live simulator would be needed for.

## Public API

### `class SubsetSimulationResult`

```python
@dataclass
class SubsetSimulationResult:
    failure_probability: float
    level_thresholds: List[float]
    level_conditional_probabilities: List[float]
    num_levels_used: int
```
Result of a staged, nested-conditional-probability failure estimate: `failure_probability` is the final estimated `P_f`; `level_thresholds` the severity threshold used at each level; `level_conditional_probabilities` the conditional probability realized at each level (their product equals `failure_probability`); `num_levels_used` how many levels the decomposition actually used.

### `estimate_failure_probability`

```python
def estimate_failure_probability(
    population: np.ndarray,
    failure_threshold: float,
    conditional_prob_target: float = 0.1,
    max_levels: int = 10,
) -> SubsetSimulationResult
```
Estimates `P(X >= failure_threshold)` via nested conditional probabilities. `population` is a continuous array of risk-score observations (larger = more severe/less safe). `conditional_prob_target` is the target conditional probability retained per level (default `0.1`). `max_levels` is a safety cap on the number of levels, in case `failure_threshold` is never reached (e.g. it exceeds the population's maximum).

At each level, the candidate threshold is the `(1 - conditional_prob_target)` quantile of the current surviving population, so each level retains approximately `conditional_prob_target` of the prior level's mass by construction. The loop stops as soon as a candidate threshold reaches or exceeds `failure_threshold`. A **final level is always appended** after the loop, computing the exact conditional probability `P(X >= failure_threshold)` within whatever population survives every prior level — this final step is computed unconditionally regardless of why the loop stopped (threshold reached, population exhausted, or `max_levels` hit), and correctly evaluates to `0.0` when `failure_threshold` is unreachable from the population, instead of silently decaying to a small-but-nonzero product from unfinished levels. Returns a `SubsetSimulationResult`.

### `density_to_severity`

```python
def density_to_severity(density_percentiles: np.ndarray, density_value: float) -> float
```
Converts an ODD-copula density into a severity score in `[0.0, 1.0]`. Severity is `1 - percentile_rank`, where `percentile_rank` is `density_value`'s rank (via `np.searchsorted(..., side="right")`) within a sorted reference population `density_percentiles` (typically `ODDCopulaModel.density_percentiles`), divided by the population size. A scene in the sparsest tail of the fitted ODD space (lowest density) gets severity close to `1.0`; a scene in the densest, most typical region gets severity close to `0.0`.

### `estimate_odd_failure_probability`

```python
def estimate_odd_failure_probability(
    copula_model: ODDCopulaModel,
    df: pd.DataFrame,
    severity_threshold: float = 0.95,
    conditional_prob_target: float = 0.1,
) -> SubsetSimulationResult
```
Estimates the probability of landing in the extreme tail of the ODD space. Computes `odd_density` for every row of `df` over `copula_model.variables`, converts each to a severity score via `density_to_severity`, and runs `estimate_failure_probability` over the resulting severity array. `severity_threshold=0.95` means "the worst 5% of scenes by ODD density." Returns a `SubsetSimulationResult` over the dataset's severity scores.

## Key Design Decisions & Edge Cases

- **Static-population Subset Simulation, not the paper's live-simulator Kriging/MCMC version.** This is the module's central, explicitly documented simplification — see Provenance above.
- **The final level's probability is computed unconditionally, correctly handling unreachable thresholds.** This is the subtlest part of `estimate_failure_probability`: without the unconditional final step, a `failure_threshold` that the population never reaches (e.g. it exceeds the population maximum) could otherwise leave the decomposition mid-product with a small but spuriously nonzero result. By always appending one final, exact conditional-probability level regardless of loop-exit reason, the function correctly returns `0.0` in that case (verified by `test_estimate_failure_probability_threshold_above_population_max`).
- **Adaptive per-level thresholds, targeting a fixed retained fraction.** Each level's threshold is a data-driven quantile of the *currently surviving* population, not a fixed absolute value — this mirrors the paper's adaptive-threshold idea (Eqs. 17-19) without needing new samples generated at each level.
- **`max_levels` is a safety cap, not a target.** The loop can and often does exit earlier (once the candidate threshold reaches `failure_threshold`), and can also exit early simply because fewer than 2 points remain in the surviving population.
- **Severity conversion (`density_to_severity`) is a percentile-rank-based convention layered on top of the copula density**, not itself a value from the paper — it's the specific way this codebase turns a raw ODD density into the "larger = more severe" score that `estimate_failure_probability`'s formula (which assumes `P(X >= threshold)` semantics) expects.

## Dependencies

- Standard library: `os`, `dataclasses`, `typing`
- External: `numpy`, `pandas`
- Internal: `src.odd.odd_boundary` (`DEFAULT_ODD_VARIABLES`, `ODDCopulaModel`, `fit_odd_copula`, `odd_density`)

## Usage Example

```python
import pandas as pd
from src.odd.odd_boundary import fit_odd_copula, DEFAULT_ODD_VARIABLES
from src.odd.risk_estimation import estimate_odd_failure_probability

df = pd.read_csv("outputs/final_features.csv")
copula = fit_odd_copula(df, DEFAULT_ODD_VARIABLES)

result = estimate_odd_failure_probability(copula, df, severity_threshold=0.95)
print(f"P(severity >= 0.95) ~= {result.failure_probability:.6f}")
print("Per-level thresholds:", result.level_thresholds)
print("Per-level conditional probabilities:", result.level_conditional_probabilities)
```

## Running Standalone

```bash
python -m src.odd.risk_estimation
```
Loads `outputs/final_features.csv` (`FEATURES_CSV`), fits an ODD copula over `DEFAULT_ODD_VARIABLES`, then for each of `severity_threshold` in `(0.90, 0.95, 0.99)` prints the estimated failure probability, the number of levels used, and the per-level conditional probabilities.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/odd/test_risk_estimation.py` covers: that the estimator recovers a known analytical quantile for a uniform-distribution population (the key correctness check for the staged decomposition), that `failure_probability` equals the product of `level_conditional_probabilities`, that a threshold above the population's maximum yields `0.0`, that a threshold at/below the population's minimum yields a probability `> 0.9`, and that `density_to_severity` behaves correctly at its extremes and is monotonically decreasing in density.
