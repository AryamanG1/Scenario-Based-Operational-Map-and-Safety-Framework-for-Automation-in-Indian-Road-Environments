# `ecofusion_gate.py`

**Stage:** 7
**Package:** `src.decision.ecofusion_gate`

## Purpose

This module implements the Stage 7 "adaptive compute" component of the Decision System: given a frame, it decides *which* of the pipeline's perception branches (SegNet segmentation, YOLO detection, raw image statistics) are actually worth running, trading off accuracy loss against compute cost, rather than always running every branch on every frame. It provides both an offline "Knowledge Gate" (uses already-computed reliabilities, useful for analysis/tuning) and a "Deep Gate" (a small regressor that predicts reliability from cheap image statistics alone, so gating can happen *before* paying for expensive branches — the actual deployment-time use case).

It exists in the pipeline as the compute-efficiency counterpart to `multi_stream_fusion.py`: where fusion combines the outputs of streams that already ran, EcoFusion decides in advance which of those streams are worth running at all for a given frame, then only runs the selected ones.

## Paper / Formula Provenance

Implements **Malawade, Mortlock, Al Faruque, "EcoFusion: Energy-Aware Adaptive Sensor Fusion for Efficient Autonomous Vehicle Perception," ACM/IEEE DAC 2022** — specifically Eqs. 6-9 and Algorithm 1, as stated in the module docstring:

- **Energy model** `E(phi, X) = P(phi, X) * t(phi, X)` (Eq. 6)
- **Candidate filtering** `Phi* = {phi in Phi : Lf(phi) - Lf(phi') <= gamma}` (Eq. 7) — the module docstring notes it uses "the prose-consistent inequality — the paper's own printed equation duplicates a term and is treated here as a transcription artifact, per direct reading."
- **Joint loss** `L_joint(phi, lambda_E) = (1-lambda_E)*Lf(phi) + lambda_E*E(phi)` (Eq. 8)
- **Selection** `phi* = argmin_{phi in Phi*} L_joint(phi, lambda_E)` (Eq. 9)
- **Algorithm 1**: stem → gate(Lf(Phi)) → filter(Phi*) → joint-loss argmin → run only the selected branches → fuse.

**This is an adaptation of the paper, and the adaptation is deliberate and documented in-code, not an oversight.** The paper's algorithm was designed for a vehicle with multiple physical sensors (camera/LiDAR/radar). This project has exactly one camera, so the three "sensor streams" gated here are actually `multi_stream_fusion.py`'s three *derived* streams from that single camera — the SegNet mask, YOLO detections, and raw-image statistics (`STREAMS`, imported from `multi_stream_fusion.py`) — standing in for the paper's physically distinct sensors. `Phi` is therefore the 7 non-empty subsets of these 3 streams, not of physical sensors.

The paper defines four gating strategies: Knowledge Gating, Deep Gating, Attention Gating, and a ground-truth Loss-Based oracle. This module implements only the first two — Attention Gating and the Loss-Based oracle both require either a live simulator or labeled ground truth this project doesn't have.

## Public API

```python
RANDOM_STATE = 42
STEM_FEATURES = ("brightness", "visibility", "wetness")
DEFAULT_GAMMA = 0.5          # matches Malawade et al.'s own experimental choice
DEFAULT_LAMBDA_E = 0.1

CONFIGURATION_SPACE: List[FrozenSet[str]]  # all 7 non-empty subsets of STREAMS
```

```python
def compute_stem_features(image: np.ndarray) -> Dict[str, float]
```
Computes the cheap "stem" features available before any expensive branch runs: `brightness` (mean pixel value), `visibility` (Laplacian variance — image sharpness), `wetness` (pixel std dev) — all via plain OpenCV/NumPy on the raw pixels.

```python
def profile_branch_latencies(
    image: np.ndarray, segnet_model: torch.nn.Module, yolo_model, num_trials: int = 3
) -> Dict[str, float]
```
Measures each branch's average wall-clock latency in seconds, timing `num_trials` repeated runs of SegNet, YOLO detection (`run_detection`), and `compute_stem_features()`. Returns `{"segnet_mask": ..., "yolo_detections": ..., "image_stats": ...}`; the `image_stats` latency is floored at `1e-6` "to avoid a literal-zero cost branch."

```python
def energy_cost(config: FrozenSet[str], branch_latencies: Dict[str, float]) -> float
```
`E(phi, X)` (Eq. 6, with energy replaced by latency): sum of `branch_latencies[stream]` for every stream in `config`.

```python
def knowledge_gate_loss(config: FrozenSet[str], reliabilities: Dict[str, float]) -> float
```
Knowledge-Gating `Lf(phi)`: the fraction of total reliability the config *omits*, `1 - sum_{s in phi} R_s / sum_{s in STREAMS} R_s`. Returns `0.0` if total reliability is `<= 0`. Range `[0.0, 1.0]`; `0.0` for the full configuration.

```python
def fit_deep_gate(fused_df: pd.DataFrame) -> RandomForestRegressor
```
Trains a multi-output `RandomForestRegressor` (100 estimators, `random_state=RANDOM_STATE`) to predict `multi_stream_fusion.py`'s per-stream `reliability_<stream>` columns from `STEM_FEATURES` alone. Expects `fused_df` to be the output of `multi_stream_fusion.fuse_dataframe()`.

```python
def deep_gate_predict_reliabilities(
    model: RandomForestRegressor, stem_features: Dict[str, float]
) -> Dict[str, float]
```
Queries a fitted Deep Gate for one frame's predicted per-stream reliability, given only its stem features. Output values are clipped to `[0.0, 1.0]`.

```python
def filter_pareto_candidates(
    losses: Dict[FrozenSet[str], float], gamma: float = DEFAULT_GAMMA
) -> List[FrozenSet[str]]
```
Filters `Phi` down to `Phi*` (Eq. 7): every config whose loss is within `gamma` of the minimum loss. The docstring notes `phi'` (the minimum-loss config) is "necessarily the full configuration, since `Lf` is monotonically non-increasing as streams are added." Always includes at least the argmin.

```python
def select_configuration(
    candidates: List[FrozenSet[str]],
    losses: Dict[FrozenSet[str], float],
    branch_latencies: Dict[str, float],
    lambda_e: float = DEFAULT_LAMBDA_E,
) -> Tuple[FrozenSet[str], float]
```
Selects `phi* = argmin L_joint(phi, lambda_e)` over `candidates` (Eqs. 8-9), where `L_joint = (1 - lambda_e) * losses[config] + lambda_e * energy_cost(config, branch_latencies)`. `lambda_e` is the accuracy/speed trade-off weight: `0.0` = pure accuracy, `1.0` = pure speed. Returns `(best_config, best_joint_loss)`.

```python
@dataclass
class EcoFusionDecision:
    selected_config: FrozenSet[str]
    joint_loss: float
    candidates: List[FrozenSet[str]]
    losses: Dict[FrozenSet[str], float]
```
The outcome of one full EcoFusion gating decision.

```python
def run_ecofusion_gate(
    stem_features: Dict[str, float],
    deep_gate: RandomForestRegressor,
    branch_latencies: Dict[str, float],
    gamma: float = DEFAULT_GAMMA,
    lambda_e: float = DEFAULT_LAMBDA_E,
) -> EcoFusionDecision
```
Runs the full Algorithm 1 pipeline using the Deep Gate: predicts reliabilities from `stem_features`, computes `Lf` for every config in `CONFIGURATION_SPACE`, filters to Pareto candidates, selects the joint-loss-minimizing config, and returns an `EcoFusionDecision`. This is the module's primary deployment-time entry point.

```python
def run_selected_branches(
    image: np.ndarray,
    config: FrozenSet[str],
    segnet_model: Optional[torch.nn.Module],
    yolo_model,
) -> Dict[str, object]
```
Executes only the branches named in `config`, skipping the rest — this is where EcoFusion's compute savings are actually realized (e.g. a config omitting `"segnet_mask"` never pays for a SegNet forward pass). Returns a dict with whichever of `"mask"`, `"detections"`, `"stem_features"` keys correspond to branches that were run (`segnet_model` can be `None` if `"segnet_mask"` isn't in `config`).

## Key Design Decisions & Edge Cases

- **"Energy" → "latency."** The paper's energy model `E(phi, X) = P(phi, X) * t(phi, X)` requires power-draw instrumentation this project's machine doesn't have (no discrete GPU, no power telemetry). The module substitutes measured wall-clock latency per branch as the cost signal, documented explicitly in both the module docstring and `energy_cost()`'s docstring ("Eq. 6, energy→latency").
- **Camera-derived streams stand in for physical sensors.** `Phi`, the fusion-configuration space, is the 7 non-empty subsets of `multi_stream_fusion.STREAMS` (SegNet mask / YOLO detections / image stats) — all derived from one camera — rather than subsets of physically distinct sensors as in the original paper.
- **A documented "transcription artifact."** The module docstring explicitly flags that Eq. 7 as printed in the paper duplicates a term, and that this implementation uses "the prose-consistent inequality" instead — a deliberate, cited deviation rather than a silent bug.
- **Knowledge Gate vs. Deep Gate have different valid use cases.** Knowledge Gating (`knowledge_gate_loss`) requires reliabilities that are only known *after* running the branches, so it's explicitly documented as useful "for offline analysis/lambda_E tuning, not real deployment-time gating." Deep Gating is the one actually usable for real deployment-time decisions, since it predicts reliability from cheap stem features alone.
- **Only two of the paper's four gating strategies are implemented** — Attention Gating and the ground-truth Loss-Based oracle are both explicitly out of scope since they need a live simulator or labeled ground truth this project doesn't have.
- **`DEFAULT_GAMMA = 0.5` matches the paper's own experimental choice** (per the inline comment); `DEFAULT_LAMBDA_E = 0.1` is this project's own default trade-off setting.
- **Deep Gate output is clipped to `[0, 1]`** since a `RandomForestRegressor` can in principle output values slightly outside the reliability range's natural bounds.

## Dependencies

**Internal:** `src.perception.multi_stream_fusion` (`STREAMS`, `calibrate_references`, `compute_stream_reliabilities`) at module scope; `src.perception.feature_extraction.run_detection` (imported locally inside `profile_branch_latencies` and `run_selected_branches`). The `if __name__ == "__main__":` block additionally imports `src.common.paths` (`DATA_DIR`, `ECOFUSION_DEEP_GATE_PATH`, `FEATURES_CSV`, `SEGNET_CHECKPOINT`, `YOLO_WEIGHTS`, `ensure_output_dirs`), `src.perception.data_pipeline.load_and_clean_dataset`, and `src.perception.segnet_model.load_segnet`.

**External:** `itertools`, `time`, `os` (stdlib); `cv2`, `joblib`, `numpy`, `pandas`, `torch`, `sklearn.ensemble.RandomForestRegressor`, and (in the standalone block) `ultralytics.YOLO`.

## Usage Example

```python
from src.decision.ecofusion_gate import (
    compute_stem_features, fit_deep_gate, run_ecofusion_gate, run_selected_branches,
)

# deep_gate: fitted via fit_deep_gate(fused_features_df), loaded from disk normally
stem = compute_stem_features(image)  # cheap, no SegNet/YOLO needed
decision = run_ecofusion_gate(stem, deep_gate, branch_latencies, lambda_e=0.1)
print(sorted(decision.selected_config))   # e.g. ['segnet_mask', 'yolo_detections']

# Only pay for the branches EcoFusion actually selected:
outputs = run_selected_branches(image, decision.selected_config, segnet_model, yolo_model)
```

## Running Standalone

```bash
python -m src.decision.ecofusion_gate
```

Loads `final_features.csv`, calibrates `multi_stream_fusion` reliability references, computes per-stream reliabilities for every row, trains and saves a Deep Gate (`models/ecofusion_deep_gate.pkl`) via `fit_deep_gate()`, profiles branch latencies on the first dataset image, then runs `run_ecofusion_gate()` across `lambda_e in (0.0, 0.1, 0.5, 0.9)`, printing the selected configuration, joint loss, and candidate count for each.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/decision/test_ecofusion_gate.py` covers: `CONFIGURATION_SPACE` has exactly 7 non-empty subsets (2³-1) and never includes the empty set; `knowledge_gate_loss` is `0.0` for the full configuration and higher when a more-reliable stream is omitted versus a less-reliable one; `energy_cost` correctly sums active-branch latencies; `filter_pareto_candidates` always includes the loss-optimal config and excludes configs far outside the `gamma` tolerance; `select_configuration` prefers the lowest-loss config at `lambda_e=0.0` and the lowest-cost config at `lambda_e=1.0`.
