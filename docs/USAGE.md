# Use the GUI or integrate the trained model

**Retraining is not required for prediction.** You need the Python model code,
its dependencies, and compatible trained weights. The WBCAtt training dataset
is only needed for training or dataset evaluation.

The pipeline is an independent MAL-ViT-inspired adaptation:

```text
WBC image -> 11 morphology probability distributions -> WBC type
```

## 1. Choose a setup

| Goal | Dependencies | Trained weights |
|---|---|---|
| Streamlit GUI | `requirements.txt` | Original checkpoint pair, complete concept checkpoint, or explicit exported checkpoint |
| Prediction in another Python app | `requirements-inference.txt` | Same choices |
| Training and automated tests | `requirements-dev.txt` | Existing weights optional for training from scratch |

From the repository root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate it with `source .venv/bin/activate` instead. If you
cannot activate PowerShell scripts, use `.\.venv\Scripts\python.exe` in place
of `python` in the commands below.

For an inference-only environment, install `requirements-inference.txt` instead.
This does not require pandas, Streamlit, the training modules, or the dataset.

## 2. Supply the trained weights

Default loading checks for:

1. `outputs/checkpoints/best_concept_model.pth`, a complete concept model.
2. If that file does not exist, the pair
   `outputs/checkpoints/best_model.pth` and
   `outputs/checkpoints/best_attribute_wbc.pth`.

The second option uses the first file's trained image-to-attribute layers and
the second file's trained attribute-to-WBC head. The first file's old WBC head
is excluded. A message describing this composition is informational; it does
not mean you must train again.

If no trained weights were supplied with your checkout, obtain them from the
project maintainer or train them. Installing Python dependencies does not
download trained weights. An incompatible checkpoint raises an error.

### Optional: export a single model file

From a checkout with the trained weights available:

```bash
python export_model.py --output outputs/wbc_inference.pth
```

This packages both stages and their label metadata into **one checkpoint**.
It does not optimize, fine-tune, or otherwise change the weights. It excludes
optimizer state and records which checkpoints supplied the weights. If the
output already exists, use it or choose a new filename; the command will not
overwrite it.

You can select the source weights explicitly:

```bash
python export_model.py --checkpoint outputs/checkpoints/best_model.pth --attribute-checkpoint outputs/checkpoints/best_attribute_wbc.pth --output outputs/wbc_export_v2.pth
```

The exported file is a PyTorch checkpoint, not a standalone executable or ONNX
model. It still requires the accompanying Python model code and inference
dependencies. It is not evidence of a new training run.

## 3. Run the graphical interface

With the default checkpoint files:

```bash
python -m streamlit run app.py
```

Or use the single exported checkpoint:

```bash
python -m streamlit run app.py -- --checkpoint outputs/wbc_inference.pth
```

The double `--` separates Streamlit's arguments from the app's arguments.
For a relocated legacy pair, pass both `--checkpoint` and
`--attribute-checkpoint` after that separator.

1. Open the local URL printed by Streamlit, normally `http://localhost:8501`.
2. Upload a JPG, JPEG, or PNG image of a WBC.
3. Click **Analyze WBC**.
4. Inspect the eleven predicted attributes. Expand the probability panel to see
   the full distributions supplied to the WBC classifier.
5. Read the WBC type below the attributes.
6. Inspect the per-attribute attention, input-gradient, and combined maps.

The application preprocesses the image automatically. The maps describe
attribute predictions, not direct explanations of the final WBC decision.
Generating eleven explanations takes longer than prediction alone, especially
on CPU. Restart Streamlit after replacing a checkpoint so its cached model is
reloaded. The GUI does not train the model or offer attribute editing.

## 4. Use the model in another Python application

Clone or copy this repository alongside your application and keep its directory
structure intact. This repository is source code, not currently an installable
Python package. Add its root to `PYTHONPATH` or to `sys.path` before importing it.
The source uses top-level module names such as `config`, `models`, and `data`;
if your application already uses those names, run this predictor in a separate
Python backend process/environment to avoid import collisions.

```python
from pathlib import Path
import sys

# Replace this with the actual location of this repository.
repository = Path("/absolute/path/to/wbc-morphology-analysis-mal-vit")
sys.path.insert(0, str(repository))

from wbc_predictor import WBCPredictor

# Create once when your application starts, then reuse for every image.
predictor = WBCPredictor(
    checkpoint_path=repository / "outputs" / "wbc_inference.pth",
    device="cpu",  # Use "cuda" if supported, or omit for automatic selection.
)

result = predictor.predict("/absolute/path/to/cell.jpg")
print(result["wbc"])
print(result["wbc_confidence"])
print(result["attributes"])
```

If using the original pair from another location:

```python
predictor = WBCPredictor(
    checkpoint_path="/models/best_model.pth",
    attribute_checkpoint_path="/models/best_attribute_wbc.pth",
    device="cpu",
)
```

### Uploaded images and output fields

The same `predict()` method accepts a file path, a PIL image, or encoded image
bytes. Pass the original image; resizing and normalization are already handled.

```python
from pathlib import Path
import json
from PIL import Image

result = predictor.predict(Path("cell.jpg").read_bytes())  # e.g. upload bytes

with Image.open("cell.jpg") as image:
    result = predictor.predict(image)

payload = json.dumps(result)  # Can be returned by your own web API.
```

| Field | Meaning |
|---|---|
| `wbc` | Predicted lowercase WBC label |
| `wbc_index` | Category index from `data/encoders.py` |
| `wbc_confidence` | Softmax probability of the selected WBC class; not a calibrated guarantee |
| `wbc_probabilities` | Probabilities for all five WBC classes |
| `attributes` | Highest-probability category for each of the eleven attributes |
| `attribute_probabilities` | All 31 category probabilities, grouped by attribute |
| `checkpoint_source` | Which native, composed, or exported checkpoint was loaded |

The method performs inference only and does not generate explanation maps or
modify trained weights. It raises errors for unreadable images or incompatible
weights; your application should handle those errors in its upload/request flow.

A runnable version of this pattern is included:

```bash
python examples/predict_in_app.py --image images/BA_1223.jpg --checkpoint outputs/wbc_inference.pth
```

It prints the complete result as JSON and supports `--device`. You can launch
the example by absolute path from another working directory; use absolute paths
for its image and checkpoint arguments in that case.

### Files needed for a small deployment

Keeping the whole source checkout is simplest. A smaller prediction-only copy
needs `wbc_predictor.py`, `inference.py`, `config.py`, `requirements-inference.txt`,
the `models/` directory, `data/__init__.py`, `data/encoders.py`,
`data/transforms.py`, `utils/__init__.py`, `utils/model_loading.py`, and the exported
checkpoint. Keep the same relative directory structure. It does not need
`datasets/`, `training/`, `tests/`, or the original checkpoint pair.

## 5. Retraining is a separate, optional task

For existing trained weights, prediction requires no training command. To adapt
both stages using your configured dataset, run:

```bash
python train.py --initialize-from-existing
```

This is a new optimization run initialized from existing weights, not an exact
resume of an old training session. Validate the resulting checkpoint before
replacing the one used by your application. Changes to labels or attribute
categories also require corresponding code/checkpoint changes.

## Troubleshooting

| Symptom | Check |
|---|---|
| Checkpoint not found | Supply the original pair or pass the explicit exported checkpoint path. |
| Cannot import `wbc_predictor` | Put this repository root on `PYTHONPATH`/`sys.path` and use the correct environment. |
| Missing model keys or shape mismatch | Confirm that the checkpoint matches this architecture; do not hide the problem with `strict=False`. |
| pandas/Streamlit conflict | Install the project requirements in a virtual environment; the GUI requirements constrain pandas below version 3. |
| Unsupported image | Pass a readable JPG/PNG path, PIL image, or encoded bytes, not a normalized tensor or arbitrary array. |
| GUI seems slow | Explanation generation uses eleven gradient passes; use `WBCPredictor` or the CLI for prediction without maps. |
| CUDA unavailable | Use `device="cpu"`; default loading automatically selects CPU when CUDA is unavailable. |

For architecture, recorded metrics, and known limitations, see the [README](../README.md).
