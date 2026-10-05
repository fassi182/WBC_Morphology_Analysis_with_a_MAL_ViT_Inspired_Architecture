# Project verification record

Audit date: **2026-10-05**. These checks apply to the local source and checkpoint
files at the time of this audit. They do not certify every possible environment
or establish clinical validity.

## Architecture and checkpoint findings

The automatic pipeline is:

```text
image -> 11 categorical attribute distributions -> 31 probabilities -> 5 WBC logits
```

The WBC head receives only the attribute probabilities. Its predictions have no
direct connection to image, patch, or register features. Register tokens still
participate in the image-to-attribute transformer.

The original image checkpoint and independent attribute classifier were composed
using the shared strict loader. The original image model's old WBC head is
excluded. The independent classifier was trained on ground-truth one-hot
attributes; its current image-based input consists of predicted probabilities.
Neither this audit nor the export jointly retrained those original stages.

`outputs/wbc_inference.pth` packages the composed weights and label metadata in
one file. Every tensor exactly matched the composed original weights. Exporting
does not change those weights and refuses to overwrite an existing file.

This is an independent **MAL-ViT-inspired adaptation**, not an exact reproduction
of the paper. The README and GUI use that description.

## Completed checks

| Check | Outcome |
|---|---|
| Automated suite: `python -m pytest -q` | **34 passed** |
| Python syntax | All 64 project Python source files parsed successfully |
| Architecture regressions | Attribute-only WBC inputs, gradient flow, intervention encoding, and checkpoint compatibility passed |
| Legacy checkpoint metadata | Mismatched attribute names, category encoders, and WBC labels rejected when supplied by the image checkpoint |
| Attribute diagnostic helper | Uses evaluation mode and the supplied model's device, independent of the configured default device |
| Application API | File paths, PIL images, and encoded image bytes returned matching, JSON-serializable results |
| Checkpoint export | Exact tensor preservation; export restored after source checkpoint deletion; overwrite protection passed |
| Invalid inputs | Corrupt image and non-finite checkpoint rejection passed |
| Shared model explanations | Input-gradient explanations preserved model parameter gradients |
| Streamlit AppTest | Empty upload, invalid image, explicit checkpoint selection, full analysis, and all 33 explanation images passed |
| Portable trained inference | Ran in a temporary directory with only the documented runtime files, one image, and exported trained weights |
| Portable dependencies | Prediction imported neither pandas, Streamlit, nor training modules; dataset and original checkpoint pair were absent |
| Standalone example | `examples/predict_in_app.py` produced the same probabilities as the portable API |
| Real-data training check | One training batch and one validation batch completed with finite losses; no trained checkpoint was saved |
| Dataset paths | All 6,169 training, 1,030 validation, and 3,099 test image paths existed; no duplicate paths within or across splits |
| Full exported-model evaluation | All 3,099 test images evaluated; classification metrics, confusion matrix, and attribute accuracies matched the previous composed-model report |

The GUI regression uses a temporary model and synthetic image to test application
behavior. Trained-model quality is assessed by the separate full dataset evaluation.
AppTest exercises Streamlit's application execution; it is not a visual browser
layout inspection. Path separation does not rule out duplicate image content or
patient-level overlap.

The final follow-up review also checked help commands for training, evaluation,
inference, export, and the integration example, and compared the original composed
model with the exported trained model again. Both attribute and WBC probabilities
matched exactly on the sample image. The metadata and diagnostic fixes do not
change trained weights or the standard evaluation pipeline; the full dataset
result below is the previously completed evaluation, not a repeated training run.

## Recorded model performance

Command:

```bash
python test.py --checkpoint outputs/wbc_inference.pth --output outputs/final_audit_test_results.json
```

The [full evaluation report](../outputs/final_audit_test_results.json) records
the checkpoint source, probability representation, sample count, losses,
confusion matrix, per-class metrics, and attribute accuracies.

| Metric | Result |
|---|---:|
| Test images | 3,099 |
| WBC accuracy | 97.64% |
| Macro precision | 96.93% |
| Macro recall | 96.77% |
| Macro F1 | 96.77% |
| Total loss | 0.352282 |

These results describe this local split. Attribute quality varies: nucleus shape
accuracy is 64.73% and cell size accuracy is 75.22%. WBC accuracy does not establish
equivalent accuracy for every attribute or performance on external datasets.

## Environment and reproducibility limits

Checks ran on Windows, Python 3.11, and CPU with PyTorch 2.2.2, torchvision 0.17.2,
NumPy 1.26.4, Pillow 10.2.0, pandas 2.3.3, matplotlib 3.10.8, scikit-learn 1.9.1,
tqdm 4.67.1, Streamlit 1.53.1, and pytest 9.0.2. All directly declared project
dependency versions satisfied the requirement files.

The local `.venv` reused system packages and installed a compatible pandas version
locally. The base environment contained unrelated dependency conflicts. This was
not a clean-install verification or a claim that the global environment passes
`pip check`. For another machine, use the fresh virtual environment instructions
in [USAGE.md](USAGE.md). Requirements specify version ranges, not a lockfile.

No complete training run, CUDA execution, external dataset evaluation, robustness
reproduction, clinical validation, or production concurrency/load test was
performed in this audit. Mermaid source diagrams are included in the README;
visual rendering depends on the Markdown viewer.

## Using the completed project

Use the [GUI and integration guide](USAGE.md) for exact commands and minimal
deployment files. Existing trained weights support inference without retraining.
The exported `.pth` still requires the model source code and runtime dependencies.
When sharing the project, include compatible trained weights or clearly provide
their download location; installing dependencies does not download them.
