# Potato Leaf Disease Classifier

An image classifier for potato leaf diseases namely Early Blight, Late Blight, Healthy with a web interface for testing it on new leaf photos.

![Web-Interface](/serve/templates/Web_Interface_1.png)

![Web-Interface](/serve/templates/Web_Interface_2.png)

![Web-Interface](/serve/templates/Web_Interface_3.png)

## Data source

PlantVillage Dataset (spMohanty/PlantVillage-Dataset), raw/color, potato classes. Exact source commit recorded in data/SOURCE_COMMIT.txt.

<https://github.com/spMohanty/PlantVillage-Dataset>

## Model Selected

The system uses MobileNetV3-Large as the base CNN architecture, with edge computing in mind having seen a trajectory for mobile phone inferencing.

Unlike heavier architectures (ResNet, VGG, EfficientNet-B4+) that assume server-side inference, MobileNetV3-Large is built for constrained compute. This makes on-device inference directly on a farmer's smartphone possible, without needing a persistent connection to a cloud backend.

---

## Result Summary

| Metric               | Value                                   |
| -------------------- | --------------------------------------- |
| CV macro-F1          | **0.992 ± 0.007**                       |
| CV healthy recall    | **0.994 ± 0.014**                       |
| Architecture         | MobileNetV3-Large (ImageNet pretrained) |
| Preprocessing        | Grey-world normalisation, 224×224       |
| Confidence threshold | 0.90 (below this the model abstains)    |
| Training images      | 2,152                                   |

Metrics are from 5-fold cross-validation. There is no separate held-out test set —
the final model is trained on all data under the configuration that CV selected.

---

## Reproduction

Run the notebooks in order. Each one writes artefacts the next one reads.

| Notebook                 | Does                                                                                              | Produces                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `01_fetching_data.ipynb` | Pulls the potato subset from spMohanty/PlantVillage-Dataset                                       | `data/potato_raw/`, `SOURCE_COMMIT.txt`                             |
| `02_data_audit.ipynb`    | Integrity, duplicate detection (SHA256 + dihedral pHash), background probe, fixed fold assignment | `results/audit/manifest.csv`, `FINDINGS.md`                         |
| `03_data_pipeline.ipynb` | Transforms, datasets, group-aware loaders, leak checks                                            | `results/classes.json`                                              |
| `04_Model.ipynb`         | MobileNetV3 builder, two-stage freeze control, BatchNorm guard                                    | —                                                                   |
| `05_Train.ipynb`         | 5-fold training, two stages, early stopping on val loss                                           | `results/runs/<run>/fold_*/best.pt`, `history.csv`, `oof_preds.csv` |
| `06_eval.ipynb`          | Confusion matrices, per-class metrics, confidence threshold sweep                                 | fold + pooled metrics                                               |
| `07_gradcam.ipynb`       | Attention maps — lesion vs. background verification                                               | Grad-CAM figures                                                    |
| `08_ablation.ipynb`      | baseline / aggressive / grey-world comparison                                                     | `results/runs/greyworld/summary.csv`                                |
| `09_final.ipynb`         | Retrains on all data with the winning config                                                      | `results/final/potato_mobilenetv3_final.pt`, `model_card.json`      |

Supporting modules live in `scripts/` and are imported by the notebooks:
`config.py` (all constants : single source of truth), `data.py`, `model.py`,
`train.py`, `eval.py`.

## Running the web app locally

```bash
# 1. Install (CPU-only torch — much smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Start the server
cd serve
python app.py

## 3.  Running
Open <http://127.0.0.1:5000>.(local host)
Upload a leaf photo : the page returns the predicted class, a model score, and a
per-class breakdown.
Below 0.90 confidence it declines to call the result rather than guessing.
```

Every submission is saved to `results/uploads/` with a row in `log.csv`. This is
deliberate, so real-world performance can be measured later against the lab-image
training set.

### Limitations

- Trained entirely on laboratory images : uniform background, controlled lighting. Performance on field photos from a phone is unverified; every upload is logged specifically to measure this gap later.
- 152 healthy images against 1,000 per disease class, so treat single-fold healthy-recall figures with caution.
- No independent held-out test set : reported metrics are cross-validated.
- Displayed percentage is a relative model score, not a calibrated probability.
- Not a substitute for agronomic advice.
