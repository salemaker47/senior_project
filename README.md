# Senior Project — Brain Tumor Analysis

Multi-dataset, multi-task medical imaging pipeline for brain MRI. The project
builds end-to-end cross-validated pipelines for two tasks and quantifies the
cost of using imperfect segmentation masks as input to a classifier.

---

## Tasks

| task | input | output |
|---|---|---|
| **Segmentation** | grayscale brain MRI (256×256) | binary tumor mask |
| **Classification** | tumor patch cropped from MRI via a mask | tumor class (meningioma / glioma / pituitary) |

Classification is evaluated twice per experiment:

- **Eval A** — patch cropped with the ground-truth mask
- **Eval B** — patch cropped with a predicted mask from the best segmentation model

The gap (Eval A macro F1 − Eval B macro F1, mean ± std across 5 folds)
quantifies how much classification degrades when the segmentation is imperfect.

---

## Datasets

| dataset | size | classes | source |
|---|---|---|---|
| `figshare` | 3,064 images / 233 patients | meningioma, glioma, pituitary | KaggleHub `ashkhagan/figshare-brain-tumor-dataset` |
| `brats2020` | ~57k slices / 369 patients | glioma (segmentation only) | KaggleHub `awsaf49/brats2020-training-data` |

Adding a new dataset requires only dropping raw files into `data/<dataset>/raw/`
and running NB01 + NB02 with `DATASET = "<dataset>"`. No source-code changes needed.

---

## Cross-validation setup

All experiments use **5-fold cross-validation**. Two split schemes are supported:

- `patient_level` — `StratifiedGroupKFold` on `patient_id`. Used for the
  project's final results. Guarantees no patient appears in more than one split.
- `image_level` — plain `KFold` on image rows. Used to reproduce the FigShare
  reference benchmark's numbers (this scheme leaks patients across folds).

Because classification and segmentation use the same fold CSVs, fold *k*'s
test set is identical for both tasks. This means Eval B can look up predicted
masks for fold *k*'s test images without any out-of-fold lookup.

---

## Folder structure

```
Senior_Project/
├── data/<dataset>/
│   ├── raw/                            original source files (.mat / .h5)
│   ├── processed/
│   │   ├── images/<id>.png             grayscale uint8 256×256
│   │   ├── masks/<id>.png              binary uint8 256×256 (0 / 255)
│   │   ├── metadata.csv                one row per image
│   │   └── preprocessing_config.json
│   └── splits/<scheme>/
│       ├── cv_split_config.json
│       ├── cv_fold_summary.csv
│       └── folds/fold_X_{train,val,test}.csv
│
├── outputs/
│   ├── checkpoints/<task>/<dataset>/<exp>/fold_X/
│   │   ├── best.ckpt                   Lightning checkpoint
│   │   ├── best_model.pt               plain PyTorch state-dict for inference
│   │   └── experiment_config.json      full EXPERIMENT dict + repro metadata
│   ├── logs/<task>/<dataset>/<exp>/fold_X/version_0/metrics.csv
│   ├── predictions/segmentation/<dataset>/<seg_exp>/
│   │   ├── prediction_manifest.json    experiment-level manifest
│   │   └── fold_X/
│   │       ├── manifest.json           per-fold manifest with SHA-256 hashes
│   │       └── <image_id>.png          predicted binary mask
│   ├── tables/<task>/<dataset>/<exp>/
│   │   ├── cv_results.csv              one row per fold
│   │   ├── cv_summary.csv              mean ± std across folds
│   │   ├── cv_summary_enriched.csv     median, IQR, 95% CI per metric
│   │   ├── cv_by_class.csv             per-(fold, class) metrics
│   │   ├── cv_class_summary.csv        per-class mean ± std across folds
│   │   └── cv_per_image.csv            ~3k rows, one per test image
│   └── figures/<task>/<dataset>/<exp>/fold_X/
│
├── notebooks/
│   ├── setups/      01_data_preparation, 02_split
│   ├── segmentation/ 03_train, 04_data_vis, 05_test
│   └── classification/ 06_cls_data, 07_train_cls, 08_test_cls
│
└── src/             see §Source modules below
```

Artifacts under `data/` and `outputs/` live in Google Drive and are not
version-controlled. Code in `src/` and `notebooks/` lives here on GitHub.

---

## Source modules

### Shared

| file | responsibility |
|---|---|
| `file_utils.py` | canonical path builders (`experiment_paths`, `seg_predictions_dir`, …), JSON helpers, SHA-256 file hashing, prediction-manifest readers and verifier |
| `notebook_setup.py` | Colab bootstrap: `setup_environment` (mount Drive + git pull), `copy_to_local` (Drive → local SSD), `sync_outputs_to_drive` (local SSD → Drive at end of run) |
| `preprocess_utils.py` | raw-file converters for figshare (.mat) and brats2020 (.h5) → 256×256 PNGs + metadata records; `get_dataset_converter` dispatch |
| `vis_utils.py` | generic image-display helpers (NB01) |
| `data_utils.py` | metadata loading/validation, patient-level and image-level fold splitting, leakage verification, fold CSV persistence |
| `eval_utils.py` | `enriched_aggregate` (cross-fold mean/std/median/IQR/CI table, Enhancement B); `build_fold_summary` |
| `optimizers.py` | string-keyed optimizer registry (adam, adamw, sgd, rmsprop) and scheduler registry (reduce_on_plateau, cosine, cosine_warm_restarts, step, multistep, exponential) |
| `train_utils.py` | `set_global_seed`, `gather_repro_metadata` (git + GPU stamp), `TrainingTimingCallback`, `EpochSummaryPrinter`, `build_callbacks`, `build_trainer`, `export_plain_state_dict`, `strip_model_prefix` |

### Segmentation (`sg_` prefix)

| file | responsibility |
|---|---|
| `sg_data_utils.py` | `BrainTumorDataset`, augmentation pipelines (`build_train_transform`, `build_eval_transform`), `build_dataloaders`, `build_test_loader` |
| `sg_metrics.py` | `binarize_logits`, `dice_score`, `iou_score`, SMP-backed `get_smp_stats` / `micro_dice_from_stats` / `micro_iou_from_stats`, full per-image metric suite (`compute_per_image_metrics_from_logits`), metric-kind registry (`get_metric_kind_pairs`) |
| `sg_losses.py` | loss factory: bce, dice, focal, lovasz, dice_bce, dice_focal, combo (`CombinedLoss`) |
| `sg_models.py` | SMP model factory: U-Net/ResNet34, U-Net/ResNet50, U-Net++/EfficientNet-B4, U-Net++/ResNet34, Linknet/ResNet34, MA-Net/ResNet34 |
| `sg_lightning_module.py` | `BrainTumorSegModule` — training/validation/test steps, micro-pooled metric accumulation via shared `_flush_epoch_buffers`, registry-driven optimizer + scheduler |
| `sg_eval_utils.py` | `summarize_fold_results` (per-fold tables), `aggregate_cv_results` (cross-fold tables), `aggregate_cv_per_patient`, `aggregate_cv_training_summary` |
| `sg_test_utils.py` | `load_model_from_pt`, `load_model_from_ckpt`, `predict_mask` (single-image), `evaluate_fold` (batched inference + PNG saving + per-fold manifest), `write_experiment_manifest` |
| `sg_vis_utils.py` | `show_triplet`, `show_overlay_triplet`, `show_image_gt_pred_overlay` |

### Classification (`cls_` prefix) — Phase 2, in progress

`cls_data_utils`, `cls_models`, `cls_losses`, `cls_metrics`, `cls_lightning_module`, `cls_test_utils`, `cls_eval_utils`

---

## Notebooks

| notebook | what it does |
|---|---|
| `01_data_preparation` | converts raw files → PNGs, writes `metadata.csv` and `preprocessing_config.json`. Knob: `DATASET` |
| `02_split` | generates 5-fold train/val/test CSVs using the chosen split scheme. Knobs: `DATASET`, `SPLIT_SCHEME` |
| `03_train` | trains one or all 5 folds for a segmentation experiment. Writes checkpoints, logs, and training-curve figures to local SSD, syncs to Drive at the end |
| `04_data_vis` | visualisation notebook: sample-prediction overlays (best / worst / random by Dice), per-class Dice bar chart, training-curves overlay across folds |
| `05_test` | runs batched inference on each fold's test set, saves predicted PNGs and per-fold manifests, writes all 6 CV tables (cv_results, cv_summary, cv_summary_enriched, cv_by_class, cv_class_summary, cv_per_image) |
| `06_cls_data` | QA: visualises patches extracted with GT masks vs predicted masks side-by-side |
| `07_train_cls` | trains classification experiment (GT masks only for training) |
| `08_test_cls` | Eval A (`EVAL_MASK_SOURCE="gt"`) and Eval B (`EVAL_MASK_SOURCE="predicted"`); computes and prints the Eval A − Eval B gap |

---

## Experiment configuration

Every run is fully specified by an `EXPERIMENT` dict in Cell 3 of the relevant
notebook. Changing one experiment only requires editing this dict — no source
files are touched between runs.

```python
EXPERIMENT = {
    "name":         "03_dicebce_image_level",
    "task":         "segmentation",
    "dataset":      "figshare",
    "split_scheme": "image_level",

    "model_name":      "smp_unet_resnet34",
    "encoder_weights": "imagenet",
    "loss_name":       "dice_bce",
    "loss_kwargs":     {},

    "optimizer_name":   "adam",
    "optimizer_kwargs": {"lr": 1e-4},
    "scheduler_name":   "reduce_on_plateau",
    "scheduler_kwargs": {"mode": "min", "factor": 0.1, "patience": 5},

    "augmentation_strength": "reference",
    "preprocessing":         "original",
    "image_size": 256,
    "batch_size": 8,
    "max_epochs": 100,
    "patience":   15,
    "threshold":  0.5,
    "seed":       42,
    "metric_kind": "micro_macro",
}
```

The full config is written to `experiment_config.json` alongside each
checkpoint so every result is reproducible from the saved dict alone.

---

## Prediction manifests

After NB05 runs, each fold writes a `manifest.json` recording:

- the checkpoint path and its SHA-256 hash
- the test CSV path and its SHA-256 hash
- number of predictions, image size, threshold, model name

Before classification Eval B reads any predicted masks, it calls
`verify_seg_predictions_match()` which recomputes the hashes and aborts if
they don't match. This catches the failure mode where the segmentation model
is retrained and the old predictions on disk silently go stale.

---

## Reference segmentation experiments (figshare, image-level)

| exp name | loss | architecture | reference Dice |
|---|---|---|---|
| `01_dice_image_level` | Dice | U-Net / ResNet34 | 0.8426 |
| `02_bce_image_level` | BCE | U-Net / ResNet34 | 0.8485 |
| `03_dicebce_image_level` | Dice + BCE | U-Net / ResNet34 | 0.8541 |
| `04_dicefocal_image_level` | Dice + Focal | U-Net / ResNet34 | 0.8509 |
| `05_lovasz_image_level` | Lovasz | U-Net / ResNet34 | 0.8418 |
| `06_clahe_dicebce_image_level` | CLAHE + Dice + BCE | U-Net / ResNet34 | 0.8501 |
| `07_unetpp_effb4_dicebce_image_level` | Dice + BCE | U-Net++ / EfficientNet-B4 | **0.8608** |

All use Adam lr=1e-4, ReduceLROnPlateau (factor=0.1, patience=5), max_epochs=100,
early-stopping patience=15, threshold=0.5, seed=42.

---

## Dev / run workflow

```
VS Code (desktop)
    edit src/*.py or notebooks/
    git commit + push

Google Colab
    Cell 1 — pip install requirements
    Cell 2 — mount Drive, git clone / git pull, copy data to local SSD
    Cell 3 — review / set EXPERIMENT dict
    ...run cells...
    final cell — sync outputs from local SSD back to Drive
```

Training writes everything to `/content/Senior_Project_local/` (local Colab
SSD) to avoid FUSE-hang on sustained Drive writes. One batched `copytree`
per output category syncs results to Drive at the end of the run.

Never edit `.py` files inside Colab — edits to the local clone die when the
runtime disconnects. All code changes go through VS Code → GitHub → `git pull`
in Cell 2.
