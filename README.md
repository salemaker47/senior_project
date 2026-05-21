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

Because classification and segmentation share the same fold CSVs, fold *k*'s
test set is identical for both tasks. This means Eval B can look up predicted
masks for fold *k*'s test images without any out-of-fold lookup.

---

## Folder structure

```
Senior_Project/
├── configs/
│   ├── seg/reference_experiments.py    canonical seg recipes (01–07)
│   └── cls/reference_experiments.py    canonical cls recipes (cls01–cls04)
│
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
│   ├── tables/
│   │   ├── segmentation/<dataset>/<exp>/
│   │   │   ├── cv_results.csv, cv_summary.csv, cv_summary_enriched.csv
│   │   │   ├── cv_by_class.csv, cv_class_summary.csv, cv_per_image.csv
│   │   │   └── fold_X_*.csv
│   │   └── classification/<dataset>/<cls_exp>/
│   │       ├── eval_gt/                Eval A tables
│   │       └── eval_pred__<seg_exp>/   Eval B tables
│   └── figures/
│       ├── segmentation/<dataset>/<exp>/fold_X/sample_predictions/
│       └── classification/<dataset>/<cls_exp>/<eval_variant>/
│
├── notebooks/
│   ├── setups/      01_data_preparation_{figshare,brats2020}
│   ├── segmentation/ 03_train, 04_data_vis, 05_test, 06_seg_compare
│   └── classification/ 06_classification_data, 07_train_cls, 08_test_cls, 09_cls_results
│
└── src/             see Source modules below
```

Artifacts under `data/` and `outputs/` live in Google Drive and are not
version-controlled. Code in `src/`, `configs/`, and `notebooks/` lives here on GitHub.

---

## Source modules

### Shared

| file | responsibility |
|---|---|
| `file_utils.py` | canonical path builders (`experiment_paths`, `cls_eval_paths`, `seg_predictions_dir`, …), JSON helpers, SHA-256 file hashing, prediction-manifest readers and verifier. Single definition of `PathLike`. |
| `notebook_setup.py` | Colab bootstrap: `setup_environment` (mount Drive + git pull), `copy_to_local` (Drive → local SSD), `sync_outputs_to_drive` (local SSD → Drive at end of run) |
| `preprocess_utils.py` | raw-file converters for figshare (.mat) and brats2020 (.h5) → 256×256 PNGs + metadata records; `get_dataset_converter` dispatch |
| `vis_utils.py` | shared image loaders (`load_grayscale_png`, `load_binary_mask_png`) and `show_class_examples` grid (NB01) |
| `data_utils.py` | metadata loading/validation, patient-level and image-level fold splitting, leakage verification, fold CSV persistence |
| `eval_utils.py` | `enriched_aggregate` (cross-fold mean/std/median/IQR/CI table); `build_fold_summary`; `add_mean_std` (shared mean/std helper used by both task aggregators) |
| `optimizers.py` | string-keyed optimizer registry (adam, adamw, sgd, rmsprop) and scheduler registry (reduce_on_plateau, cosine, cosine_warm_restarts, step, multistep, exponential) |
| `train_utils.py` | `set_global_seed`, `gather_repro_metadata` (git + GPU stamp), `TrainingTimingCallback`, `EpochSummaryPrinter`, `build_callbacks`, `build_trainer`, `export_plain_state_dict`, `strip_model_prefix` |

### Segmentation (`sg_` prefix)

| file | responsibility |
|---|---|
| `sg_data_utils.py` | `BrainTumorDataset`, augmentation pipelines (`build_train_transform`, `build_eval_transform`), `build_dataloaders`, `build_test_loader`. Source of truth for `IMAGENET_MEAN/STD`. |
| `sg_metrics.py` | `binarize_logits`, `dice_score`, `iou_score`, SMP-backed `get_smp_stats` / `micro_dice_from_stats` / `micro_iou_from_stats`, full per-image metric suite (`compute_per_image_metrics_from_logits`), metric-kind registry (`get_metric_kind_pairs`) |
| `sg_losses.py` | loss factory: bce, dice, focal, lovasz, dice_bce, dice_focal, combo (`CombinedLoss`) |
| `sg_models.py` | SMP model factory: U-Net/ResNet34, U-Net/ResNet50, U-Net++/EfficientNet-B4, U-Net++/ResNet34, Linknet/ResNet34, MA-Net/ResNet34 |
| `sg_lightning_module.py` | `BrainTumorSegModule` — training/validation steps, micro-pooled metric accumulation, registry-driven optimizer + scheduler |
| `sg_eval_utils.py` | `micro_dice_from_counts`, `micro_iou_from_counts`, `micro_sensitivity_from_counts`, `micro_precision_from_counts`; `summarize_fold_results`, `aggregate_cv_results`, `aggregate_cv_per_patient`, `aggregate_cv_training_summary` |
| `sg_test_utils.py` | `load_model_from_pt`, `load_model_from_ckpt`, `predict_mask` (single-image), `evaluate_fold` (batched inference + PNG saving + per-fold manifest), `write_experiment_manifest` |
| `sg_vis_utils.py` | `show_triplet`, `show_overlay_triplet`, `show_image_gt_pred_overlay` |

### Classification (`cls_` prefix)

| file | responsibility |
|---|---|
| `cls_data_utils.py` | `extract_patch` (mask-guided ROI crop), `BrainTumorClsDataset`, cls transforms, `build_dataloaders_cls`, `build_test_loader_cls`. Supports both `mask_source="gt"` and `mask_source="predicted"`. |
| `cls_models.py` | timm-based model factory: resnet50, efficientnet_b0/b4, vit_small_patch16_224 |
| `cls_losses.py` | loss factory: cross_entropy, cross_entropy_smooth (label smoothing), focal_ce |
| `cls_metrics.py` | `macro_f1_from_preds`, `accuracy_from_preds`, `per_class_metrics`, `confusion_matrix_from_preds`, `compute_per_image_metrics_cls` |
| `cls_lightning_module.py` | `BrainTumorClsModule` — training/validation steps, per-epoch macro F1 + accuracy accumulation, registry-driven optimizer + scheduler |
| `cls_test_utils.py` | `load_cls_model_from_pt`, `evaluate_fold_cls` (handles both mask sources, writes per-fold manifest and tables) |
| `cls_eval_utils.py` | `aggregate_cv_results_cls` (cross-fold tables + enriched summary + confusion matrix), `aggregate_cv_confusion_from_matrix` |
| `cls_vis_utils.py` | `plot_confusion_matrix`, `plot_confusion_pair`, `plot_per_class_f1`, `plot_eval_gap`, `plot_sample_patches` |

---

## Notebooks

| notebook | what it does |
|---|---|
| `01_data_preparation_figshare` | converts figshare .mat files → PNGs, writes `metadata.csv` and `preprocessing_config.json` |
| `01_data_preparation_brats2020` | converts brats2020 .h5 slices → PNGs (FLAIR + whole-tumor mask), same output layout |
| `02_split` | generates 5-fold train/val/test CSVs using the chosen split scheme. Knobs: `DATASET`, `SPLIT_SCHEME` |
| `03_train` | trains one or all 5 folds for a segmentation experiment; syncs checkpoints + logs to Drive |
| `04_data_vis` | qualitative figures: sample-prediction overlays (best / worst / random by Dice), per-class Dice bar chart, training-curves across folds |
| `05_test` | batched inference → predicted PNGs + per-fold manifests + all 6 CV tables (cv_results, cv_summary, cv_summary_enriched, cv_by_class, cv_class_summary, cv_per_image) |
| `06_seg_compare` | cross-experiment comparison: overlay multiple `cv_summary.csv` tables, rank by Dice |
| `06_classification_data` | QA: visualises patches extracted with GT masks vs predicted masks side-by-side |
| `07_train_cls` | trains one or all 5 classification folds (GT masks only); syncs to Drive |
| `08_test_cls` | Eval A (`mask_source="gt"`) and Eval B (`mask_source="predicted"`); prints the Eval A − Eval B macro F1 gap |
| `09_cls_results` | loads saved CSVs from Drive, produces publication-ready figures: confusion matrix pair, per-class F1 bars, Eval A vs B gap chart, sample-patch grid |

---

## Experiment configuration

Every run is fully specified by an `EXPERIMENT` dict in Cell 3 of the relevant
notebook. The canonical recipes live in `configs/seg/reference_experiments.py`
and `configs/cls/reference_experiments.py`. Changing one experiment only requires
editing this dict — no source files are touched between runs.

```python
# Segmentation
EXPERIMENT = get_experiment("03_dicebce_image_level", fold=1)

# Classification
EXPERIMENT = get_experiment(
    "cls01_resnet50",
    dataset="figshare",
    split_scheme="image_level",
    fold=1,
    name="cls01_resnet50",    # always pass name= explicitly
)
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

## Reference experiments

### Segmentation (figshare, image-level)

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

### Classification (figshare, image-level)

| exp name | model | notes |
|---|---|---|
| `cls01_resnet50` | ResNet-50 | baseline; Eval B uses `07_unetpp_effb4_dicebce_image_level` |
| `cls02_effb0` | EfficientNet-B0 | |
| `cls03_effb4` | EfficientNet-B4 | batch_size=16 for VRAM |
| `cls04_vit` | ViT-Small/16 | batch_size=16 for VRAM |

All use AdamW lr=1e-4 wd=1e-4, cosine schedule (T_max=50), label-smoothing CE
(smoothing=0.1), max_epochs=50, early-stopping patience=10, seed=42.

---

## Dev / run workflow

```
VS Code (desktop)
    edit src/*.py, configs/*.py, or notebooks/
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
