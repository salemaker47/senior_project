# Senior Project — Brain Tumor Analysis

Multi-dataset, multi-task medical imaging pipeline for brain MRI:

- **Segmentation** — binary tumor mask (256×256 grayscale → mask)
- **Classification** — tumor type (meningioma / glioma / pituitary)

## Datasets

| dataset | size | classes |
|---|---|---|
| `figshare` | 3,064 images / 233 patients | meningioma, glioma, pituitary |

## Workflow

Code is edited on the desktop in VS Code, committed, and pushed to GitHub.
Notebooks run in Google Colab, which clones this repo and reads / writes data
in Google Drive. See the project instruction and dev/run instruction for the
full setup.

## Open in Colab

Notebook badges will be added as notebooks land in subsequent milestones.

- [ ] `notebooks/setups/00_migration.ipynb` (M2)
- [ ] `notebooks/setups/01_data_preparation.ipynb` (M3)
- [ ] `notebooks/setups/02_split.ipynb` (M4)
- [ ] `notebooks/segmentation/03_train.ipynb` (M5)
- [ ] `notebooks/segmentation/04_results.ipynb` (M6)
- [ ] `notebooks/segmentation/05_test.ipynb` (M6)