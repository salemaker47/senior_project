# Senior Project — Brain Tumor Analysis

Multi-dataset, multi-task medical imaging pipeline for brain MRI:

- **Segmentation** — binary tumor mask (256×256 grayscale → mask)
- **Classification** — tumor type (meningioma / glioma / pituitary)

## Datasets

| dataset | size | classes |
|---|---|---|
| `figshare` | 3,064 images / 233 patients | meningioma, glioma, pituitary |
| `brats2020` | 57,195 / 369 patients | tumor, no tumor ( segmentation only)

## Workflow

Code is edited on the desktop in VS Code, committed, and pushed to GitHub.
Notebooks run in Google Colab, which clones this repo and reads / writes data
in Google Drive. See the project instruction and dev/run instruction for the
full setup.

