# Medical Image Segmentation

A lightweight U-Net project for segmenting regions of interest in medical images. The project uses the open-source Oxford-IIIT Pet dataset as a practical, small-scale segmentation benchmark, while keeping the training workflow CPU-friendly.

> **Interview project:** The notebook explains the segmentation pipeline, U-Net architecture, training, evaluation, and inference. A small Flask web app lets a user upload an image and view the predicted segmentation mask.

## Features

- U-Net implemented with PyTorch.
- Semantic segmentation with image/mask pairs.
- CPU-friendly training defaults.
- Dataset downloaded with `requests` and handled with `pathlib`.
- Notebook-first workflow.
- Dice score and IoU evaluation.
- Saved lightweight model checkpoint.
- Flask demo for interactive inference.
- GitHub and LinkedIn links included in the web app.

## Dataset

This project uses the **Oxford-IIIT Pet Dataset**, an open dataset containing pet images and pixel-level segmentation trimaps. It is used here as a lightweight segmentation benchmark so the project can run without a dedicated GPU.

Dataset information: https://www.robots.ox.ac.uk/~vgg/data/pets/

## Repository Structure

```text
Medical-Image-Segmentation/
├── app.py
├── requirements.txt
├── README.md
├── CONTRIBUTE.md
├── CHANGELOG.md
├── notebooks/
│   └── medical_image_segmentation.ipynb
├── models/
│   └── README.md
├── data/
│   └── README.md
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   └── result.html
└── uploads/
    └── .gitkeep
```

## Run the Notebook

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook notebooks/medical_image_segmentation.ipynb
```

The notebook downloads the dataset only when needed. For a CPU-only computer, start with the small training configuration shown in the notebook.

## Run the Flask App

After training and saving `models/unet_pet_segmentation.pth`:

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

The application accepts a pet image, runs the trained U-Net model, and displays the predicted foreground mask.

## Important Note

This is an educational computer-vision segmentation project, not a clinical diagnostic system. The dataset contains animal images rather than human medical scans, which keeps the project lightweight and suitable for demonstrating U-Net segmentation fundamentals without implying medical diagnosis.

## Author

**Praveen Kumar**

- GitHub: https://github.com/InfinitePraveen
- LinkedIn: https://www.linkedin.com/in/infinitepraveen/
