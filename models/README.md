# Models

The trained U-Net checkpoint is saved here after running the notebook:

```text
unet_pet_segmentation.pth
```

The checkpoint is intentionally not included in the repository ZIP because binary model files can become large and are reproducible from the notebook.

The Flask application automatically looks for:

```text
models/unet_pet_segmentation.pth
```

If the checkpoint is missing, run the training cells in the notebook first.
