from pathlib import Path

import numpy as np
import torch
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
MODEL_PATH = BASE_DIR / "models" / "unet_pet_segmentation.pth"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# U-Net
# ---------------------------------------------------------

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(3, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)

        self.pool = torch.nn.MaxPool2d(2)

        self.up2 = torch.nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.conv2 = DoubleConv(128, 64)

        self.up1 = torch.nn.ConvTranspose2d(
            64, 32, 2, stride=2
        )
        self.conv1 = DoubleConv(64, 32)

        self.out = torch.nn.Conv2d(32, 1, 1)

    def forward(self, x):

        x1 = self.down1(x)

        x2 = self.down2(
            self.pool(x1)
        )

        x3 = self.down3(
            self.pool(x2)
        )

        x = self.up2(x3)

        x = torch.cat(
            [x, x2],
            dim=1
        )

        x = self.conv2(x)

        x = self.up1(x)

        x = torch.cat(
            [x, x1],
            dim=1
        )

        x = self.conv1(x)

        return self.out(x)


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

model = UNet()

if not MODEL_PATH.exists():

    print(
        "\nWARNING: Model checkpoint was not found."
    )

    print(
        f"Expected model at: {MODEL_PATH}"
    )

    print(
        "Run the training notebook first.\n"
    )

else:

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    print(
        f"Loaded model: {MODEL_PATH}"
    )


model.eval()


# ---------------------------------------------------------
# IMPORTANT:
# This MUST match the notebook preprocessing.
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(
        (128, 128)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict_segmentation(image):

    original_image = image.copy()

    image_tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )[0, 0].numpy()

    return original_image, probabilities


# ---------------------------------------------------------
# Home
# ---------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "GET":

        return render_template(
            "index.html"
        )

    image_file = request.files.get(
        "image"
    )

    if (
        image_file is None
        or not image_file.filename
    ):

        return render_template(
            "index.html",
            error="Please choose an image."
        )

    safe_name = Path(
        image_file.filename
    ).name

    input_path = (
        UPLOAD_DIR / safe_name
    )

    image_file.save(
        input_path
    )

    image = Image.open(
        input_path
    ).convert("RGB")

    original_image, probabilities = (
        predict_segmentation(image)
    )

    # -----------------------------------------------------
    # Binary mask
    # -----------------------------------------------------

    binary_mask = (
        probabilities > 0.5
    ).astype(np.uint8) * 255

    # -----------------------------------------------------
    # Probability mask
    # Useful for debugging.
    # -----------------------------------------------------

    probability_mask = (
        probabilities * 255
    ).clip(
        0,
        255
    ).astype(np.uint8)

    # -----------------------------------------------------
    # Save binary mask
    # -----------------------------------------------------

    stem = Path(
        safe_name
    ).stem

    binary_name = (
        f"{stem}_binary_mask.png"
    )

    probability_name = (
        f"{stem}_probability_mask.png"
    )

    overlay_name = (
        f"{stem}_overlay.png"
    )

    Image.fromarray(
        binary_mask
    ).save(
        RESULT_DIR / binary_name
    )

    Image.fromarray(
        probability_mask
    ).save(
        RESULT_DIR / probability_name
    )

    # -----------------------------------------------------
    # Create overlay
    # -----------------------------------------------------

    original_resized = original_image.resize(
        (128, 128)
    )

    original_array = np.array(
        original_resized
    ).astype(np.float32)

    # Create a visible red overlay.
    overlay = original_array.copy()

    foreground = probabilities > 0.5

    overlay[foreground, 0] = 255
    overlay[foreground, 1] *= 0.35
    overlay[foreground, 2] *= 0.35

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    Image.fromarray(
        overlay
    ).save(
        RESULT_DIR / overlay_name
    )

    # -----------------------------------------------------
    # Debug information
    # -----------------------------------------------------

    print(
        "\nPrediction statistics:"
    )

    print(
        "Minimum probability:",
        float(probabilities.min())
    )

    print(
        "Maximum probability:",
        float(probabilities.max())
    )

    print(
        "Mean probability:",
        float(probabilities.mean())
    )

    print(
        "Foreground percentage:",
        float(
            (foreground.mean()) * 100
        ),
        "%"
    )

    return render_template(
        "result.html",

        filename=safe_name,

        result_filename=binary_name,

        probability_filename=probability_name,

        overlay_filename=overlay_name,

        min_probability=float(
            probabilities.min()
        ),

        max_probability=float(
            probabilities.max()
        ),

        mean_probability=float(
            probabilities.mean()
        ),

        foreground_percentage=float(
            foreground.mean() * 100
        )
    )


# ---------------------------------------------------------
# Uploaded images
# ---------------------------------------------------------

@app.route(
    "/uploads/<path:filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_DIR,
        filename
    )


# ---------------------------------------------------------
# Run Flask
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )
