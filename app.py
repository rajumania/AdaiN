
import os
import gc
import threading

import torch
from flask import Flask, render_template, send_from_directory, request
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from huggingface_hub import hf_hub_download

from utils.model import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# =========================================================
# PYTORCH CPU OPTIMIZATION
# =========================================================

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "supersecretkey"
)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

Bootstrap(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.errorhandler(413)
def file_too_large(error):
    return "Upload too large. Please use images under 10 MB combined.", 413


# =========================================================
# MODEL DOWNLOADS
# =========================================================

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Downloading/loading VGG model...", flush=True)

VGG_PATH = hf_hub_download(
    repo_id="rjaumania/adain-models",
    filename="vgg_normalised.pth",
    local_dir=MODEL_DIR,
)

print("Downloading/loading decoder...", flush=True)

DECODER_PATH = hf_hub_download(
    repo_id="rjaumania/adain-models",
    filename="decoder_150.pth",
    local_dir=MODEL_DIR,
)

print("VGG model path:", VGG_PATH, flush=True)
print("Decoder path:", DECODER_PATH, flush=True)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device, flush=True)


# =========================================================
# UPLOAD FORM
# =========================================================

class UploadForm(FlaskForm):
    content = FileField("Content image")
    style = FileField("Style image")

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField("Alpha", default=1.0)
    submit = SubmitField("Transfer Style")


# =========================================================
# LOAD MODELS ONCE
# =========================================================

print("Loading VGG encoder...", flush=True)

encoder = VGGEncoder(VGG_PATH).to(device)
encoder.eval()

print("Loading decoder...", flush=True)

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(DECODER_PATH, map_location=device)
)

decoder.eval()

print("AdaIN models loaded successfully!", flush=True)


# Prevent concurrent inference from multiplying peak memory use.
inference_lock = threading.Lock()


# =========================================================
# FILE VALIDATION
# =========================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


# =========================================================
# RESIZE WHILE PRESERVING ASPECT RATIO
# =========================================================

def resize_keep_aspect(image, max_size=256):
    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions.")

    scale = min(
        max_size / width,
        max_size / height,
        1.0,
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


# =========================================================
# STYLE TRANSFER
# =========================================================

def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device,
):
    content_image = resize_keep_aspect(
        content_image, max_size=256
    )
    style_image = resize_keep_aspect(
        style_image, max_size=256
    )

    print("Content size:", content_image.size, flush=True)
    print("Style size:", style_image.size, flush=True)

    to_tensor = transforms.ToTensor()

    content_tensor = to_tensor(content_image).unsqueeze(0).to(device)
    style_tensor = to_tensor(style_image).unsqueeze(0).to(device)

    content_feats = None
    style_feats = None
    stylized_feats = None
    output = None

    try:
        with torch.inference_mode():
            print("Extracting content features...", flush=True)
            content_feats = encoder(content_tensor, is_test=True)

            print("Extracting style features...", flush=True)
            style_feats = encoder(style_tensor, is_test=True)

            print("Applying AdaIN...", flush=True)
            stylized_feats = adaptive_instance_normalization(
                content_feats,
                style_feats,
            )

            # Blend in-place to avoid allocating another feature tensor.
            stylized_feats.mul_(alpha)
            stylized_feats.add_(
                content_feats,
                alpha=(1.0 - alpha),
            )

            # Release encoder features before decoding.
            del content_feats
            content_feats = None

            del style_feats
            style_feats = None

            del content_tensor, style_tensor
            content_tensor = None
            style_tensor = None

            gc.collect()

            print("Generating stylized image...", flush=True)
            output = decoder(stylized_feats)

            # Move result to CPU before releasing feature maps.
            output = output.cpu()

            del stylized_feats
            stylized_feats = None

            gc.collect()

        print("Style transfer completed!", flush=True)
        return output

    finally:
        # Release remaining tensors even if inference raises an error.
        for name in (
            "content_tensor",
            "style_tensor",
            "content_feats",
            "style_feats",
            "stylized_feats",
        ):
            value = locals().get(name)
            if value is not None:
                del value

        gc.collect()


# =========================================================
# SAVE OUTPUT IMAGE
# =========================================================

def save_image(tensor, path):
    image_tensor = tensor.detach().cpu().squeeze(0).clamp(0, 1)

    image = transforms.ToPILImage()(image_tensor)
    image.save(path, format="PNG")


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():
    print("REQUEST METHOD:", request.method, flush=True)

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == "POST":
        print("POST REQUEST RECEIVED", flush=True)

    if form.validate_on_submit():
        print("FORM VALIDATION SUCCESS", flush=True)

        content_upload = form.content.data
        style_upload = form.style.data

        # Require new uploads for each transfer.
        if not content_upload or not content_upload.filename:
            error = "Please upload a content image."
        elif not style_upload or not style_upload.filename:
            error = "Please upload a style image."
        elif not allowed_file(content_upload.filename):
            error = "Content image must be PNG, JPG, or JPEG."
        elif not allowed_file(style_upload.filename):
            error = "Style image must be PNG, JPG, or JPEG."
        else:
            content_filename = secure_filename(content_upload.filename)
            style_filename = secure_filename(style_upload.filename)

            # Avoid empty or unsafe filenames.
            if not content_filename or not style_filename:
                error = "Invalid filename. Please rename your images."
            else:
                content_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    content_filename,
                )
                style_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    style_filename,
                )

                try:
                    print("Opening uploaded images...", flush=True)

                    with Image.open(content_path) if False else open(os.devnull, "rb") as _unused:
                        pass

                    content_upload.save(content_path)
                    style_upload.save(style_path)

                    with Image.open(content_path) as image:
                        content_image = image.convert("RGB")

                    with Image.open(style_path) as image:
                        style_image = image.convert("RGB")

                    alpha = float(form.alpha.data or 1.0)
                    alpha = max(0.0, min(1.0, alpha))

                    print("Starting style transfer", flush=True)
                    print("Alpha:", alpha, flush=True)
                    print("Maximum image dimension: 256", flush=True)

                    # Only one inference at a time.
                    with inference_lock:
                        stylized_image = style_transfer(
                            content_image,
                            style_image,
                            encoder,
                            decoder,
                            alpha,
                            device,
                        )

                    result_filename = (
                        "stylized_" + os.path.splitext(content_filename)[0]
                        + ".png"
                    )

                    result_path = os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        result_filename,
                    )

                    save_image(stylized_image, result_path)

                    result_image = result_filename

                    print("Result saved:", result_path, flush=True)

                    del stylized_image
                    del content_image
                    del style_image
                    gc.collect()

                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    print("Image processing error:", repr(exc), flush=True)
                    error = "Could not process an image. Please try valid JPG or PNG files."

                except Exception as exc:
                    print("Style transfer error:", repr(exc), flush=True)
                    error = "Style transfer failed. Please check the server logs."

    elif request.method == "POST":
        print("FORM VALIDATION FAILED:", form.errors, flush=True)
        if not error:
            error = "Form validation failed. Please upload both images and try again."

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
    )


# =========================================================
# SERVE UPLOADED IMAGES
# =========================================================

@app.route("/uploads/<path:filename>")
def send_image(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
    )


# =========================================================
# EXAMPLES
# =========================================================

@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory("examples", filename)


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=False,
    )
