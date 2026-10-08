import os
import gc

import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download


# Import the existing AdaIN code
from utils.model import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# =========================================================
# CPU MEMORY OPTIMIZATION
# =========================================================

# Render has limited RAM.
# Restrict PyTorch CPU threads to reduce memory usage.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

Bootstrap(app)

os.makedirs(
    app.config['UPLOAD_FOLDER'],
    exist_ok=True
)


# =========================================================
# HUGGING FACE MODEL DOWNLOAD
# =========================================================

MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


print("Downloading/loading VGG model...")

VGG_PATH = hf_hub_download(
    repo_id="rjaumania/adain-models",
    filename="vgg_normalised.pth",
    local_dir=MODEL_DIR
)


print("Downloading/loading decoder...")

DECODER_PATH = hf_hub_download(
    repo_id="rjaumania/adain-models",
    filename="decoder_150.pth",
    local_dir=MODEL_DIR
)


print("VGG model path:", VGG_PATH)
print("Decoder path:", DECODER_PATH)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================================================
# FORM
# =========================================================

class UploadForm(FlaskForm):

    content = FileField('Content image')
    style = FileField('Style image')

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField(
        'Alpha',
        default=1.0
    )

    submit = SubmitField(
        'Transfer Style'
    )


# =========================================================
# LOAD ADAIN MODEL
# =========================================================

print("Loading VGG encoder...")

encoder = VGGEncoder(
    VGG_PATH
).to(device)


print("Loading decoder...")

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        DECODER_PATH,
        map_location=device
    )
)


encoder.eval()
decoder.eval()

print("AdaIN models loaded successfully!")


# =========================================================
# FILE VALIDATION
# =========================================================

def allowed_file(filename):

    return (
        '.' in filename
        and
        filename.rsplit('.', 1)[1].lower()
        in app.config['ALLOWED_EXTENSIONS']
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
    device
):

    # -----------------------------------------------------
    # KEEPING 512 RESOLUTION
    # -----------------------------------------------------

    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])


    # -----------------------------------------------------
    # CONVERT IMAGES TO TENSORS
    # -----------------------------------------------------

    content_image = (
        content_transform(content_image)
        .unsqueeze(0)
        .to(device)
    )

    style_image = (
        style_transform(style_image)
        .unsqueeze(0)
        .to(device)
    )


    # -----------------------------------------------------
    # ADAIN INFERENCE
    # -----------------------------------------------------

    # inference_mode uses less memory than normal
    # autograd/no_grad inference.
    with torch.inference_mode():

        # Extract content features
        content_feats = encoder(
            content_image,
            is_test=True
        )


        # Extract style features
        style_feats = encoder(
            style_image,
            is_test=True
        )


        # Apply AdaIN
        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )


        # Alpha blending
        stylized_feats = (
            alpha * stylized_feats
            + (1 - alpha) * content_feats
        )


        # Decode stylized feature
        stylized_image = decoder(
            stylized_feats
        )


    # -----------------------------------------------------
    # FREE UNNECESSARY MEMORY
    # -----------------------------------------------------

    del content_image
    del style_image
    del content_feats
    del style_feats
    del stylized_feats

    gc.collect()


    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return stylized_image


# =========================================================
# SAVE IMAGE
# =========================================================

def save_image(tensor, path):

    image = tensor.cpu().clone()

    image = image.squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    image.save(path)


# =========================================================
# HOME PAGE
# =========================================================

@app.route('/', methods=['GET', 'POST'])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None


    # =====================================================
    # FORM SUBMITTED
    # =====================================================

    if form.validate_on_submit():

        # -------------------------------------------------
        # CONTENT IMAGE
        # -------------------------------------------------

        if (
            form.content.data
            and
            form.content.data.filename
        ):

            if allowed_file(
                form.content.data.filename
            ):

                content_filename = secure_filename(
                    form.content.data.filename
                )

                form.content.data.save(
                    os.path.join(
                        app.config['UPLOAD_FOLDER'],
                        content_filename
                    )
                )

                form.content_path.data = content_filename

        else:

            content_filename = form.content_path.data


        # -------------------------------------------------
        # STYLE IMAGE
        # -------------------------------------------------

        if (
            form.style.data
            and
            form.style.data.filename
        ):

            if allowed_file(
                form.style.data.filename
            ):

                style_filename = secure_filename(
                    form.style.data.filename
                )

                form.style.data.save(
                    os.path.join(
                        app.config['UPLOAD_FOLDER'],
                        style_filename
                    )
                )

                form.style_path.data = style_filename

        else:

            style_filename = form.style_path.data


        # -------------------------------------------------
        # STYLE TRANSFER
        # -------------------------------------------------

        if content_filename and style_filename:

            content_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                content_filename
            )

            style_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                style_filename
            )


            try:

                # Load content image
                content_image = Image.open(
                    content_path
                ).convert('RGB')


                # Load style image
                style_image = Image.open(
                    style_path
                ).convert('RGB')


                # Get alpha value
                alpha = float(
                    form.alpha.data
                )


                # Keep alpha between 0 and 1
                alpha = max(
                    0.0,
                    min(1.0, alpha)
                )


                print("Starting style transfer...")
                print("Image size: 512x512")
                print("Alpha:", alpha)


                # Perform AdaIN
                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device
                )


                # -------------------------------------------------
                # SAVE RESULT
                # -------------------------------------------------

                result_filename = (
                    'stylized_' + content_filename
                )


                result_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    result_filename
                )


                save_image(
                    stylized_image,
                    result_path
                )


                result_image = result_filename


                print(
                    "Style transfer completed successfully!"
                )


                # -------------------------------------------------
                # CLEANUP
                # -------------------------------------------------

                del stylized_image
                del content_image
                del style_image

                gc.collect()


            except Exception as e:

                print(
                    "Style transfer error:",
                    e
                )

                error = str(e)


        else:

            if not content_filename:

                error = 'Please upload content image'


            if not style_filename:

                error = 'Please upload style image'


    # =====================================================
    # RENDER TEMPLATE
    # =====================================================

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# =========================================================
# SERVE UPLOADED IMAGES
# =========================================================

@app.route('/uploads/<filename>')
def send_image(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


# =========================================================
# EXAMPLES
# =========================================================

@app.route('/examples/<path:filename>')
def send_example(filename):

    return send_from_directory(
        'examples',
        filename
    )


# =========================================================
# LOCAL / RENDER RUN
# =========================================================

if __name__ == '__main__':

    port = int(
        os.environ.get(
            'PORT',
            5000
        )
    )

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )