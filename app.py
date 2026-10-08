import os
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms
import io



# import the existing AdIN Code
from utils.model import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, cal_mean_std

app=Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# form create kar abbb
class UploadForm(FlaskForm):
    content = FileField('Content image')
    style = FileField('Style image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default = 1.0)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VGGEncoder('vgg_normalised.pth').to(device)

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        r"C:\Users\rajub\OneDrive\Documents\Desktop\NST_PROJECT\NST_P\experiment\experiment_irh_b4\decoder_150.pth",
        map_location=device
    )
)

# bro we have not to train the model sho we will evalatite it okkk bhai...

encoder.eval()
decoder.eval()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



def style_transfer(content_image, style_image, encoder, decoder, alpha, device):

    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():

        # Extract content and style features
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

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
        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(tensor, path):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0,1)
    image = transforms.ToPILImage()(image)
    image.save(path)
    

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data =  content_filename

        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:

            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:

            content_path =  os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path =  os.path.join(app.config['UPLOAD_FOLDER'], style_filename)


            try: 
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')
                alpha = float(form.alpha.data)
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                result_filename = 'stylized' + content_filename

                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)

                result_image = result_filename

            except Exception as e:
                error = str(e)

        



        
    else:
        if not content_filename:
            error = 'Please upload contnet Image'

        if not style_filename:
            error = 'Please upload style image'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename, style_image=style_filename, error=error )


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)


