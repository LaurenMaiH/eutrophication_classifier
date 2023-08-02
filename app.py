from flask import Flask, request, render_template
import pickle
from PIL import Image
from fastai.vision.all import *


app = Flask(__name__, static_url_path='/static', static_folder='static')
path2 = Path()
model = load_learner(path2/'export.pkl')

@app.route('/')
def home():
    return render_template("index.html")
  

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files["water"]
    img = Image.open(img_file.stream)
    img = img.convert("RGB")
    img.save('upload.jpg')
    pred, pred_idx, probs = model.predict(img)
    probability = round(probs[pred_idx].item(), 4)

    return render_template('index.html', prediction_text=f'The image of water you uploaded looks to be {pred}. Probability: {probability}%')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/examples')
def example():
    return render_template('examples.html')

if __name__ == "__main__":
    app.run(debug=True)