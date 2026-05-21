
from flask import Flask, render_template, request, jsonify
from inference import load_json_file, gen_pred, to_eng
import os
from torch import nn
from flask_ngrok import run_with_ngrok
from PIL import Image



C2LP = './class2label.json'
encodername = 'efficientnet'
MW = './finalw.pth'
UPLOAD_FOLDER = 'static/'

class2label = load_json_file(C2LP)
translate = load_json_file('./rus_direction_to_eng_direction.json') 

app = Flask(__name__) 
run_with_ngrok(app)  # Start ngrok when app is run
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return render_template('index.html') # HomePage


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':  
        file = request.files.get('file') # Extracting The File From Storage
        pred = gen_pred(file, class2label)
        en_pred = to_eng(pred, translate)
        image = Image.open(file)
        image.save(os.path.join(app.config["UPLOAD_FOLDER"], file.filename))
        filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        return render_template('result.html', pred_en=en_pred, pred_ru=pred, image=filename)
