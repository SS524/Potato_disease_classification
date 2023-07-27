from flask import Flask, render_template, request
import os
import tensorflow as tf
from prediction import PredictionPipeline
from PIL import Image
import numpy as np


app = Flask(__name__)

UPLOAD_FOLDER = 'testing_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method=='GET':
        return render_template('index.html')
    else:

        if 'image' not in request.files:
            return "No file part in the request."
        
        image_file = request.files['image']
        
    
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename))
        img= Image.open(os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename))
        print(img.size)
        print(np.array(img).shape)
        new_img=img.resize((256,256))
        new_img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'modified.jpg'))
        print(new_img.size)
        print(np.array(new_img).shape)
        np_img = np.array(new_img)
    

        pred_obj = PredictionPipeline(np_img)
        predicted_class, confidence = pred_obj.predict()
        


        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))


        return render_template('index.html',predicted_class=predicted_class, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
