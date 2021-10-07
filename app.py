import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
import PIL
import numpy as np
from tensorflow.keras.models import Model, load_model
import shutil

# UPLOAD_FOLDER = './uploads/'
UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
CORS(app)

class_dictionary = {
    '10 Rupees': 0,
    '100 Rupees': 1,
    '20 Rupees': 2,
    '200 Rupees': 3,
    '2000 Rupees': 4,
    '50 Rupees': 5,
    '500 Rupees': 6
}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home_page():
    return "Hello World"


@app.route('/currencydetection', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':

        if 'file' not in request.files:
            return jsonify(message="No file selected")
        file = request.files['file']

        if file.filename == '':
            return jsonify(success=False, message="No file selected")

        if file and allowed_file(file.filename):
            file.save(file.filename)

            prediction = predict(file.filename)
            response = jsonify(success=True, prediction=prediction)
            response.headers.add("Access-Control-Allow-Origin", "*")
            # shutil.rmtree("./uploads/")
            os.remove(file.filename)
            return response
        else:
            return jsonify(success=False,
                           message="Only jpg, jpeg, png and gif files allowed")

    elif request.method == 'GET':
        return jsonify(success=False, message="Cannot GET")


def load_image(img_path):

    img = image.load_img(img_path, target_size=(250, 500))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor


def predict(img_path):
    model = load_model("Xception.h5")
    new_image = load_image(img_path)
    pred = model.predict(new_image)
    z = np.argmax(pred)
    return list(class_dictionary.keys())[z]


if __name__ == '__main__':
    app.debug = True
    app.run()
