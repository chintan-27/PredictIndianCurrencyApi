import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
import PIL
import numpy as np
from tensorflow.keras.models import Model, load_model

UPLOAD_FOLDER = '/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
CORS(app)

class_dictionary = {
    '10 Rs': 0,
    '100 Rs': 1,
    '20 Rs': 2,
    '200 Rs': 3,
    '2000 Rs': 4,
    '50 Rs': 5,
    '500 Rs': 6
}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':

        if 'file' not in request.files:
            return jsonify(message="No file selected")
        file = request.files['file']

        if file.filename == '':
            return jsonify(success=False, message="No file selected")

        if file and allowed_file(file.filename):
            file.save(os.path.join(os.getcwd() + UPLOAD_FOLDER, file.filename))

            prediction = predict("uploads/" + file.filename)
            response = jsonify(
                success=True,
                prediction=prediction,
                link="https://extract-text-image.herokuapp.com/static/uploads/"
                + file.filename)
            response.headers.add("Access-Control-Allow-Origin", "*")
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
    model = load_model("MobileNet.h5")
    new_image = load_image(img_path)
    pred = model.predict(new_image)
    z = np.argmax(pred)
    return list(class_dictionary.keys())[z]


if __name__ == '__main__':
    app.debug = True
    app.run()
