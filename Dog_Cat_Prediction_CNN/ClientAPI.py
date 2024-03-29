from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from Dhruvil_openCV.utils import decodeImage
from predict import dogcat

app = Flask(__name__)
CORS(app)

#@cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = dogcat(self.filename)

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predictiondogcat()
    return jsonify(result)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='127.0.0.1', port=8000, debug=True)