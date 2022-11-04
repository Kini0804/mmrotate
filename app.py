import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from detection import Detection
from flask import Flask, request, Response, render_template, jsonify, flash, send_file
import os
import pdb

detector = Detection("config.json")
app = Flask(__name__)

@app.route('/status', methods= ['GET'])
def hello():
    return "The AI-Graph-Detection service is running"

@app.route('/', methods= ['POST'])
def ImageObjectDetection():
    # pdb.set_trace()
    image = request.files["file"]
    image_name = image.filename
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    image_path = os.path.join('tmp', image_name)
    image.save(image_path)
    image.close()
    detector.detect(image_path)
    os.remove(image_path)
    return send_file("tmp/result.jpg", mimetype="image/jpg", download_name="result.jpg")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

