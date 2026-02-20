import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from model import Net

# Load model
model = Net()
model.load_state_dict(torch.load("fashion_mnist.pth", map_location="cpu"))
model.eval()
def collate_fn(text):
  return "whatever shit"

app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
  data = request.get_json()
  data = collate_fn(data['features])
  futuip = np.array(data).reshape(-1,1)
  predict = model.predict(futuip)
  return jsonify({'prediction':predict[0].item()})

if __name__ == '__main__':
    app.run(debug=True)
