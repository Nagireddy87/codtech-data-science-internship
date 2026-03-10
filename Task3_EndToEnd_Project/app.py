from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "House Price Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    bedrooms = data["bedrooms"]
    bathrooms = data["bathrooms"]
    sqft = data["sqft_living"]

    features = np.array([[bedrooms, bathrooms, sqft]])

    prediction = model.predict(features)

    return jsonify({
        "predicted_price": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)