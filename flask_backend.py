from flask import Flask, request, jsonify, render_template, url_for, make_response
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("random_forest_model.joblib")

@app.route('/')
def home():
    response = make_response(render_template('homepage.html'))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

@app.route('/index')
def index():
    response = make_response(render_template('index.html'))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

@app.route('/diabetes_positive')
def diabetes_positive():
    response = make_response(render_template('diabetes_positive.html'))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

@app.route('/diabetes_negative')
def diabetes_negative():
    response = make_response(render_template('diabetes_negative.html'))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    print("Received Input Features:", data)  

    prediction = model.predict([data])[0]
    print("Model Prediction Output:", prediction)  

    if prediction == 1:
        return jsonify({"redirect": url_for('diabetes_positive')})
    else:
        return jsonify({"redirect": url_for('diabetes_negative')})

if __name__ == '__main__':
    app.run(debug=True)
