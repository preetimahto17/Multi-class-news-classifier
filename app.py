from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model and vectorizer
with open('models/text_classification.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/count_vect.pkl', 'rb') as f1:
    count_vect = pickle.load(f1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        text = data['text']
        transformed_text = count_vect.transform([text])
        prediction = model.predict(transformed_text)
        return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')