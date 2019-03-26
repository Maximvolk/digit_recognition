from app import app
from flask import render_template, jsonify, request
from app.recognize_digits import recognize


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def predict():
    label, prob = recognize(request.form['img'])
    return jsonify({'prediction': label,
                    'probability': prob})
