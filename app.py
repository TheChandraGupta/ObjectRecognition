"""
Created on Fri Feb 15 20:19:21 2019

@author: M1041921
"""

import os
from flask import Flask, flash, redirect, render_template, request, session, abort, Response, jsonify
from flask_cors import CORS
from model.model import MyModel

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

my_model = MyModel()

@app.route("/")
def hello():
    data = {
        'Hello'  : 'World'
    }
    resp = jsonify(data)
    resp.status_code = 200
    return resp
    
@app.route('/upload', methods=['POST'])
def upload_file():
    print()
    file = request.files['test_img']
    file_name = file.filename
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #print(f)
    obj = file.save(f)
    obj_data = my_model.predict_image(file_name)
    data = {
        'prediction'  : obj_data
    }
    print(data)
    resp = jsonify(data)
    resp.status_code = 200
    print(resp)
    return resp

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=80)