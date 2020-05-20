# importing the packages
import json
import pickle
# import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, session, logging
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# value of __name__ should be  '__main__'
app = Flask(__name__)
# Loading model so that it works on production
model = joblib.load('./model/model.pkl')


@app.route('/')
def index():
	# Index page
	return render_template('index.html')

@app.route('/about')
def about():
	# about me page
	return render_template('about.html')

@app.route('/products')
def PRODUCTS():
    """ Main page of the API """
    return render_template('products.html')

@app.route('/store')
def STORE():
    """ Main page of the API """
    return render_template('store.html')

@app.route('/RF')
def RF():
    """ Main page of the API """
    return render_template('predict.html')

@app.route('/predict',methods=['POST'])
def predict():
	'''
	For rendering results on HTML GUI
	'''
	int_features = [float(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)
	output = prediction[0]
	return render_template('result.html', prediction_text='Rp {}'.format(output))
@app.route('/predict_api',methods=['GET'])
def predict_api():
	'''
	For direct API calls trought request
	'''
	data = request.get_json(force=True)
	prediction = model.predict([np.array(list(data.values()))])
	output = prediction[0]
	return jsonify(output)
if __name__ == "__main__":
    app.run(debug=True)
