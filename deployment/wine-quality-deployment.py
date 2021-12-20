#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Model Deployment
# After analysing the given datasets, we have chosen and saved both the red and white wine models with the highest accuracy.
# 
# This Flask application is written to deploy these models in production so that users can supply wine parameters and obtain an estimated quality prediction.
# 
# There are two deployment models supported:
# 
# 1. Deployment to local machine
#     1. Just run this notebook in your local machine
#     
# 2. Deployment to Heroku, which needs
#     1. Procfile
#     2. requirements.txt
# 
# (search for and follow the steps to deploy the complete `deployment` folder, subfolders and contents to Heroku)

# In[ ]:


import pandas as pd
from flask import Flask, request, render_template, redirect
import pickle


# ### Load the Linear Regression  Red and White Wine Models

# In[ ]:


print(f'Creating a flask app for {__name__}')
app = Flask(__name__)


# In[ ]:


def load_model(filename):
    pklfile = open(filename, 'rb')
    lr_model = pickle.load(pklfile)
    pklfile.close()
    return lr_model


# In[ ]:


lr_model_white = load_model("models/white_wine_model.pkl")
lr_model_red = load_model("models/red_wine_model.pkl")


# ### Create a flask app

# In[ ]:


print(f'Creating a flask app for {__name__}')
app = Flask(__name__)


# ### Establish Routes for Red and White Wine Prediction

# In[ ]:


#Establish default route
@app.route('/')
def default():
    return redirect('/red_input')


# In[ ]:


# Establish red wine routes
@app.route('/red_input')
def red_input():
    return render_template('red_wine_input.html')

@app.route('/red_predict', methods =["GET","POST"])
def red_predict():
   
    if request.method == 'GET':
        va = request.args.get('va')
        sul = request.args.get('sul')
        alc =request.args.get('alc')

        data_red_wine = pd.DataFrame([[va,sul,alc]] )    
        arr_red_predict = lr_model_red.predict(data_red_wine)[0]
        arr_red_predict = int(round(arr_red_predict,0))
        
        return render_template('red_wine_quality_predict.html', va=va, sul=sul, alc=alc, arr_red_predict=arr_red_predict) ### --> HTML response
        
    return "Unsupported request method,{}".format(request.method),400


# In[ ]:


# Establish white wine routes
@app.route('/white_input')
def white_input():
    return render_template('white_wine_input.html')

@app.route('/white_predict',methods =["GET","POST"])
def white_predict():
    
    if request.method == 'GET':
        fix_acid = request.args.get('fix_acid')         
        vol_acid = request.args.get('vol_acid')
        citric_acid = request.args.get('citric_acid')
        resi_sugar  = request.args.get('resi_sugar')
        chlorides = request.args.get('chlorides')
        free_sul_o2 = request.args.get('free_sul_o2')
        total_sul_o2 = request.args.get('total_sul_o2')
        density = request.args.get('density')
        pH = request.args.get('pH')
        sulphates = request.args.get('sulphates')
        alcohol = request.args.get('alcohol')
        
        data_white_wine = pd.DataFrame([[fix_acid,vol_acid,citric_acid,resi_sugar,chlorides,free_sul_o2,total_sul_o2,density,pH,sulphates,alcohol]] )
        arr_white_predict = lr_model_white.predict(data_white_wine)[0]
        arr_white_predict = int(round(arr_white_predict,0))
        
        return render_template('white_wine_quality_predict.html',fix_acid = fix_acid, vol_acid = vol_acid,
                               citric_acid = citric_acid, resi_sugar = resi_sugar, chlorides = chlorides, 
                               free_sul_o2 = free_sul_o2, total_sul_o2 = total_sul_o2, density = density, 
                               pH = pH, sulphates = sulphates, alcohol = alcohol, 
                               arr_white_predict=arr_white_predict)
        
    return "Unsupported request method,{}".format(request.method),400


# ### Run the Flask Web Server

# In[ ]:


if __name__ == '__main__':
    app.run()

