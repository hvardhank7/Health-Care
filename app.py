#Important Modules
from flask import Flask, render_template, url_for, flash, redirect
import joblib
from flask import request
import numpy as np
#from this import SQLAlchemy
app=Flask(__name__,template_folder='template')



# RELATED TO THE SQL DATABASE
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'


@app.route("/home.html")
def home():
    return render_template("home.html")



@app.route("/cancer.html")
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes.html")
def diabetes():
    #if form.validate_on_submit():
    return render_template("diabetes.html")

@app.route("/heart.html")
def heart():
    return render_template("heart.html")


@app.route("/liver.html")
def liver():
    #if form.validate_on_submit():
    return render_template("liver.html")

@app.route("/kidney.html")
def kidney():
    #if form.validate_on_submit():
    return render_template("kidney.html")


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):#Diabetes
        loaded_model = joblib.load("diabetes_model.pkl")
        result = loaded_model.predict(to_predict)
    elif(size==5):#Cancer
        loaded_model = joblib.load("cancer_model.pkl")
        result = loaded_model.predict(to_predict)
    elif(size==7):#Kidney
        loaded_model = joblib.load("kidney_model.pkl")
        result = loaded_model.predict(to_predict)
    elif(size==7):#Liver
        loaded_model = joblib.load("liver_model.pkl")
        result = loaded_model.predict(to_predict)
    elif(size==11):#Heart
        loaded_model = joblib.load("liver_model.pkl")
        result =loaded_model.predict(to_predict)
    return result[0]

@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if(len(to_predict_list)==5):#Cancer
            result = ValuePredictor(to_predict_list,5)
        elif(len(to_predict_list)==7):#Daiabtes
            result = ValuePredictor(to_predict_list,7)
        elif(len(to_predict_list)==7):
            result = ValuePredictor(to_predict_list,7)
        elif(len(to_predict_list)==7):
            result = ValuePredictor(to_predict_list,7)
          
        elif(len(to_predict_list)==11):
            result = ValuePredictor(to_predict_list,11)
    if(int(result)==1):
        prediction = " Sorry you chances of getting the disease. Kindly consult to the doctor. Thank You "
    else:
        prediction = " You are Healthy !!" 
    return(render_template("result.html", prediction=prediction))


if __name__ == "__main__":
    app.run(debug=True)
