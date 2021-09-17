from flask import Flask, render_template,request
import pickle
import numpy as np

filename="diabetes-prediction-rfc-model.pkl"
randomforest=pickle.load(open(filename,'rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route("/predict",methods=["POST"])
def predict():
    if request.method=="POST":
        #features=[float(x) for x in request.form.values() ]
        preg=int(request.form['pregnancies'])
        glucose=int(request.form['glucose'])
        bp=int(request.form['bloodpressure'])
        st=int(request.form['skinthickness'])
        insulin=int(request.form['insulin'])
        bmi=float(float(request.form['bmi']))
        dfp=float(request.form['dpf'])
        age=int(request.form['age'])
        data=[np.array([preg,glucose,bp,st,insulin,bmi,dfp,age])]
        # data=[np.array(features)]
        my_prediction=randomforest.predict(data)

        if my_prediction==1:
            output="you have diabetes";
        elif my_prediction==0:
            output="you dont have diabetes"

        return render_template("result.html",prediction_text=f"The prediction is that {output}")

if __name__=='__main__':
    app.run(debug=True)

