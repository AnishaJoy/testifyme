from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/live-update")
def update():
    return render_template("live-update.html")

@app.route("/symptoms")
def symptoms():
    return render_template("symptoms.html")

@app.route("/prediction", methods=['POST'])
def result():
    int_features = [int(x) for x in request.form.values()]
    final = np.asarray(int_features)
    input_data_reshaped = final.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    if int(prediction) == int(1):
        return render_template('result.html', pred='You are COVID positive')
    else:
        return render_template('result.html', pred='You are COVID negative')

if __name__ == "__main__":
    app.run(debug=True)
