import pickle, bz2
from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("modeljob.pkl")

# Define the StandardScaler object
scaler = StandardScaler()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/home")
def show_home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the form
    input_data = [float(x) for x in request.form.values()]
    # Define the 20 other values statically
    static_data = [0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563]
    n = len(input_data)
    sample_std_dev = np.std(input_data, ddof = 1)
    sem_values = [sample_std_dev / np.sqrt(n)] * n

    worst_mean_values = [input_data[i] for i in np.argsort(sem_values)[-10:]]
    # input_data.extend(sem_values)
    # input_data.extend(worst_mean_values)

    # Identify the 10 mean values with the largest SEM values
    
    print(input_data)
    # change the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for one data point
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # Fit the StandardScaler object on some data before using it to transform the input data
    scaler.fit(input_data_reshaped)

    # standardizing the input data
    input_data_std = scaler.transform(input_data_reshaped)

    # Use the pre-trained model to make a prediction
    prediction = model.predict(np.array(input_data_std).reshape(1, -1))[0]

    prediction_label = [np.argmax(prediction)]
    print(prediction_label)
    print(prediction_label[0])
    # Return the prediction as a string
    return render_template("result.html", result=prediction_label[0])

if __name__ == "__main__":
    app.run(debug=True, port=3004)

#11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888