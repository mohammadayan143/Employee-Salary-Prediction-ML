from flask import Flask, render_template, request
import joblib
import os
import pandas as pd


app = Flask(__name__)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


model = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "salary_model.pkl"
    )
)


encoders = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "encoders.pkl"
    )
)


# Home Page

@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route

@app.route("/predict", methods=["POST"])
def predict():

    # Get data from HTML form

    age = int(request.form["age"])

    gender = request.form["gender"]

    education = request.form["education"]

    job = request.form["job"]

    experience = float(
        request.form["experience"]
    )


    # Create dataframe same as training data

    input_data = pd.DataFrame({

        "Age": [age],

        "Gender": [gender],

        "Education Level": [education],

        "Job Title": [job],

        "Years of Experience": [experience]

    })


    # Clean inputs

    input_data["Gender"] = (
        input_data["Gender"]
        .str.strip()
        .str.title()
    )


    input_data["Education Level"] = (
        input_data["Education Level"]
        .str.strip()
    )


    input_data["Job Title"] = (
        input_data["Job Title"]
        .str.strip()
        .str.title()
    )


    print(input_data)


    # Convert text values into numbers

    for col in [
        "Gender",
        "Education Level",
        "Job Title"
    ]:

        input_data[col] = encoders[col].transform(
            input_data[col]
        )


    # Prediction

    prediction = model.predict(input_data)


    salary = round(prediction[0], 2)


    return render_template(
        "index.html",
        prediction_text=f"Predicted Salary: ₹ {salary}"
    )


if __name__ == "__main__":
    app.run(debug=True)