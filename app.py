
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model and preprocessing objects
model = joblib.load("model/xgboost_real.pkl")
threshold = joblib.load("model/xgboost_threshold.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            exposure = float(request.form["Exposure"])
            area = request.form["Area"]
            veh_power = float(request.form["VehPower"])
            veh_age = float(request.form["VehAge"])
            driv_age = float(request.form["DrivAge"])
            bonus_malus = float(request.form["BonusMalus"])
            veh_brand = request.form["VehBrand"]
            veh_gas = request.form["VehGas"]
            region = request.form["Region"]
            log_density = float(request.form["LogDensity"])

            # Validate numerical inputs
            if not 0 <= exposure <= 1:
                raise ValueError("Exposure must be between 0 and 1.")

            if veh_power < 0 or veh_age < 0 or driv_age < 18:
                raise ValueError("Please enter valid vehicle and driver details.")

            # Prepare numerical features in scaler's training order
            numerical_input = pd.DataFrame([{
                "Exposure": exposure,
                "VehPower": veh_power,
                "VehAge": veh_age,
                "DrivAge": driv_age,
                "BonusMalus": bonus_malus,
                "LogDensity": log_density
            }])

            # Prepare categorical features
            categorical_input = pd.DataFrame([{
                "Area": area,
                "VehBrand": veh_brand,
                "VehGas": veh_gas,
                "Region": region
            }])

            scaled_numerical = pd.DataFrame(
                scaler.transform(numerical_input),
                columns=numerical_input.columns
            )

            encoded_categorical = pd.DataFrame(
                encoder.transform(categorical_input),
                columns=encoder.get_feature_names_out()
            )

            model_input = pd.concat(
                [scaled_numerical, encoded_categorical], axis=1
            )
            model_input = model_input.reindex(
                columns=feature_columns, fill_value=0
            )

            probability = float(model.predict_proba(model_input)[0, 1])
            prediction = int(probability >= threshold)

        except (ValueError, KeyError) as exc:
            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
