from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load your model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the HTML form
    age = float(request.form['Age'])
    mileage = float(request.form['Mileage'])
    power = float(request.form['Power'])

    # Create DataFrame for prediction
    input_df = pd.DataFrame([[age, mileage, power]],
                            columns=['Age', 'Mileage', 'Power'])

    prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction_text=f'Estimated Car Price: ${prediction:,.2f}')


if __name__ == "__main__":
    app.run(debug=True)