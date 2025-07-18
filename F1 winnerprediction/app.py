from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('f1_winner_model.pkl')


# Load dummy columns used during training
with open('feature_columns.txt') as f:


    feature_columns = f.read().splitlines()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    grid = int(request.form['grid'])
    qualifying_position = int(request.form['qualifying_position'])
    past_wins_at_circuit = int(request.form['past_wins_at_circuit'])
    constructor_name = request.form['constructor_name']
    circuit_name = request.form['circuit_name']

    # Build input DataFrame
    input_dict = {
        'grid': grid,
        'qualifying_position': qualifying_position,
        'past_wins_at_circuit': past_wins_at_circuit,
    }

    for col in feature_columns:
        if col not in input_dict:
            input_dict[col] = 1 if (col == f'constructor_name_{constructor_name.lower()}' or col == f'circuit_name_{circuit_name}') else 0

    input_df = pd.DataFrame([input_dict])
    prob = model.predict_proba(input_df)[0][1]
    
    return render_template('result.html', prediction=round(prob * 100, 2), constructor=constructor_name)

if __name__ == '__main__':
    app.run(debug=True)
