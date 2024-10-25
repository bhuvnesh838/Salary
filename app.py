from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Pre-calculated values (use your own values here)
weights = np.array([-3.3362201889985435e-16, 0.9782416184887595])  # Use the actual values from your trained model
mean_experience = 5.413333333333332  # Replace with your calculated values
std_experience = 2.790189161249745
mean_salary = 76004.0
std_salary = 26953.65024877583

def predict_salary(years_experience, weights, mean_experience, std_experience, mean_salary, std_salary):
    standardized_experience = (years_experience - mean_experience) / std_experience
    X_new = np.array([1, standardized_experience])
    standardized_salary = np.dot(X_new, weights)
    salary = standardized_salary * std_salary + mean_salary
    return salary

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# API route to handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    years_experience = float(data['years_experience'])
    predicted_salary = predict_salary(years_experience, weights, mean_experience, std_experience, mean_salary, std_salary)
    return jsonify({'predicted_salary': f"${predicted_salary:.2f}"})

if __name__ == '__main__':
    app.run(debug=True)
