📊 AI-Based Employee Prediction (2018–2030)
This project uses machine learning (Linear Regression) to predict the number of employees in various companies based on historical data. The model is trained using data from 2018 to 2023, tested on 2024–2025, and then used to predict future values from 2026 to 2030.

🚀 Features
📈 Trains an AI model using company employee data (2018–2023)

🧪 Evaluates model accuracy on 2024–2025 data using Mean Absolute Error (MAE)

🔮 Predicts employee count for 2026–2030

📤 Outputs:

predicted_2026_to_2030.csv — Future predictions

employee_growth_percent.csv — Year-on-year % change

📊 Plots trends for visual analysis

🛠️ Requirements
Python 3.x

pandas

numpy

scikit-learn

matplotlib

Install them using:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib
📂 File Structure
data.csv — Input file containing employee data from 2018 to 2025

prediction.py — Main script for training, prediction, evaluation, and output

predicted_2026_to_2030.csv — AI-generated future employee predictions

employee_growth_percent.csv — Yearly percentage growth per company

🧠 How It Works
Loads employee data from data.csv

Trains a Linear Regression model using data from 2018 to 2023

Tests model performance on 2024–2025 (optional step)

Predicts data for 2026–2030

Saves both predicted values and % increase/decrease year-by-year

Displays visual graphs for each company

