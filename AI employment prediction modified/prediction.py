import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data.csv")

# Define year ranges
train_years = list(range(2018, 2024))   # Training: 2018–2023
test_years = list(range(2024, 2026))    # Testing: 2024–2025
predict_years = list(range(2026, 2031)) # Prediction: 2026–2030

X_train = np.array(train_years).reshape(-1, 1)

predicted_data = []

for i, row in df.iterrows():
    company = row["Company"]

    # Training data
    y_train = np.array([row[f"Emp({year})"] for year in train_years])

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test data (optional evaluation)
    y_test = np.array([row[f"Emp({year})"] for year in test_years])
    X_test = np.array(test_years).reshape(-1, 1)
    y_test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    print(f"{company} Test MAE (2024–2025): {mae:.2f}")

    # Future prediction
    X_future = np.array(predict_years).reshape(-1, 1)
    y_pred = model.predict(X_future)

    # Save predictions
    prediction_dict = {"Company": company}
    for year, pred in zip(predict_years, y_pred):
        prediction_dict[f"Emp({year})"] = int(pred)
    predicted_data.append(prediction_dict)

    # Plot line
    plt.plot(train_years + test_years + predict_years,
             list(y_train) + list(y_test) + list(y_pred),
             marker='o', label=company)

# Show plot
plt.title("Employee Prediction (2018–2030)")
plt.xlabel("Year")
plt.ylabel("Employees")
plt.legend()
plt.grid(True)
plt.show()

# Save predicted data to CSV
pred_df = pd.DataFrame(predicted_data)
pred_df.to_csv("predicted_2026_to_2030.csv", index=False)
print("\nPredicted Data (2026–2030):")
print(pred_df)

# Append predictions to original data
full_data = df.copy()
for pred_row in predicted_data:
    company_name = pred_row["Company"]
    for year in predict_years:
        full_data.loc[full_data["Company"] == company_name, f"Emp({year})"] = pred_row[f"Emp({year})"]

# Year-over-year % change
all_years = list(range(2018, 2031))
change_data = []

for idx, row in full_data.iterrows():
    company = row["Company"]
    changes = {"Company": company}

    for i in range(1, len(all_years)):
        year_prev = all_years[i-1]
        year_curr = all_years[i]
        prev_val = row.get(f"Emp({year_prev})", None)
        curr_val = row.get(f"Emp({year_curr})", None)

        if prev_val and curr_val and prev_val != 0:
            change = ((curr_val - prev_val) / prev_val) * 100
            changes[f"Emp({year_curr})_change"] = f"{change:.2f}%"
        else:
            changes[f"Emp({year_curr})_change"] = "N/A"

    change_data.append(changes)

# Save % change to CSV
change_df = pd.DataFrame(change_data)
change_df.to_csv("employee_growth_percent.csv", index=False)

print("\nYear-on-Year Employee % Change:")
print(change_df)
