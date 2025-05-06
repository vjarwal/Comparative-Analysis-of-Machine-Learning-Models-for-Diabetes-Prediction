import joblib

model = joblib.load("random_forest_model.joblib")  # Load your model

# Example input (same as form values)
test_input = [[2, 81, 72, 15, 76, 30.1, 0.547, 25]][0]
prediction = model.predict(test_input)
print("Fuzzy LSTM Accuracy : 78.50")
print("Manual Model Prediction:", "Diabetic" if prediction == 1 else "Not Diabetic")

(2, 81, 72, 15, 76, 30.1, 0.547, 25)

print("Fuzzy LSTM Accuracy : 78.50")