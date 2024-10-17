import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = input("Enter the path to your CSV file: ")
df = pd.read_csv(file_path)

# Remove the Date_Time column if it exists
if 'Date_Time' in df.columns:
    df = df.drop('Date_Time', axis=1)

# Print information about the dataset
print("Dataset Information:")
print(df.info())

# Create a binary target variable: 1 if it rains, 0 if it doesn't
df['Rain'] = (df['Precipitation_mm'] > 0).astype(int)

print("\nValue counts for 'Rain':")
print(df['Rain'].value_counts(normalize=True))

# If there's only one class, adjust the threshold
if len(df['Rain'].unique()) == 1:
    print("\nAdjusting rain threshold...")
    df['Rain'] = (df['Precipitation_mm'] > 0.1).astype(int)
    print("New value counts for 'Rain':")
    print(df['Rain'].value_counts(normalize=True))

# Select features for prediction
features = ['Temperature_C', 'Humidity_pct', 'Wind_Speed_kmh']
X = df[features]
y = df['Rain']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining set 'Rain' distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest set 'Rain' distribution:")
print(y_test.value_counts(normalize=True))

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Train a Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Make predictions on the test set for both models
rf_pred = rf_classifier.predict(X_test_scaled)
log_pred = log_reg.predict(X_test_scaled)

# Print classification reports
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, log_pred))

# Print confusion matrices
rf_cm = confusion_matrix(y_test, rf_pred)
log_cm = confusion_matrix(y_test, log_pred)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

print("\nLogistic Regression Confusion Matrix:")
print(log_cm)

# Calculate and print additional statistics
total_samples = len(y_test)
rf_correct = (y_test == rf_pred).sum()
log_correct = (y_test == log_pred).sum()
rf_accuracy = rf_correct / total_samples
log_accuracy = log_correct / total_samples
rain_frequency = y_test.mean()

print(f"\nTotal Samples: {total_samples}")
print(f"Random Forest Correct Predictions: {rf_correct}")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"Logistic Regression Correct Predictions: {log_correct}")
print(f"Logistic Regression Accuracy: {log_accuracy:.2f}")
print(f"Rain Frequency: {rain_frequency:.2f}")

# Feature importance for Random Forest
rf_importance = pd.DataFrame({'feature': features, 'importance': rf_classifier.feature_importances_})
rf_importance = rf_importance.sort_values('importance', ascending=False)
print("\nRandom Forest Feature Importance:")
print(rf_importance)

# Logistic Regression coefficients
log_coef = pd.DataFrame({'feature': features, 'coefficient': log_reg.coef_[0]})
log_coef['abs_coefficient'] = abs(log_coef['coefficient'])
log_coef = log_coef.sort_values('abs_coefficient', ascending=False)
print("\nLogistic Regression Coefficients:")
print(log_coef)

# Visualize feature importance and coefficients
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.barplot(x='importance', y='feature', data=rf_importance, ax=ax1)
ax1.set_title('Random Forest Feature Importance')

sns.barplot(x='coefficient', y='feature', data=log_coef, ax=ax2)
ax2.set_title('Logistic Regression Coefficients')

plt.tight_layout()
plt.show()

# Visualize the confusion matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Random Forest Confusion Matrix')
ax1.set_ylabel('Actual')
ax1.set_xlabel('Predicted')

sns.heatmap(log_cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('Logistic Regression Confusion Matrix')
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')

plt.tight_layout()
plt.show()

# Function to predict rain probability for user input
def predict_rain_probability(temperature, humidity, wind_speed):
    user_input = np.array([[temperature, humidity, wind_speed]])
    user_input_scaled = scaler.transform(user_input)
    rf_prob = rf_classifier.predict_proba(user_input_scaled)[0][1]
    log_prob = log_reg.predict_proba(user_input_scaled)[0][1]
    return rf_prob, log_prob

# Example usage
print("\nEnter weather conditions to predict rain probability:")
temp = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
wind_speed = float(input("Wind Speed (km/h): "))

rf_probability, log_probability = predict_rain_probability(temp, humidity, wind_speed)
print(f"Random Forest Probability of rain: {rf_probability:.2f}")
print(f"Logistic Regression Probability of rain: {log_probability:.2f}")
print(f"Random Forest: {'It will likely rain' if rf_probability > 0.5 else 'It will likely not rain'}")
print(f"Logistic Regression: {'It will likely rain' if log_probability > 0.5 else 'It will likely not rain'}")