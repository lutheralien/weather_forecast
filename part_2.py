import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Get file path from console
file_path = input("Enter the path to your CSV file: ")

# Load the dataset
df = pd.read_csv(file_path)

# Get features from console
print("Available columns:", df.columns.tolist())
features = input("Enter the feature column names separated by commas: ").split(',')
features = [f.strip() for f in features]

# Prepare feature matrix (X) and target vector (y)
X = df[features]
df['Rain'] = (df['Precipitation_mm'] >= 2.3).astype(int)
y = df['Rain']

print("\nTarget variable distribution:")
print(y.value_counts(normalize=True))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate Logistic Regression
logreg = LogisticRegression(random_state=42, class_weight='balanced')
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)

# Added print statements for model performance
print("\nLogistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict rain probability
def predict_rain_probability(*feature_values):
    user_input = np.array([feature_values])
    user_input_scaled = scaler.transform(user_input)
    probability = logreg.predict_proba(user_input_scaled)[0][1]
    return probability

# Get user input for prediction
print("\nEnter values for prediction:")
user_values = []
for feature in features:
    value = float(input(f"Enter value for {feature}: "))
    user_values.append(value)

rain_prob = predict_rain_probability(*user_values)
print(f"\nProbability of Rain (>= 2.3mm): {rain_prob:.2f}")

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'coefficient': abs(logreg.coef_[0])})
print("\nFeature Importance:")
print(feature_importance.sort_values('coefficient', ascending=False))