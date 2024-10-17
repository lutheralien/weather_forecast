import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_data():
    while True:
        file_path = input("Please enter the path to your Chicago weather CSV file: ")
        if file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                print("Great! The file has been loaded successfully.")
                return df
            except Exception as e:
                print(f"Oops! There was an error reading the file: {str(e)}")
                print("Please make sure the file is a valid CSV and try again.")
        else:
            print("The file name seems incorrect. Please ensure the file has a .csv extension.")

def preprocess_data(df):
    # Keep only numerical columns
    df = df.select_dtypes(include=[np.number])
    print("\nGreat! I've processed the data and kept only the numerical columns.")
    print("Here are the columns available in the dataset:", df.columns.tolist())
    return df

def select_target(df):
    while True:
        target = input("\nWhich column would you like to predict? Please enter the column name: ").strip()
        if target in df.columns:
            return target
        else:
            print(f"I'm sorry, but '{target}' is not a column in the dataset. Please try again.")

def select_features(df, target):
    available_features = [col for col in df.columns if col != target]
    print(f"\nAvailable features for predicting {target}:", available_features)
    
    features = []
    while True:
        feature = input("Please enter a feature name to use for prediction (or press Enter if you're done): ").strip()
        if feature == "":
            break
        if feature in available_features:
            features.append(feature)
        else:
            print(f"I'm sorry, but '{feature}' is not an available feature. Please try again.")
    
    if len(features) == 0:
        raise ValueError("You need to select at least one feature for prediction.")
    
    return features

def train_model(df, features, target):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")

    print("\nFeature Importances:")
    for feature, coef in zip(features, model.coef_):
        print(f"{feature}: {coef:.4f}")

    return model, mse

def predict_weather(model, features, target, mse):
    print(f"\nLet's predict {target}! Please enter the weather details:")
    input_data = {}
    
    for feature in features:
        while True:
            try:
                value = float(input(f"Enter the value for {feature}: "))
                input_data[feature] = value
                break
            except ValueError:
                print("Oops! That wasn't a valid number. Please try again.")

    input_df = pd.DataFrame([input_data])
    predicted_value = model.predict(input_df)[0]

    print(f"\nBased on the information you provided, the predicted {target} is: {predicted_value:.2f}")

    # Save prediction
    with open('chicago_weather_predictions.txt', 'a') as f:
        f.write("=" * 50 + "\n")
        f.write("New Prediction\n")
        f.write("=" * 50 + "\n")
        f.write("Input:\n")
        for feature, value in input_data.items():
            f.write(f"  {feature}: {value}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Predicted {target}: {predicted_value:.2f}\n")
        f.write(f"Mean Squared Error: {mse:.4f}\n")
        f.write("=" * 50 + "\n\n")

    print(f"I've saved this prediction to 'chicago_weather_predictions.txt' for your reference.")

def main():
    try:
        print("Welcome to the Chicago Weather Prediction Program!")
        df = load_data()
        df = preprocess_data(df)
        
        target = select_target(df)
        features = select_features(df, target)

        print(f"\nGreat! We'll use {', '.join(features)} to predict {target}.")
        model, mse = train_model(df, features, target)

        while True:
            predict_weather(model, features, target, mse)
            another = input(f"\nWould you like to predict {target} for another set of conditions? (yes/no): ").lower()
            if another != 'yes' and another != 'y':
                break

        print("\nThank you for using the Chicago Weather Prediction Program. Have a great day!")
    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        input("Press Enter to exit the program...")

if __name__ == "__main__":
    main()