import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return df

def prepare_data(df):
    # Create target variable: 1 if rainfall > 30mm, 0 otherwise
    df['Heavy_Rainfall'] = (df['Precipitation_mm'] > 30).astype(int)
    
    # Select features (you can modify this list based on your dataset)
    features = ['Temperature_C', 'Humidity_pct', 'Wind_Speed_kmh']
    
    X = df[features]
    y = df['Heavy_Rainfall']
    
    return X, y, features

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model

def get_user_input(features):
    user_data = {}
    for feature in features:
        while True:
            try:
                value = float(input(f"Enter {feature}: "))
                user_data[feature] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    return user_data

def predict_rainfall(model, user_data, features):
    user_input = [user_data[feature] for feature in features]
    prediction = model.predict([user_input])[0]
    
    if prediction == 1:
        return "The rainfall is predicted to be greater than 30mm."
    else:
        return "The rainfall is predicted to be 30mm or less."

def main():
    file_path = input("Enter the path to your Chicago weather CSV file: ")
    
    try:
        df = load_data(file_path)
        X, y, features = prepare_data(df)
        model = train_model(X, y)
        
        while True:
            print("\nEnter weather conditions:")
            user_data = get_user_input(features)
            result = predict_rainfall(model, user_data, features)
            print(result)
            
            again = input("Do you want to make another prediction? (yes/no): ").lower()
            if again != 'yes' and again != 'y':
                break
        
        print("Thank you for using the Chicago Rainfall Prediction program!")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()