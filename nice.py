import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from typing import List, Dict, Tuple, Optional
import os

def display_message(message: str, message_type: str = "info") -> None:
    """Display formatted messages."""
    colors = {
        "info": "\033[94m",  # Blue
        "success": "\033[92m",  # Green
        "warning": "\033[93m",  # Yellow
        "error": "\033[91m"  # Red
    }
    end_color = "\033[0m"
    print(f"{colors.get(message_type, '')}{message}{end_color}")

class ChicagoRainfallPredictor:
    def __init__(self, file_path: str = "weather_data.csv"):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        self.features: List[str] = ['Temperature_C', 'Humidity_pct', 'Wind_Speed_kmh']
        self.model: Optional[DecisionTreeClassifier] = None

    def load_data(self) -> None:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        
        try:
            self.df = pd.read_csv(self.file_path)
            display_message("Data loaded successfully.", "success")
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")
        except pd.errors.ParserError:
            raise ValueError("Unable to parse the CSV file. Please check the file format.")

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_columns = set(self.features + ['Precipitation_mm']) - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        self.df['Heavy_Rainfall'] = (self.df['Precipitation_mm'] > 30).astype(int)
        X = self.df[self.features]
        y = self.df['Heavy_Rainfall']
        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        display_message(f"Model trained. Accuracy: {accuracy:.2f}", "success")

    def predict_rainfall(self, user_data: Dict[str, float]) -> str:
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        user_input = [user_data[feature] for feature in self.features]
        prediction = self.model.predict([user_input])[0]
        
        if prediction == 1:
            return "Heavy rainfall (>30mm) is predicted."
        else:
            return "No heavy rainfall predicted (<=30mm)."

def get_user_input(features: List[str]) -> Dict[str, float]:
    user_data = {}
    display_message("Enter current weather conditions:", "info")
    for feature in features:
        while True:
            try:
                value = float(input(f"  {feature}: "))
                user_data[feature] = value
                break
            except ValueError:
                display_message("Please enter a valid number.", "warning")
    return user_data

def main():
    predictor = ChicagoRainfallPredictor()  # Using default file_path
    
    try:
        predictor.load_data()
        X, y = predictor.prepare_data()
        predictor.train_model(X, y)
        
        while True:
            user_data = get_user_input(predictor.features)
            result = predictor.predict_rainfall(user_data)
            display_message(result, "info")
            
            again = input("Make another prediction? (y/n): ").lower()
            if again not in ['y', 'yes']:
                break
        
        display_message("Thank you for using the Chicago Rainfall Prediction program!", "success")
    
    except FileNotFoundError as e:
        display_message(f"Error: {e}", "error")
        display_message("Please ensure the 'weather_data.csv' file is in the same directory as this script.", "warning")
    except ValueError as e:
        display_message(f"Error: {e}", "error")
        display_message("Please check your data file and ensure it contains the required columns.", "warning")
    except Exception as e:
        display_message(f"An unexpected error occurred: {str(e)}", "error")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()