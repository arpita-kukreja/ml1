
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  

# Step 1: Load dataset
def load_data(path):
    # TODO: Read the CSV file using pandas
    df = pd.read_csv(path)  
    
    # TODO: Select features (Hours_Studied should be a DataFrame, not Series)
    X = df[['Hours_Studied']]  
    
    # TODO: Select target column (Exam_Pass)
    y = df['Exam_Pass']  
    
    return X, y


# Step 2: Train logistic regression
def train_model(X, y):
    # TODO: Split the dataset into training and testing (30% test, random_state=42)
    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=42)  
    
    # TODO: Create logistic regression model
    model = LogisticRegression()  
    
    # TODO: Train the model
    model.fit(X_train, y_train)  
    
    return model, X_test, y_test


# Step 3: Evaluate predictions
def evaluate_model(model, X_test, y_test):
    # TODO: Generate predictions on the test set
    y_pred = model.predict(X_test)  
    
    # TODO: Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)  
    
    # TODO: Calculate Precision
    precision = precision_score(y_test, y_pred)  
    
    # TODO: Calculate Recall
    recall = recall_score(y_test, y_pred)  
    
    # TODO: Calculate F1 Score
    f1 = f1_score(y_test, y_pred)  
    
    return accuracy, precision, recall, f1


# Step 4: Display results
def display_results(model, accuracy, precision, recall, f1):
    print("Model used:", type(model).__name__)  
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)



# Step 5: Main workflow
if __name__ == "__main__":
    # TODO: Provide the dataset path
    path = "students.csv"  
    
    # TODO: Load dataset
    X, y = load_data(path)
    
    # TODO: Train logistic regression
    model, X_test, y_test = train_model(X, y)
    
    # TODO: Evaluate metrics
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    
    # TODO: Print results
    display_results(model, accuracy, precision, recall, f1)
# Output
# Model used: LogisticRegression
# Accuracy: 1.0
# Precision: 1.0
# Recall: 1.0
# F1-score: 1.0
