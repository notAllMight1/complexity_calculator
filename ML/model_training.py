import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from ML.Feature_Engineering import extract_features
import joblib

# Load dataset from CSV file
def load_data(file_path):
    """
    Load dataset (CSV) and return the DataFrame.
    """
    return pd.read_csv(file_path)

# Convert code snippets into feature vectors using extract_features
def create_feature_matrix(df):
    """
    Convert code snippets into feature vectors using extract_features.
    """
    features_list = []
    labels = []
    
    # Iterate through the DataFrame and extract features for each code snippet
    for index, row in df.iterrows():
        features = extract_features(row["code"], row["language"])
        features_list.append(features)
        labels.append(row["time_complexity"])  # Assume target column is 'Time Complexity'
    
    # Convert feature dictionaries into a matrix using DictVectorizer
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(features_list)
    y = labels
    
    return X, y, vectorizer

# Train an ensemble model using Random Forest, Logistic Regression, and SVM
def train_ensemble_model(X_train, y_train):
    """
    Train an ensemble model using Random Forest, Logistic Regression, and SVM.
    Perform K-Fold Cross Validation with regularization.
    """
    # Create individual models with regularization
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    svm = SVC(kernel='linear', C=0.1, random_state=42)
    
    # Create an ensemble of models (Voting Classifier)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('svm', svm)], voting='hard')
    
    # Train the model on the training dataset
    ensemble.fit(X_train, y_train)
    
    return ensemble

def main():
    """
    Main function to load data, create feature matrix, train model, and evaluate.
    """
    # Load dataset
    df = load_data("./Dataset/Normal3lang.csv")  # Replace with the actual path
    
    # Create feature matrix
    X, y, vectorizer = create_feature_matrix(df)
    
    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train ensemble model
    model = train_ensemble_model(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy}")
    
    # Save the trained model and vectorizer
    joblib.dump(model, 'ensemble_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    print("Model and vectorizer have been saved successfully!")

# Load the trained model and vectorizer for predictions
def predict_time_complexity(code_snippet, language, model_path, vectorizer_path):
    """
    Given a code snippet and language, predict the time complexity using the trained model.
    Now accepts model_path and vectorizer_path as arguments to load them dynamically.
    """
    # Load the trained model and vectorizer from the provided paths
    ensemble_model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Extract features from the code snippet
    features = extract_features(code_snippet, language)

    # Transform the features into the same format the model was trained on
    features_vector = vectorizer.transform([features])

    # Make prediction
    predicted_time_complexity = ensemble_model.predict(features_vector)

    return predicted_time_complexity[0]  # Return the predicted time complexity

if __name__ == "__main__":
    main()
