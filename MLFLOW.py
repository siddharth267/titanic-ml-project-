import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Titanic Dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Handle Missing Values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)

# Encode Categorical Data
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Feature Selection
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set Custom Experiment Name
mlflow.set_experiment("Titanic_RF_Experiment")

# Hyperparameter tuning list
n_estimators_list = [50, 100, 150, 200]

# Create folder for confusion matrices
os.makedirs("confusion_matrices", exist_ok=True)

for n in n_estimators_list:
    with mlflow.start_run(run_name=f"RandomForest_Run_{n}"):
        # Train Model
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Print Scores
        print(f"\nRun for n_estimators={n}")
        print("Accuracy:", accuracy)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Log Parameters and Metrics
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n)
        mlflow.log_metric("accuracy", accuracy)

        # Confusion Matrix Artifact
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (n_estimators={n})")
        cm_file = f"confusion_matrices/cm_{n}.png"
        plt.savefig(cm_file)
        plt.close()
        mlflow.log_artifact(cm_file)

        # Log Model with Signature
        input_example = X_test.iloc[0:1]
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, "random_forest_model", input_example=input_example, signature=signature)

        print(f"Run for n_estimators={n} logged successfully.\n")

print("\n✅ All runs completed and logged to MLflow.")
