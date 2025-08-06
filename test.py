# Step 1: Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load Titanic Dataset from URL
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')



# Step 4: Handle Missing Values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)

# Step 5: Encode Categorical Data
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Step 6: Feature Selection
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# Step 7: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train the Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 9: Evaluate the Model
y_pred = model.predict(X_test)
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Ask a Question (Predict for New Input)
print("\n--- Ask a question to the model ---")

# Taking user input
pclass = int(input("Enter Pclass (1 = 1st, 2 = 2nd, 3 = 3rd): "))
sex = int(input("Enter Sex (0 = male, 1 = female): "))
age = float(input("Enter Age: "))
sibsp = int(input("Enter number of siblings/spouses aboard: "))
parch = int(input("Enter number of parents/children aboard: "))
fare = float(input("Enter Fare: "))
embarked_input = input("Enter Embarked location (C, Q, or S): ").upper()

# Encode Embarked as per model
embarked_Q = 1 if embarked_input == 'Q' else 0
embarked_S = 1 if embarked_input == 'S' else 0

# Create DataFrame for new input
new_passenger = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked_Q': embarked_Q,
    'Embarked_S': embarked_S
}])

# Make prediction
prediction = model.predict(new_passenger)

# Show result
if prediction[0] == 1:
    print("\n🎉 Prediction: This passenger would SURVIVE.")
else:
    print("\n☠️ Prediction: This passenger would NOT survive.")
