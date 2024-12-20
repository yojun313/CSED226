import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load train and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# Preprocessing
def preprocess_data(df):
    df = df.copy()
    X = df.drop(['ID', 'age_group'], axis=1)
    y = df['age_group']

    # Encoding target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, label_encoder


X, y, label_encoder = preprocess_data(train_data)
X_test, _, _ = preprocess_data(test_data)

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# Define models

def random_forest_model(X_train, y_train, X_val, y_val):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return model, accuracy


def logistic_regression_model(X_train, y_train, X_val, y_val):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return model, accuracy


def svm_model(X_train, y_train, X_val, y_val):
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return model, accuracy


def knn_model(X_train, y_train, X_val, y_val):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return model, accuracy


# Evaluate models
models = {
    'Random Forest': random_forest_model,
    'Logistic Regression': logistic_regression_model,
    'SVM': svm_model,
    'KNN': knn_model
}

results = {}
best_model = None
best_accuracy = 0

for name, model_func in models.items():
    model, accuracy = model_func(X_train, y_train, X_val, y_val)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.2f}")
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy

# Make predictions with the best model
y_test_pred = best_model.predict(X_test)

# Save predictions
test_data['age_group'] = label_encoder.inverse_transform(y_test_pred)
output = test_data[['ID', 'age_group']]
output.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")
