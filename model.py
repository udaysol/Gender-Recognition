import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler

# Load data
X_train = pd.read_csv('x_train.csv')
X_test = pd.read_csv('x_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test_scaled)
print("SVM Model Accuracy:", accuracy_score(y_test, y_pred))
print('\nSVM Classification Report:\n', classification_report(y_test, y_pred))
print('\nSVM Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print("SVM Classification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(svm_model, 'svm_gender.pkl')
