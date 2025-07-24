import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target
print(data.head())
print(data.isnull().sum())
data.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in data.columns]
X = data.drop('target', axis=1)
y = data['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
plt.figure(figsize=(8, 5))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
joblib.dump(model, 'iris_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
model = joblib.load('iris_model.pkl')
scaler = joblib.load('scaler.pkl')
sample = [[5.1, 3.5, 1.4, 0.2]]  # sepal length, sepal width, petal length, petal width
sample_scaled = scaler.transform(sample)
print("Predicted class:", iris.target_names[model.predict(sample_scaled)[0]])