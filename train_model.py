import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

print("Loading dataset...")
df = pd.read_csv('Ambonga_MobilePricing.csv')

X = df.drop('price_range', axis=1)
y = df['price_range']

print("Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

print("Saving model and scaler...")
joblib.dump(model, 'mobile_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")
