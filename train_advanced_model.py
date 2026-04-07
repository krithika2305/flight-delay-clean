import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Define Features & Synthesize Data
N = 5000
np.random.seed(42)

data = {
    'Month': np.random.randint(1, 13, N),
    'DayOfWeek': np.random.randint(1, 8, N),
    'Distance': np.random.randint(100, 3000, N),
    'CRSDep_MIN': np.random.randint(0, 1440, N),
    
    # New features
    'Temperature': np.random.uniform(20, 100, N),
    'WindSpeed': np.random.uniform(0, 50, N),
    'Visibility': np.random.uniform(0.1, 10, N),
    'PreviousDelay': np.random.exponential(15, N), # mostly low, some high
    
    'Reporting_Airline': np.random.choice(["DL", "OO", "UA", "WN", "Other"], N),
    'Origin': np.random.choice(["CLT", "DEN", "DFW", "IAH", "LAX", "ORD", "PHX", "SFO", "Other"], N),
    'Dest': np.random.choice(["CLT", "DEN", "DFW", "IAH", "LAX", "ORD", "PHX", "SFO", "Other"], N),
    
    # New categorical features
    'WeatherCondition': np.random.choice(["Clear", "Rain", "Fog", "Storm"], N, p=[0.7, 0.15, 0.05, 0.1]),
    'TrafficLevel': np.random.choice(["Low", "Medium", "High"], N),
    'TimeCategory': np.random.choice(["Morning", "Afternoon", "Evening", "Night"], N)
}

df = pd.DataFrame(data)

# Create a realistic target based on features
delay_prob = np.zeros(N)
delay_prob += df['PreviousDelay'] * 0.01
delay_prob += (df['WeatherCondition'] == 'Storm') * 0.3
delay_prob += (df['WeatherCondition'] == 'Rain') * 0.1
delay_prob += (df['WeatherCondition'] == 'Fog') * 0.15
delay_prob += (df['TrafficLevel'] == 'High') * 0.1
delay_prob += (df['Visibility'] < 2.0) * 0.2
delay_prob += (df['WindSpeed'] > 30) * 0.1

delay_prob = np.clip(delay_prob, 0, 1)
df['delayed'] = np.random.binomial(1, delay_prob)

# 2. Metadata Extraction
top_k_maps = {
    "Reporting_Airline": ["DL", "OO", "UA", "WN"],
    "Origin": ["CLT", "DEN", "DFW", "IAH", "LAX", "ORD", "PHX", "SFO"],
    "Dest": ["CLT", "DEN", "DFW", "IAH", "LAX", "ORD", "PHX", "SFO"],
    "WeatherCondition": ["Clear", "Rain", "Fog", "Storm"],
    "TrafficLevel": ["Low", "Medium", "High"],
    "TimeCategory": ["Morning", "Afternoon", "Evening", "Night"]
}

num_features = ["Month", "DayOfWeek", "Distance", "CRSDep_MIN", "Temperature", "WindSpeed", "Visibility", "PreviousDelay"]

# 3. Preprocess for training
for col, top_list in top_k_maps.items():
    df[col] = df[col].apply(lambda x: x if x in top_list else "Other")

# One-hot encode
X = df.drop(columns=['delayed'])
y = df['delayed']

for col in top_k_maps.keys():
    dummies = pd.get_dummies(X[col], prefix=col)
    X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

feature_cols = list(X.columns)

# 4. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Random Forest Classifier on synthetic dataset...")
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f"Validation Accuracy: {acc:.4f}")

# 5. Save Outputs
print("Saving model to FlightDelayPredictionModel.pkl...")
joblib.dump(clf, "FlightDelayPredictionModel.pkl")

print("Saving model_feature_columns.xls...")
pd.Series(feature_cols).to_csv("model_feature_columns.xls", index=False, header=False)

print("Saving model_metadata.json...")
meta = {
    "top_k_maps": top_k_maps,
    "num_features": num_features
}
with open("model_metadata.json", "w") as f:
    json.dump(meta, f, indent=4)

print("Done successfully.")
