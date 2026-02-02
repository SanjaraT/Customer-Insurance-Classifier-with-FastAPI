import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("insurance.csv")
# print(df.shape)
# print(df.head())

#Class Distribution
plt.figure(figsize=(6,4))
df['insurance'].value_counts().plot(kind='bar')
plt.title("Target Class Distribution")
plt.xlabel("Insurance Class")
plt.ylabel("Count")
# plt.show()

#Feature Engineering
df_feat = df.copy()

#BMI
df_feat["bmi"] = df_feat["weight"] / (df_feat["height"] ** 2)

#Age Group
def get_age_group(age):
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle_aged"
    else:
        return "senior"

df_feat["age_group"] = df_feat["age"].apply(get_age_group)

#lifestyle risk
df_feat["smoker"] = df_feat["smoker"].astype(int)

def get_lifestyle_risk(smoker, bmi):
    if smoker == 1 and bmi >= 30:
        return "high"
    elif smoker == 1 or bmi >= 27:
        return "medium"
    else:
        return "low"

df_feat["lifestyle_risk"] = df_feat.apply(
    lambda x: get_lifestyle_risk(x["smoker"], x["bmi"]), axis=1
)

#city tier
tier_1_cities = {
    "Mumbai", "Delhi", "Bangalore", "Chennai",
    "Kolkata", "Hyderabad", "Pune"
}

tier_2_cities = {
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi",
    "Visakhapatnam", "Coimbatore", "Bhopal", "Nagpur", "Vadodara",
    "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati",
    "Thiruvananthapuram", "Ludhiana", "Nashik", "Allahabad",
    "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem",
    "Vijayawada", "Tiruchirappalli", "Bhavnagar", "Gwalior",
    "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode",
    "Warangal", "Kolhapur", "Bilaspur", "Jalandhar", "Noida",
    "Guntur", "Asansol", "Siliguri"
}

def get_city_tier(city):
    if city in tier_1_cities:
        return 1
    elif city in tier_2_cities:
        return 2
    else:
        return 3

df_feat["city_tier"] = df_feat["city"].apply(get_city_tier)

df_feat.drop(
    columns=["age", "weight", "height", "smoker", "city"],
    inplace=True
)

#Final features
final_features = [
    "income_lpa",
    "occupation",
    "bmi",
    "age_group",
    "lifestyle_risk",
    "city_tier",
    "insurance"
]

# print(df_feat[final_features].head())

#Split
X = df_feat.drop("insurance", axis=1)
y = df_feat["insurance"]

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# print("Target classes:", target_encoder.classes_)

num_features = ["income_lpa", "bmi", "city_tier"]
cat_features = ["occupation", "age_group", "lifestyle_risk"]

#Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

#Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

model.fit(X_train, y_train)

#evaluation
y_pred = model.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

joblib.dump(model, "insurance_model.pkl")
