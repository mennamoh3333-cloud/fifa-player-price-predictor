
# ===============================


# 0️⃣ Imports
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ===============================
df = pd.read_csv("players_22.csv", low_memory=False)
print("Original Shape:", df.shape)

# ===============================
# 2️⃣ Visualization BEFORE Cleaning
# ===============================
plt.figure()
df['age'].hist()
plt.title('Age Distribution (Before Cleaning)')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure()
df['overall'].hist()
plt.title('Overall Distribution (Before Cleaning)')
plt.xlabel('Overall')
plt.ylabel('Count')
plt.show()

# ===============================
# 3️⃣ Data Cleaning
# ===============================

# Drop useless columns
cols_to_drop = [
    'player_url', 'player_face_url',
    'nation_flag_url', 'club_logo_url',
    'club_flag_url', 'sofifa_id'
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Handle missing values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Convert money columns
def convert_money(value):
    if isinstance(value, str):
        value = value.replace('€', '')
        if 'M' in value:
            return float(value.replace('M', '')) * 1_000_000
        elif 'K' in value:
            return float(value.replace('K', '')) * 1_000
    return value

for col in ['value_eur', 'wage_eur', 'release_clause_eur']:
    if col in df.columns:
        df[col] = df[col].apply(convert_money)

# Remove duplicates & age outliers
df.drop_duplicates(inplace=True)
df = df[(df['age'] >= 16) & (df['age'] <= 45)]

print("After Cleaning Shape:", df.shape)

# ===============================
# 4️⃣ Visualization AFTER Cleaning
# ===============================
plt.figure()
df['age'].hist()
plt.title('Age Distribution (After Cleaning)')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# ===============================
# 5️⃣ Feature Selection + Target
# ===============================

features = [
    'age', 'overall', 'potential',
    'pace', 'shooting', 'passing',
    'dribbling', 'defending', 'physic'
]

X = df[features]

# Log-transform target
y = np.log1p(df['value_eur'])

# ===============================
# 6️⃣ Train / Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 7️⃣ Model (Random Forest)
# ===============================
rf = RandomForestRegressor(
    n_estimators=80,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)



# ===============================
# 8️⃣ Evaluation
# ===============================
y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(
    np.expm1(y_test),
    np.expm1(y_pred)
)

print("\nFinal Model Performance")
print("R2 Score:", round(r2, 3))
print("MAE (EUR):", round(mae, 2))

# ===============================
# 9️⃣ Save Model
# ===============================
joblib.dump(rf, "player_price_predictor.pkl")