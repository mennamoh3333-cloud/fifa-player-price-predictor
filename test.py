# ===============================
# 0️⃣ Imports
# ===============================
from turtle import lt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# ===============================
# 1️⃣ Load Dataset (FIFA 22 - Kaggle)
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
plt.ylabel('Number of Players')
plt.show()

plt.figure()
df['overall'].hist()
plt.title('Overall Rating Distribution (Before Cleaning)')
plt.xlabel('Overall Rating')
plt.ylabel('Number of Players')
plt.show()

plt.figure()
df['value_eur'].hist(bins=50)
plt.title('Player Market Value Distribution (Before Cleaning)')
plt.xlabel('Market Value (EUR)')
plt.ylabel('Count')
plt.show()

plt.figure()
plt.scatter(df['age'], df['value_eur'], alpha=0.3)
plt.title('Age vs Market Value (Before Cleaning)')
plt.xlabel('Age')
plt.ylabel('Market Value (EUR)')
plt.show()

plt.figure()
plt.boxplot(df['age'].dropna())
plt.title('Age Outliers (Before Cleaning)')
plt.ylabel('Age')
plt.show()


# ===============================
# 3️⃣ Data Cleaning
# ===============================

# Drop useless URL columns
cols_to_drop = [
    'player_url', 'player_face_url',
    'nation_flag_url', 'club_logo_url',
    'club_flag_url', 'sofifa_id'
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Fill missing numerical values with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Convert money columns from string to numeric
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

# Remove duplicates and age outliers
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
plt.ylabel('Number of Players')
plt.show()

plt.figure()
df['overall'].hist()
plt.title('Overall Rating Distribution (After Cleaning)')
plt.xlabel('Overall Rating')
plt.ylabel('Count')
plt.show()

plt.figure()
df['value_eur'].hist(bins=50)
plt.title('Market Value Distribution (After Cleaning)')
plt.xlabel('Market Value (EUR)')
plt.ylabel('Count')
plt.show()

plt.figure()
plt.scatter(df['age'], df['value_eur'], alpha=0.3)
plt.title('Age vs Market Value (After Cleaning)')
plt.xlabel('Age')
plt.ylabel('Market Value (EUR)')
plt.show()


plt.figure()
plt.boxplot(df['age'])
plt.title('Age Boxplot (After Cleaning)')
plt.ylabel('Age')
plt.show()



# ===============================
# 5️⃣ Feature Selection + Target
# ===============================
features = [
    'age', 'potential',
    'pace', 'shooting', 'passing',
    'dribbling', 'defending', 'physic'
]

X = df[features]

# Log Transform Target (to reduce skewness)
y = np.log1p(df['value_eur'])


# ===============================
# 6️⃣ Train / Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ===============================
# 7️⃣ Random Forest Model (Reduced Complexity)
# ===============================
rf = RandomForestRegressor(
    n_estimators=30,
    max_depth=8,
    min_samples_split=30,
    min_samples_leaf=12,
    max_features=0.6,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

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
print("R2 Score:", round(r2, 3))      # ≈ 0.93 – 0.94
print("MAE (EUR):", round(mae, 2))


# ===============================
# 9️⃣ Sample Prediction Comparison
# ===============================
comparison = pd.DataFrame({
    'Actual Value (€)': np.expm1(y_test[:10]),
    'Predicted Value (€)': np.expm1(y_pred[:10])
})

print("\nFirst 10 Players Comparison:")
print(comparison)
# ===============================
# 🔟 Save Model
# ===============================
joblib.dump(rf, "player_price_predictor.pkl")
print("\nModel saved successfully!")