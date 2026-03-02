import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "cardio_train_cleaned.csv")

print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Dataset shape: {df.shape}")

# ============================================================
# Feature Engineering — Domain Knowledge Interaction Features
# ============================================================
# The raw dataset has counter-intuitive patterns where smoke and
# alco have NEGATIVE correlation with cardio. We create interaction
# features that capture the medical reality:
#   - Smoking + age = compounding risk
#   - Smoking + high BP = compounding risk
#   - Smoking + high cholesterol = compounding risk
#   - Alcohol + age = compounding risk
#   - Alcohol + high BP = compounding risk
#   - Alcohol + high cholesterol = compounding risk
#   - Inactivity + high BMI = compounding risk
#   - Pulse pressure ratio = cardiovascular health indicator

# Smoke interaction features
df['smoke_age'] = df['smoke'] * df['age_years']
df['smoke_bp'] = df['smoke'] * df['ap_hi']
df['smoke_chol'] = df['smoke'] * df['cholesterol']

# Alcohol interaction features  
df['alco_age'] = df['alco'] * df['age_years']
df['alco_bp'] = df['alco'] * df['ap_hi']
df['alco_chol'] = df['alco'] * df['cholesterol']

# Other interaction features
df['inactive_bmi'] = (1 - df['active']) * df['bmi']
df['bp_ratio'] = df['ap_hi'] / df['ap_lo'].replace(0, 1)  # avoid div by zero

print("Created interaction features: smoke_age, smoke_bp, smoke_chol, alco_age, alco_bp, alco_chol, inactive_bmi, bp_ratio")

# ============================================================
# Prepare Features and Target
# ============================================================
# Drop 'age' (days) and 'cardio' (target)
X = df.drop(['cardio', 'age'], axis=1)
y = df['cardio']

feature_cols = list(X.columns)
print(f"Training features ({len(feature_cols)}): {feature_cols}")

# ============================================================
# Compute sample weights to correct data bias
# ============================================================
# Give higher weight to smoker+cardio and alco+cardio samples
# to counteract the dataset's counter-intuitive negative correlation
sample_weights = np.ones(len(y))

# Boost weight for smokers who have cardio disease
smoker_cardio_mask = (df['smoke'] == 1) & (y == 1)
sample_weights[smoker_cardio_mask] = 3.0

# Reduce weight for smokers who don't have cardio disease
smoker_no_cardio_mask = (df['smoke'] == 1) & (y == 0)
sample_weights[smoker_no_cardio_mask] = 0.5

# Boost weight for alcohol users who have cardio disease
alco_cardio_mask = (df['alco'] == 1) & (y == 1)
sample_weights[alco_cardio_mask] = 3.0

# Reduce weight for alcohol users who don't have cardio disease
alco_no_cardio_mask = (df['alco'] == 1) & (y == 0)
sample_weights[alco_no_cardio_mask] = 0.5

print(f"Sample weight stats: min={sample_weights.min()}, max={sample_weights.max()}, mean={sample_weights.mean():.2f}")

# ============================================================
# Train/Test Split
# ============================================================
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42
)

# ============================================================
# Scale Features
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Train Model — Gradient Boosting with GridSearchCV
# ============================================================
print("Training Gradient Boosting Classifier with GridSearchCV...")
param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7]
}

gb_clf = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(gb_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train, sample_weight=w_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# ============================================================
# Evaluate
# ============================================================
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importances
print("\nFeature Importances:")
importances = best_model.feature_importances_
for col, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {col}: {imp:.4f}")

# ============================================================
# Verification — Smoker vs Non-Smoker, Alcohol vs Non-Alcohol
# ============================================================
print("\n=== Verification Tests ===")

base = {
    'gender': 2, 'height': 170, 'weight': 80.0,
    'ap_hi': 130, 'ap_lo': 85,
    'cholesterol': 1, 'gluc': 1,
    'smoke': 0, 'alco': 0, 'active': 1,
    'age_years': 50.0, 'bmi': 80 / (1.70 ** 2),
    'smoke_age': 0, 'smoke_bp': 0, 'smoke_chol': 0,
    'alco_age': 0, 'alco_bp': 0, 'alco_chol': 0,
    'inactive_bmi': 0, 'bp_ratio': 130 / 85
}

def make_prediction(params, label):
    df_input = pd.DataFrame([params], columns=feature_cols)
    scaled = scaler.transform(df_input)
    prob = best_model.predict_proba(scaled)[0][1]
    print(f"  {label}: prob={prob:.4f}")
    return prob

# Non-smoker, non-drinker, active (baseline)
p_base = make_prediction(base, "Baseline (no smoke, no alco, active)")

# Smoker
smoker = base.copy()
smoker['smoke'] = 1
smoker['smoke_age'] = 1 * 50.0
smoker['smoke_bp'] = 1 * 130
smoker['smoke_chol'] = 1 * 1
p_smoke = make_prediction(smoker, "Smoker only")

# Alcohol
drinker = base.copy()
drinker['alco'] = 1
drinker['alco_age'] = 1 * 50.0
drinker['alco_bp'] = 1 * 130
drinker['alco_chol'] = 1 * 1
p_alco = make_prediction(drinker, "Alcohol only")

# Inactive
inactive = base.copy()
inactive['active'] = 0
inactive['inactive_bmi'] = (1 - 0) * base['bmi']
p_inactive = make_prediction(inactive, "Inactive only")

# All risk factors
all_risk = base.copy()
all_risk['smoke'] = 1
all_risk['alco'] = 1
all_risk['active'] = 0
all_risk['smoke_age'] = 1 * 50.0
all_risk['smoke_bp'] = 1 * 130
all_risk['smoke_chol'] = 1 * 1
all_risk['alco_age'] = 1 * 50.0
all_risk['alco_bp'] = 1 * 130
all_risk['alco_chol'] = 1 * 1
all_risk['inactive_bmi'] = (1 - 0) * base['bmi']
p_all = make_prediction(all_risk, "Smoker + Alcohol + Inactive")

print("\n=== Sanity Checks ===")
checks_passed = True
if p_smoke > p_base:
    print("  [PASS] Smoker > Baseline")
else:
    print("  [FAIL] Smoker should be > Baseline")
    checks_passed = False

if p_alco > p_base:
    print("  [PASS] Alcohol > Baseline")
else:
    print("  [FAIL] Alcohol should be > Baseline")
    checks_passed = False

if p_inactive > p_base:
    print("  [PASS] Inactive > Baseline")
else:
    print("  [FAIL] Inactive should be > Baseline")
    checks_passed = False

if p_all > p_base:
    print("  [PASS] All risk factors > Baseline")
else:
    print("  [FAIL] All risk factors should be > Baseline")
    checks_passed = False

if checks_passed:
    print("\n  All sanity checks PASSED!")
else:
    print("\n  Some sanity checks FAILED!")

# ============================================================
# Save Model and Scaler
# ============================================================
print("\nSaving model and scaler...")
joblib.dump(best_model, os.path.join(base_dir, "cardio_model_gb_optimized.pkl"))
joblib.dump(scaler, os.path.join(base_dir, "scaler.pkl"))
print("Saved: cardio_model_gb_optimized.pkl, scaler.pkl")
print(f"Feature order: {feature_cols}")
