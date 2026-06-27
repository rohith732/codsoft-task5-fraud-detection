import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   TASK 5: CREDIT CARD FRAUD DETECTION")
print("   CodSoft Data Science Internship")
print("=" * 60)

np.random.seed(42)
n_legit  = 284315
n_fraud  = 492
n_total  = n_legit + n_fraud

print(f"\nGenerating dataset: {n_total:,} transactions")
print(f"  Legitimate : {n_legit:,}  ({n_legit/n_total:.3%})")
print(f"  Fraudulent : {n_fraud:,}   ({n_fraud/n_total:.3%})")

V_legit = np.random.randn(n_legit, 10)
V_fraud = np.random.randn(n_fraud, 10) + np.array([2,-2,1.5,-1,0.8,-0.5,1,-1.2,0.5,-0.3])

amount_legit = np.random.exponential(88, n_legit)
amount_fraud = np.random.exponential(122, n_fraud)
time_legit   = np.random.uniform(0, 172792, n_legit)
time_fraud   = np.random.uniform(0, 172792, n_fraud)

cols = [f'V{i}' for i in range(1, 11)]
df_legit = pd.DataFrame(V_legit, columns=cols)
df_legit['Amount'] = amount_legit
df_legit['Time']   = time_legit
df_legit['Class']  = 0

df_fraud = pd.DataFrame(V_fraud, columns=cols)
df_fraud['Amount'] = amount_fraud
df_fraud['Time']   = time_fraud
df_fraud['Class']  = 1

df = pd.concat([df_legit, df_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
df['Time']   = (df['Time']   - df['Time'].mean())   / df['Time'].std()

print(f"\nClass distribution:\n{df['Class'].value_counts()}")

features = cols + ['Amount', 'Time']
X = df[features]
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_major = X_train[y_train == 0]
X_train_minor = X_train[y_train == 1]
y_train_major = y_train[y_train == 0]
y_train_minor = y_train[y_train == 1]

X_minor_up, y_minor_up = resample(X_train_minor, y_train_minor,
                                   replace=True, n_samples=len(X_train_major)//10,
                                   random_state=42)
X_bal = pd.concat([X_train_major, X_minor_up])
y_bal = pd.concat([y_train_major, y_minor_up])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
}

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

best_model, best_f1 = None, 0
for name, clf in models.items():
    clf.fit(X_bal, y_bal)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"\n{name}:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")

    if f1 > best_f1:
        best_f1, best_model, best_name = f1, clf, name

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_name}")
print(f"{'='*60}")
y_pred_best = best_model.predict(X_test)
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_best)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_best, target_names=['Legitimate','Fraud'])}")

if hasattr(best_model, 'feature_importances_'):
    print("\nTop Feature Importances:")
    imp = sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1])[:8]
    for feat, val in imp:
        bar = "█" * int(val * 200)
        print(f"  {feat:<8} {val:.4f}  {bar}")
