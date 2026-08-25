# ============================================================
# CHAPTER 4: RESULTS AND ANALYSIS (FINAL CLEAN VERSION)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# LOAD PRIMARY DATA
# ============================================================
survey = pd.read_excel("Primary Data.xlsx", engine="openpyxl")
survey.columns = survey.columns.str.strip()

# ============================================================
# 4.2 DESCRIPTIVE STATISTICS
# ============================================================
print("\n==============================")
print("SECTION 4.2: DESCRIPTIVE STATISTICS")
print("==============================\n")

# AGE
if "Age" in survey.columns:
    age_counts = survey["Age"].value_counts()
    print(age_counts)
    age_counts.plot(kind='bar', title="Age Distribution")
    plt.show()

# GENDER
if "Gender" in survey.columns:
    gender_counts = survey["Gender"].value_counts()
    print(gender_counts)
    gender_counts.plot(kind='bar', title="Gender Distribution")
    plt.show()

# PLATFORM
platform_col = "Which social media platform do you use the most?"
if platform_col in survey.columns:
    platform_counts = survey[platform_col].value_counts()
    print(platform_counts)
    platform_counts.plot(kind='bar', title="Platform Usage")
    plt.show()

# POSTING
freq_col = "How often do you post on social media?"
if freq_col in survey.columns:
    freq_counts = survey[freq_col].value_counts()
    print(freq_counts)
    freq_counts.plot(kind='bar', title="Posting Frequency")
    plt.show()

# SELF-REFERENCE
self_col = "How often do you use words like I, me, or my?"
if self_col in survey.columns:
    self_counts = survey[self_col].value_counts()
    print(self_counts)
    self_counts.plot(kind='bar', title="Self-Referential Language")
    plt.show()

# ============================================================
# 4.4 RELIABILITY ANALYSIS (FINAL — CORRECT VERSION)
# ============================================================

print("\n==============================")
print("SECTION 4.4: RELIABILITY ANALYSIS")
print("==============================\n")

likert_cols = ['Statement 12', 'Statement 13', 'Statement 14', 'Statement 15']

# Step 1: Clean text properly
for col in likert_cols:
    survey[col] = survey[col].astype(str).str.strip().str.lower()

# Step 2: Correct mapping (based on your Excel data)
mapping = {
    # Strongly Agree
    "strongly agree (sa)": 5,
    "strongly agree": 5,
    "strongly agreed(sa)": 5,
    "strongly agree (sa": 5,

    # Agree
    "agree (a)": 4,
    "agree(a)": 4,
    "agree": 4,
    "agreed (a)": 4,

    # Undecided
    "undecided (u)": 3,
    "undecided": 3,
    "neither agree nor disagree": 3,

    # Disagree
    "disagree (d)": 2,
    "disagree(d)": 2,
    "disagree": 2,
    "disagree(da)": 2,

    # Strongly Disagree
    "strongly disagree (sd)": 1,
    "strongly disagree": 1
}

for col in likert_cols:
    survey[col] = survey[col].replace(mapping)
    survey[col] = pd.to_numeric(survey[col], errors='coerce')

# ✅ FINAL STEP: strict clean dataset (this matches your Chapter 4 logic)
clean_data = survey[likert_cols].dropna()

print("Rows used after cleaning:", clean_data.shape[0])

# Cronbach function
def cronbach_alpha(df):
    item_scores = df.values
    item_vars = item_scores.var(axis=0, ddof=1)
    total_scores = item_scores.sum(axis=1)
    total_var = total_scores.var(ddof=1)
    n_items = df.shape[1]
    return (n_items/(n_items-1)) * (1 - (item_vars.sum()/total_var))

alpha = cronbach_alpha(clean_data)

print("✅ Cronbach Alpha:", round(alpha, 3))



# ============================================================
# 4.5 CORRELATION ANALYSIS (FINAL — MATCHES CHAPTER)
# ============================================================

print("\n==============================")
print("SECTION 4.5: CORRELATION ANALYSIS")
print("==============================\n")

# ✅ IMPORTANT: use original data (pairwise)
corr = survey[likert_cols].corr()

print(corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ============================================================
# 4.6 USER PERCEPTION
# ============================================================
print("\n==============================")
print("SECTION 4.6: USER PERCEPTION")
print("==============================\n")

cols = [
    "Would you be willing to use an AI system that analyzes your social media for mental health insights?",
    "Would you trust an AI system more if it explains why it flagged you as at-risk?",
    "Do you think your social media data should be used for mental health research?",
    "Would you allow your anonymized data to be used to improve mental health detection systems?"
]

for col in cols:
    if col in survey.columns:
        counts = survey[col].value_counts()
        print(f"\n{col}\n", counts)
        counts.plot(kind='bar', title=col)
        plt.show()

# ============================================================
# LOAD REDDIT DATA
# ============================================================
df = pd.read_excel("Reddit data.xlsx", engine="openpyxl")

# ============================================================
# 4.7 DATASET OVERVIEW
# ============================================================
print("\n==============================")
print("SECTION 4.7: DATASET OVERVIEW")
print("==============================\n")

print(df.shape)
print(df.head())

df["status"].value_counts().plot(kind='bar', title="Class Distribution")
plt.show()

# ============================================================
# 4.8 PREPROCESSING
# ============================================================
df = df.dropna(subset=["statement", "status"])
df["statement"] = df["statement"].astype(str)
df["status"] = df["status"].astype(str)

# ============================================================
# 4.9 MACHINE LEARNING
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = df["statement"]
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

# RESULTS
print("\n==============================")
print("MODEL RESULTS")
print("==============================\n")

print("✅ Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(classification_report(y_test, predictions))

# ============================================================
# 4.9.3 MODEL COMPARISON (NEW SECTION)
# ============================================================

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

print("\n==============================")
print("SECTION 4.9.3: MODEL COMPARISON")
print("==============================\n")

# ----------------------------
# SVM MODEL
# ----------------------------
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
svm_pred = svm_model.predict(X_test_vec)

svm_acc = accuracy_score(y_test, svm_pred)

# ----------------------------
# RANDOM FOREST MODEL
# ----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_vec, y_train)
rf_pred = rf_model.predict(X_test_vec)

rf_acc = accuracy_score(y_test, rf_pred)

# ----------------------------
# PRINT COMPARISON
# ----------------------------
lr_acc = accuracy_score(y_test, predictions)

print(f"Logistic Regression Accuracy: {round(lr_acc, 4)}")
print(f"SVM Accuracy: {round(svm_acc, 4)}")
print(f"Random Forest Accuracy: {round(rf_acc, 4)}")

# ----------------------------
# COMPARISON TABLE
# ----------------------------
comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "Random Forest"],
    "Accuracy": [lr_acc, svm_acc, rf_acc]
})

print("\nModel Comparison Table:\n")
print(comparison_df)

# ----------------------------
# PLOT COMPARISON
# ----------------------------
comparison_df.plot(kind='bar', x='Model', y='Accuracy', legend=False)
plt.title("Model Comparison (Accuracy)")
plt.ylabel("Accuracy")
plt.xticks(rotation=0)
plt.show()

# ============================================================
# 4.10 CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, predictions)

print("Confusion Matrix:\n", cm)

sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================================
# END
# ============================================================