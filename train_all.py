# train_all.py
import pandas as pd
import joblib, json
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Paths
data_dir = Path("data")
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)

# Collect results
results = []

for file in data_dir.glob("*_dataset.csv"):
    print(f"\n📂 Processing {file.name} ...")
    df = pd.read_csv(file)

    # Ensure target column exists
    if "target_disease" not in df.columns:
        print(f"❌ Skipping {file.name}, no 'target_disease' column found")
        continue

    # Features and target
    X = df.drop(columns=["target_disease"])
    y = df["target_disease"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Imputer
    imputer = SimpleImputer(strategy="most_frequent")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Classifier
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train_imp, y_train)

    # ✅ Evaluation
    y_pred = clf.predict(X_test_imp)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"📊 Results for {file.stem.replace('_dataset','')}:")
    print("  Accuracy :", round(acc, 3))
    print("  Precision:", round(prec, 3))
    print("  Recall   :", round(rec, 3))
    print("  F1 Score :", round(f1, 3))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Save results for CSV
    results.append({
        "disease": file.stem.replace("_dataset", ""),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    # Save model files
    disease_name = file.stem.replace("_dataset", "")
    joblib.dump(clf, model_dir / f"{disease_name}_model.pkl")
    joblib.dump(imputer, model_dir / f"{disease_name}_imputer.pkl")
    with open(model_dir / f"{disease_name}_features.json", "w") as f:
        json.dump(list(X.columns), f)

    print(f"✅ Saved model, imputer, and features for {disease_name}")

# Save all results into CSV
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(model_dir / "training_results.csv", index=False)
    print("\n📁 Metrics saved in:", model_dir / "training_results.csv")

print("\n🎉 Training + Evaluation complete for all datasets!")
