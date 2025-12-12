import dagshub
import mlflow
import os
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ========= DAGSHUB AUTHENTICATION =========
# Requires 2 environment variables:
#   DAGSHUB_USERNAME  → your Dagshub username
#   CAPSTONE_TEST     → your Dagshub token
dagshub.auth(
    os.getenv("DAGSHUB_USERNAME"),
    os.getenv("CAPSTONE_TEST")
)

# ========= TRACKING URI SETUP =========
repo_owner = "kumarashutoshbtech2023"
repo_name = "mlflow-test"

mlflow.set_tracking_uri(
    f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
)

# ========= EXPERIMENT SETUP =========
mlflow.set_experiment("remote server")

# ========= DATA LOADING =========
wine = load_wine()
X = wine.data
y = wine.target

# Model hyperparameters
max_depth = 10
n_estimators = 5
test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

model_name = "WineRandomForestModel"

# ========= TRAINING & LOGGING =========
with mlflow.start_run():
    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Predictions & accuracy
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and parameters
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("test_size", test_size)

    # Log confusion matrix as artifact
    with open("confusion_matrix.txt", "w") as f:
        f.write(str(confusion_matrix(y_test, y_pred)))
    mlflow.log_artifact("confusion_matrix.txt")

    # Tags
    mlflow.set_tags({
        "Author": "ashutosh",
        "Project": "log and register mlflow models"
    })

    # Log and register the model
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="model",
        registered_model_name=model_name
    )

print("Training and logging complete.")
