"""
modelling.py — MLflow Project Entry Point
==========================================
Versi ini dijalankan oleh MLflow Project (mlflow run) melalui GitHub Actions CI.
Model: CBR (KNeighborsClassifier)
Nama : Mochamad Ferdynand Winarto
"""

import os
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ================================================================
# KONFIGURASI
# ================================================================
EXPERIMENT_NAME = "CBR_Disease_Prediction_MochamadFerdynandWinarto"

# Dataset diambil dari folder dataset_preprocessing di dalam MLProject
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_DIR = os.path.join(BASE_DIR, "dataset_preprocessing")

X_TRAIN_PATH = os.path.join(PREPROCESS_DIR, "X_train.csv")
X_TEST_PATH  = os.path.join(PREPROCESS_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(PREPROCESS_DIR, "y_train.csv")
Y_TEST_PATH  = os.path.join(PREPROCESS_DIR, "y_test.csv")

# Hyperparameter terbaik dari hasil tuning Kriteria 2
N_NEIGHBORS = int(os.getenv("N_NEIGHBORS", 3))
METRIC      = os.getenv("METRIC", "euclidean")
WEIGHTS     = os.getenv("WEIGHTS", "uniform")


# ================================================================
# FUNGSI
# ================================================================

def load_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test  = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test  = pd.read_csv(Y_TEST_PATH).squeeze()
    print(f"[LOAD]  X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, output_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=0.5, annot_kws={"size": 7})
    ax.set_title("Confusion Matrix — CBR (KNN)\nDisease Prediction", fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


# ================================================================
# PIPELINE UTAMA
# ================================================================

def run():
    print("\n" + "=" * 60)
    print("  MLflow Project — CBR Disease Prediction")
    print("  Mochamad Ferdynand Winarto")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(disable=True)
    
    with mlflow.start_run(run_name="CI_CBR_KNN_Run"):

        # Train model
        model = KNeighborsClassifier(
            n_neighbors=N_NEIGHBORS,
            metric=METRIC,
            weights=WEIGHTS
        )
        print(f"[TRAIN] n_neighbors={N_NEIGHBORS}, metric={METRIC}, weights={WEIGHTS}")
        model.fit(X_train, y_train)

        # Prediksi
        y_pred    = model.predict(X_test)
        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
        f1        = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

        print(f"[HASIL] Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Manual log
        mlflow.log_param("n_neighbors", N_NEIGHBORS)
        mlflow.log_param("metric",      METRIC)
        mlflow.log_param("weights",     WEIGHTS)
        mlflow.log_metric("accuracy",           acc)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted",    recall)
        mlflow.log_metric("f1_weighted",        f1)

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model              = model,
            artifact_path         = "model",
            signature             = signature,
            registered_model_name = "CBR_KNN_Disease_Prediction"
        )

        # Log confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact(cm_path, "plots")

        run_id = mlflow.active_run().info.run_id
        print(f"\n[MLFLOW] Run ID: {run_id}")
        
        # Simpan run_id ke file agar bisa diambil oleh GitHub Actions
        # Disimpan di dua lokasi karena ci.yml mencari di root maupun folder MLProject
        with open("latest_run_id.txt", "w") as f:
            f.write(run_id)
        print(f"[SAVE]  Run ID disimpan ke latest_run_id.txt")

        mlproject_run_id_path = os.path.join(BASE_DIR, "latest_run_id.txt")
        with open(mlproject_run_id_path, "w") as f:
            f.write(run_id)
        print(f"[SAVE]  Run ID disimpan ke MLProject/latest_run_id.txt")

        # Simpan metric_info.json agar Inference.py bisa membaca akurasi training
        import json
        metric_info = {
            "accuracy"          : round(acc,       4),
            "precision_weighted": round(precision, 4),
            "recall_weighted"   : round(recall,    4),
            "f1_weighted"       : round(f1,        4),
            "n_neighbors"       : N_NEIGHBORS,
            "metric"            : METRIC,
            "weights"           : WEIGHTS,
            "run_id"            : run_id
        }
        # Simpan di root (untuk Inference.py di Monitoring_dan_Logging)
        with open("metric_info.json", "w") as f:
            json.dump(metric_info, f, indent=2)
        print(f"[SAVE]  Metrik disimpan ke metric_info.json")
        print(f"        accuracy={acc:.4f} | f1={f1:.4f}")

    print("\n✅ MLflow Project selesai!\n")


if __name__ == "__main__":
    run()
