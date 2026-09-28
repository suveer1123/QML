# ================================================================
# HYBRID QUANTUM MACHINE LEARNING
# LEVEL 1 - FEVER / ACUTE INFLAMMATION
#
# Purpose:
# Test the ML + QML algorithm using a locally generated
# biomedical-style dataset.
#
# Classical model : SVM
# Quantum model   : QSVM
#
# Later, the dataset-generation section can be replaced with
# a CSV/Excel upload without changing the ML/QML pipeline.
# ================================================================

import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

RANDOM_STATE = 42

# Number of artificial patients
N_SAMPLES = 120

# Number of cross-validation folds
N_FOLDS = 5

# Number of features that will be passed to the quantum circuit
# This is also the number of qubits.
N_QUBITS = 4


# ================================================================
# 1. GENERATE A ROUGH BIOMEDICAL DATASET
# ================================================================

def generate_dataset():

    print()
    print("=" * 70)
    print("GENERATING FEVER / ACUTE INFLAMMATION DATA")
    print("=" * 70)

    rng = np.random.default_rng(RANDOM_STATE)

    data = []

    for _ in range(N_SAMPLES):

        # --------------------------------------------------------
        # Temperature
        # --------------------------------------------------------

        temperature = rng.normal(
            loc=38.0,
            scale=1.2
        )

        temperature = np.clip(
            temperature,
            35.5,
            41.5
        )

        # --------------------------------------------------------
        # Symptoms
        #
        # 0 = No
        # 1 = Yes
        # --------------------------------------------------------

        nausea = rng.binomial(
            1,
            0.35
        )

        lumbar_pain = rng.binomial(
            1,
            0.35
        )

        urine_pushing = rng.binomial(
            1,
            0.40
        )

        micturition_pain = rng.binomial(
            1,
            0.40
        )

        burning_urethra = rng.binomial(
            1,
            0.40
        )

        # --------------------------------------------------------
        # Artificial disease-risk relationship
        #
        # This is ONLY for testing the algorithm.
        # It is NOT a medical formula.
        # --------------------------------------------------------

        risk = 0

        # Temperature
        if temperature >= 38.5:
            risk += 2
        elif temperature >= 37.8:
            risk += 1

        # Symptoms
        risk += nausea
        risk += lumbar_pain * 2
        risk += urine_pushing
        risk += micturition_pain
        risk += burning_urethra

        # Small amount of randomness
        risk += rng.normal(
            0,
            1.0
        )

        # --------------------------------------------------------
        # Target
        #
        # 0 = No inflammation
        # 1 = Inflammation
        # --------------------------------------------------------

        if risk >= 4:
            inflammation = 1
        else:
            inflammation = 0

        data.append([
            temperature,
            nausea,
            lumbar_pain,
            urine_pushing,
            micturition_pain,
            burning_urethra,
            inflammation
        ])

    columns = [
        "temperature",
        "nausea",
        "lumbar_pain",
        "urine_pushing",
        "micturition_pain",
        "burning_urethra",
        "inflammation"
    ]

    df = pd.DataFrame(
        data,
        columns=columns
    )

    # Save it so we can inspect the actual data
    df.to_csv(
        "rough_fever_dataset.csv",
        index=False
    )

    print()
    print("Dataset created successfully.")

    print()
    print("Dataset shape:")
    print(df.shape)

    print()
    print("First 10 records:")
    print(df.head(10).to_string(index=False))

    print()
    print("Class distribution:")
    print(df["inflammation"].value_counts())

    return df


# ================================================================
# 2. BASIC DATA ANALYSIS
# ================================================================

def analyze_dataset(df):

    print()
    print("=" * 70)
    print("BASIC DATA ANALYSIS")
    print("=" * 70)

    no_inflammation = df[
        df["inflammation"] == 0
    ]

    inflammation = df[
        df["inflammation"] == 1
    ]

    print()
    print(
        "Average temperature - no inflammation:",
        round(
            no_inflammation["temperature"].mean(),
            2
        ),
        "°C"
    )

    print(
        "Average temperature - inflammation:",
        round(
            inflammation["temperature"].mean(),
            2
        ),
        "°C"
    )

    print()
    print(
        "Minimum temperature:",
        round(
            df["temperature"].min(),
            2
        ),
        "°C"
    )

    print(
        "Maximum temperature:",
        round(
            df["temperature"].max(),
            2
        ),
        "°C"
    )

    fever_count = (
        df["temperature"] >= 38.0
    ).sum()

    print()
    print(
        "Patients with temperature >= 38°C:",
        fever_count
    )


# ================================================================
# 3. PREPARE DATA
# ================================================================

def prepare_data(df):

    feature_columns = [
        "temperature",
        "nausea",
        "lumbar_pain",
        "urine_pushing",
        "micturition_pain",
        "burning_urethra"
    ]

    X = df[
        feature_columns
    ].to_numpy(
        dtype=float
    )

    y = df[
        "inflammation"
    ].to_numpy(
        dtype=int
    )

    return X, y


# ================================================================
# 4. METRIC FUNCTION
# ================================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_score
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    if cm.shape == (2, 2):

        tn = cm[0, 0]
        fp = cm[0, 1]

        if tn + fp > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0.0

    else:

        specificity = 0.0

    try:

        auc = roc_auc_score(
            y_true,
            y_score
        )

    except Exception:

        auc = 0.0

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": auc
    }


# ================================================================
# 5. CLASSICAL MACHINE LEARNING
#
# Pipeline:
#
# Raw data
#     ↓
# Feature selection
#     ↓
# Standardization
#     ↓
# SVM
#     ↓
# Prediction
# ================================================================

def run_classical_svm(X, y):

    print()
    print("=" * 70)
    print("CLASSICAL MACHINE LEARNING")
    print("MODEL: SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_results = []

    all_true = []
    all_pred = []

    total_start = time.perf_counter()

    for fold, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print()
        print(
            f"CLASSICAL ML - FOLD {fold}/{N_FOLDS}"
        )

        X_train = X[
            train_index
        ]

        X_test = X[
            test_index
        ]

        y_train = y[
            train_index
        ]

        y_test = y[
            test_index
        ]

        fold_start = time.perf_counter()

        # --------------------------------------------------------
        # Feature selection
        # --------------------------------------------------------

        selector = SelectKBest(
            score_func=f_classif,
            k=N_QUBITS
        )

        X_train_selected = selector.fit_transform(
            X_train,
            y_train
        )

        X_test_selected = selector.transform(
            X_test
        )

        # --------------------------------------------------------
        # Scaling
        # --------------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train_selected
        )

        X_test_scaled = scaler.transform(
            X_test_selected
        )

        # --------------------------------------------------------
        # SVM
        # --------------------------------------------------------

        model = SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        predictions = model.predict(
            X_test_scaled
        )

        probabilities = model.predict_proba(
            X_test_scaled
        )[:, 1]

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities
        )

        fold_time = (
            time.perf_counter()
            - fold_start
        )

        metrics["runtime"] = fold_time

        fold_results.append(
            metrics
        )

        all_true.extend(
            y_test
        )

        all_pred.extend(
            predictions
        )

        print(
            f"Accuracy    : {metrics['accuracy']:.4f}"
        )

        print(
            f"Sensitivity : {metrics['sensitivity']:.4f}"
        )

        print(
            f"Specificity : {metrics['specificity']:.4f}"
        )

        print(
            f"Precision   : {metrics['precision']:.4f}"
        )

        print(
            f"F1 Score    : {metrics['f1']:.4f}"
        )

        print(
            f"AUC         : {metrics['auc']:.4f}"
        )

        print(
            f"Runtime     : {fold_time:.4f} sec"
        )

    total_time = (
        time.perf_counter()
        - total_start
    )

    results = pd.DataFrame(
        fold_results
    )

    print()
    print("-" * 70)
    print("CLASSICAL SVM FINAL RESULT")
    print("-" * 70)

    print(
        f"Accuracy    : "
        f"{results['accuracy'].mean():.4f}"
        f" +/- "
        f"{results['accuracy'].std():.4f}"
    )

    print(
        f"Sensitivity : "
        f"{results['sensitivity'].mean():.4f}"
        f" +/- "
        f"{results['sensitivity'].std():.4f}"
    )

    print(
        f"Specificity : "
        f"{results['specificity'].mean():.4f}"
        f" +/- "
        f"{results['specificity'].std():.4f}"
    )

    print(
        f"Precision   : "
        f"{results['precision'].mean():.4f}"
    )

    print(
        f"F1 Score    : "
        f"{results['f1'].mean():.4f}"
        f" +/- "
        f"{results['f1'].std():.4f}"
    )

    print(
        f"AUC         : "
        f"{results['auc'].mean():.4f}"
    )

    print(
        f"Total time  : "
        f"{total_time:.4f} sec"
    )

    print()
    print("Overall confusion matrix:")

    print(
        confusion_matrix(
            all_true,
            all_pred
        )
    )

    return results


# ================================================================
# 6. QUANTUM MACHINE LEARNING
#
# Pipeline:
#
# Raw data
#     ↓
# Feature selection
#     ↓
# Standardization
#     ↓
# Quantum encoding
#     ↓
# Quantum feature map
#     ↓
# Quantum kernel
#     ↓
# QSVM
#     ↓
# Prediction
# ================================================================

def run_qsvm(X, y):

    print()
    print("=" * 70)
    print("QUANTUM MACHINE LEARNING")
    print("MODEL: QUANTUM SUPPORT VECTOR MACHINE (QSVM)")
    print("=" * 70)

    print()
    print("Creating quantum feature map...")

    feature_map = zz_feature_map(
        feature_dimension=N_QUBITS,
        reps=1,
        entanglement="linear"
    )

    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_results = []

    all_true = []
    all_pred = []

    total_start = time.perf_counter()

    for fold, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print()
        print(
            f"QUANTUM ML - FOLD {fold}/{N_FOLDS}"
        )

        X_train = X[
            train_index
        ]

        X_test = X[
            test_index
        ]

        y_train = y[
            train_index
        ]

        y_test = y[
            test_index
        ]

        fold_start = time.perf_counter()

        # --------------------------------------------------------
        # Feature selection
        #
        # SAME procedure as classical ML
        # --------------------------------------------------------

        selector = SelectKBest(
            score_func=f_classif,
            k=N_QUBITS
        )

        X_train_selected = selector.fit_transform(
            X_train,
            y_train
        )

        X_test_selected = selector.transform(
            X_test
        )

        # --------------------------------------------------------
        # Standardization
        #
        # SAME procedure as classical ML
        # --------------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train_selected
        )

        X_test_scaled = scaler.transform(
            X_test_selected
        )

        # --------------------------------------------------------
        # Convert classical values into quantum angles
        # --------------------------------------------------------

        X_train_quantum = np.clip(
            X_train_scaled,
            -3,
            3
        )

        X_test_quantum = np.clip(
            X_test_scaled,
            -3,
            3
        )

        X_train_quantum = (
            X_train_quantum
            * np.pi
            / 6
        )

        X_test_quantum = (
            X_test_quantum
            * np.pi
            / 6
        )

        # --------------------------------------------------------
        # QSVM
        # --------------------------------------------------------

        qsvm = QSVC(
            quantum_kernel=quantum_kernel
        )

        qsvm.fit(
            X_train_quantum,
            y_train
        )

        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        predictions = qsvm.predict(
            X_test_quantum
        )

        # QSVC does not necessarily provide probabilities.
        # We use the binary predictions as the score for the
        # basic benchmark.
        scores = predictions.astype(float)

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        metrics = calculate_metrics(
            y_test,
            predictions,
            scores
        )

        fold_time = (
            time.perf_counter()
            - fold_start
        )

        metrics["runtime"] = fold_time

        fold_results.append(
            metrics
        )

        all_true.extend(
            y_test
        )

        all_pred.extend(
            predictions
        )

        print(
            f"Accuracy    : {metrics['accuracy']:.4f}"
        )

        print(
            f"Sensitivity : {metrics['sensitivity']:.4f}"
        )

        print(
            f"Specificity : {metrics['specificity']:.4f}"
        )

        print(
            f"Precision   : {metrics['precision']:.4f}"
        )

        print(
            f"F1 Score    : {metrics['f1']:.4f}"
        )

        print(
            f"AUC         : {metrics['auc']:.4f}"
        )

        print(
            f"Runtime     : {fold_time:.4f} sec"
        )

    total_time = (
        time.perf_counter()
        - total_start
    )

    results = pd.DataFrame(
        fold_results
    )

    print()
    print("-" * 70)
    print("QUANTUM SVM FINAL RESULT")
    print("-" * 70)

    print(
        f"Accuracy    : "
        f"{results['accuracy'].mean():.4f}"
        f" +/- "
        f"{results['accuracy'].std():.4f}"
    )

    print(
        f"Sensitivity : "
        f"{results['sensitivity'].mean():.4f}"
        f" +/- "
        f"{results['sensitivity'].std():.4f}"
    )

    print(
        f"Specificity : "
        f"{results['specificity'].mean():.4f}"
        f" +/- "
        f"{results['specificity'].std():.4f}"
    )

    print(
        f"Precision   : "
        f"{results['precision'].mean():.4f}"
    )

    print(
        f"F1 Score    : "
        f"{results['f1'].mean():.4f}"
        f" +/- "
        f"{results['f1'].std():.4f}"
    )

    print(
        f"AUC         : "
        f"{results['auc'].mean():.4f}"
    )

    print(
        f"Total time  : "
        f"{total_time:.4f} sec"
    )

    print()
    print("Overall confusion matrix:")

    print(
        confusion_matrix(
            all_true,
            all_pred
        )
    )

    return results


# ================================================================
# 7. COMPARE CLASSICAL ML AND QML
# ================================================================

def compare_models(
    classical_results,
    quantum_results
):

    print()
    print("=" * 70)
    print("CLASSICAL ML vs QUANTUM ML")
    print("=" * 70)

    classical = {
        "Model": "Classical SVM",

        "Accuracy":
            classical_results["accuracy"].mean(),

        "Sensitivity":
            classical_results["sensitivity"].mean(),

        "Specificity":
            classical_results["specificity"].mean(),

        "Precision":
            classical_results["precision"].mean(),

        "F1":
            classical_results["f1"].mean(),

        "AUC":
            classical_results["auc"].mean(),

        "Runtime":
            classical_results["runtime"].mean()
    }

    quantum = {
        "Model": "Quantum SVM",

        "Accuracy":
            quantum_results["accuracy"].mean(),

        "Sensitivity":
            quantum_results["sensitivity"].mean(),

        "Specificity":
            quantum_results["specificity"].mean(),

        "Precision":
            quantum_results["precision"].mean(),

        "F1":
            quantum_results["f1"].mean(),

        "AUC":
            quantum_results["auc"].mean(),

        "Runtime":
            quantum_results["runtime"].mean()
    }

    comparison = pd.DataFrame([
        classical,
        quantum
    ])

    print()

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    comparison.to_csv(
        "ml_vs_qml_results.csv",
        index=False
    )

    print()
    print(
        "Comparison saved to:"
    )

    print(
        "ml_vs_qml_results.csv"
    )


# ================================================================
# 8. EXPLAIN WHAT THE METRICS MEAN
# ================================================================

def explain_results():

    print()
    print("=" * 70)
    print("WHAT THE METRICS MEAN")
    print("=" * 70)

    print(
        """
Accuracy
--------
Percentage of all predictions that were correct.


Sensitivity
-----------
Also called True Positive Rate.

It answers:

"Of the patients who actually have inflammation,
how many did the model detect?"

Higher sensitivity means fewer disease cases are missed.


Specificity
-----------
Also called True Negative Rate.

It answers:

"Of the patients who do NOT have inflammation,
how many did the model correctly identify?"

Higher specificity means fewer healthy cases are
incorrectly classified as diseased.


Precision
---------
Of the patients predicted as having inflammation,
how many actually had inflammation?


F1 Score
--------
Combines precision and sensitivity into one metric.


AUC
---
Measures how well the model separates the two classes.


Runtime
-------
Measures how long the model takes to perform the
cross-validation experiment.

Because QSVM is being simulated on a classical computer,
its runtime represents simulation cost, NOT the speed
of a real quantum computer.
"""
    )


# ================================================================
# 9. MAIN
# ================================================================

def main():

    print()
    print("=" * 70)
    print("HYBRID QUANTUM MACHINE LEARNING PLATFORM")
    print("LEVEL 1: FEVER / ACUTE INFLAMMATION")
    print("=" * 70)

    print(
        """
GOAL
----
Test the same classification problem using:

    Classical SVM
          VS
    Quantum SVM (QSVM)

The dataset is generated locally so that we can
focus on testing the algorithm itself.

Later, the same pipeline can accept an external
CSV/Excel biomedical dataset.
"""
    )

    # ------------------------------------------------------------
    # STEP 1
    # ------------------------------------------------------------

    df = generate_dataset()

    # ------------------------------------------------------------
    # STEP 2
    # ------------------------------------------------------------

    analyze_dataset(
        df
    )

    # ------------------------------------------------------------
    # STEP 3
    # ------------------------------------------------------------

    X, y = prepare_data(
        df
    )

    print()
    print("=" * 70)
    print("DATA READY")
    print("=" * 70)

    print(
        "Samples :",
        X.shape[0]
    )

    print(
        "Features:",
        X.shape[1]
    )

    print(
        "Classes :",
        len(np.unique(y))
    )

    # ------------------------------------------------------------
    # STEP 4
    # Classical ML
    # ------------------------------------------------------------

    classical_results = run_classical_svm(
        X,
        y
    )

    # ------------------------------------------------------------
    # STEP 5
    # Quantum ML
    # ------------------------------------------------------------

    quantum_results = run_qsvm(
        X,
        y
    )

    # ------------------------------------------------------------
    # STEP 6
    # Compare
    # ------------------------------------------------------------

    compare_models(
        classical_results,
        quantum_results
    )

    # ------------------------------------------------------------
    # STEP 7
    # Explain
    # ------------------------------------------------------------

    explain_results()

    # ------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print()
    print("Generated files:")

    print(
        "1. rough_fever_dataset.csv"
    )

    print(
        "2. ml_vs_qml_results.csv"
    )

    print()
    print(
        "The algorithm has now tested both Classical ML"
    )

    print(
        "and Quantum ML on the same dataset."
    )

    print()
    print(
        "NOTE:"
    )

    print(
        "This generated dataset is only for algorithm"
    )

    print(
        "testing. It is not real patient/clinical data."
    )


# ================================================================
# START PROGRAM
# ================================================================

if __name__ == "__main__":
    main()