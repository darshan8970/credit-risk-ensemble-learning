import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)
from src.config import (
    REPORTS_DIR,
    ASSETS_DIR
)

# Professional plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

def evaluate_models(trained_models: dict, X_test: np.ndarray, y_test: pd.Series, feature_names: list):
    """
    Evaluates all trained models on held-out test set, exports metrics JSON/CSV,
    and generates high-resolution publication-quality visualization figures.
    """
    metrics_summary = {}
    comparison_list = []

    plt.figure(figsize=(9, 7))
    colors = {"Random Forest": "#2E86AB", "Bagging": "#A23E48", "AdaBoost": "#F18F01"}

    # 1. Compute Metrics & Prepare Curves
    roc_data = {}
    pr_data = {}
    cm_dict = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        metrics_summary[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "Confusion Matrix": cm.tolist()
        }

        comparison_list.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4)
        })

        # Curves
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, roc_auc)

        precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_proba)
        pr_data[name] = (precision_pts, recall_pts, pr_auc)

        cm_dict[name] = cm

    # Export CSV & JSON reports
    metrics_df = pd.DataFrame(comparison_list)
    csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    metrics_df.to_csv(csv_path, index=False)
    
    json_path = os.path.join(REPORTS_DIR, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print("\n" + "="*60)
    print("[EVALUATION RESULTS SUMMARY]")
    print(metrics_df.to_string(index=False))
    print(f"\n[SAVED REPORTS] {csv_path} and {json_path}")

    best_model = metrics_df.loc[
        metrics_df["ROC-AUC"].idxmax(),
        "Model"
    ]

    best_roc_auc = metrics_df["ROC-AUC"].max()

    print(
        f"\n[BEST MODEL] {best_model} "
        f"with ROC-AUC = {best_roc_auc:.4f}"
    )   

    # 2. Plot 1: Combined ROC-AUC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6), dpi=300)
    for name, (fpr, tpr, auc_val) in roc_data.items():
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors.get(name, "blue"), linewidth=2.5)
    ax_roc.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.500)", alpha=0.7)
    ax_roc.set_title("ROC Curves Comparison — Loan Default Prediction", fontsize=13, fontweight="bold", pad=12)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    roc_img_path = os.path.join(ASSETS_DIR, "roc_curves.png")
    fig_roc.savefig(roc_img_path)
    plt.close(fig_roc)

    # 3. Plot 2: Precision-Recall Curves
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6), dpi=300)
    for name, (precision_pts, recall_pts, pr_auc_val) in pr_data.items():
        ax_pr.plot(recall_pts, precision_pts, label=f"{name} (PR-AUC = {pr_auc_val:.3f})", color=colors.get(name, "blue"), linewidth=2.5)
    ax_pr.set_title("Precision-Recall Curves Comparison", fontsize=13, fontweight="bold", pad=12)
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    pr_img_path = os.path.join(ASSETS_DIR, "precision_recall_curves.png")
    fig_pr.savefig(pr_img_path)
    plt.close(fig_pr)

    # 4. Plot 3: Side-by-Side Confusion Matrices
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    for ax, (name, cm) in zip(axes_cm, cm_dict.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label (0: Non-Default, 1: Default)", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xticklabels(["Non-Default", "Default"])
        ax.set_yticklabels(["Non-Default", "Default"])
    plt.suptitle("Confusion Matrix Benchmarks", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    cm_img_path = os.path.join(ASSETS_DIR, "confusion_matrices.png")
    fig_cm.savefig(cm_img_path)
    plt.close(fig_cm)

    # 5. Plot 4: Random Forest Feature Importances
    if "Random Forest" in trained_models:
        rf_model = trained_models["Random Forest"]
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False).head(15)

        fig_fi, ax_fi = plt.subplots(figsize=(10, 6), dpi=300)
        sns.barplot(data=fi_df, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False, ax=ax_fi)
        ax_fi.set_title("Top 15 Predictive Features — Random Forest Model", fontsize=13, fontweight="bold", pad=12)
        ax_fi.set_xlabel("Gini Feature Importance Score", fontsize=11)
        ax_fi.set_ylabel("Applicant Attributes", fontsize=11)
        plt.tight_layout()
        fi_img_path = os.path.join(ASSETS_DIR, "feature_importance.png")
        fig_fi.savefig(fi_img_path)
        plt.close(fig_fi)

    # 6. Plot 5: Metrics Comparison Bar Chart
    df_melted = metrics_df.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                                var_name="Metric", value_name="Score")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6), dpi=300)
    sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette=[colors.get(model, "gray") for model in df_melted["Model"].unique()],
        ax=ax_bar,
    ) 
    ax_bar.set_ylim(0.0, 1.05)
    for p in ax_bar.patches:
        height = p.get_height()
        if height > 0:
            ax_bar.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height + 0.01),
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_bar.set_title("Multi-Metric Benchmark Comparison", fontsize=13, fontweight="bold", pad=12)
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    bar_img_path = os.path.join(ASSETS_DIR, "metrics_comparison_bar.png")
    fig_bar.savefig(bar_img_path)
    plt.close(fig_bar)

    print(f"[SAVED PLOTS] All 5 visualization plots saved to {ASSETS_DIR}/")
    return metrics_df
