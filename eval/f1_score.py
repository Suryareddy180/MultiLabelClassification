import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from loguru import logger as printer
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, roc_curve, auc, multilabel_confusion_matrix
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # For better heatmap display

from dataset.loader import ODIR5K
from network.model import get_model


class Evaluater:

    def __init__(self, params):
        self.params = params
        self.device = self.params.get("device")
        self.set_seed(42)

        test_dataset = ODIR5K(
            img_dir=self.params.get("img_dir"),
            label_dir=self.params.get("label_dir"),
            train_test_size=self.params.get("train_test_size"),
            is_train=False,
            augment=self.params.get("augment")
        )

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=self.params["num_workers"]
        )

        self.classes = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def set_seed(self, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        printer.info(f"Random seed set as {seed}")

    def run(self):
        model = get_model(self.params.get("model_name"), self.device, {})
        model.load_state_dict(torch.load(self.params.get("load_model_path"), map_location=self.device))
        model.to(self.device)
        model.eval()

        all_true, all_pred, all_prob = [], [], []

        with torch.no_grad():
            for batch in self.test_loader:
                data = batch["data"].to(self.device)
                label = batch["label"].to(self.device).float()

                output = model(data)
                probs = torch.sigmoid(output)

                preds = (probs > 0.5).int()

                all_true.append(label.cpu().numpy())
                all_pred.append(preds.cpu().numpy())
                all_prob.append(probs.cpu().numpy())

        y_true = np.vstack(all_true)
        y_pred = np.vstack(all_pred)
        y_prob = np.vstack(all_prob)

        self.evaluate_and_visualize(y_true, y_pred, y_prob)

    def evaluate_and_visualize(self, y_true, y_pred, y_prob):
        # === Classification Report ===
        print("\nClassification Report:\n")
        report = classification_report(
            y_true, y_pred, target_names=self.classes, output_dict=True, zero_division=0
        )
        print(pd.DataFrame(report).transpose())

        # === Macro F1 Score ===
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Macro F1 Score: {macro_f1:.4f}")

        # === ROC-AUC ===
        try:
            roc_auc = roc_auc_score(y_true, y_prob, average="macro")
            print(f"Macro ROC-AUC: {roc_auc:.4f}")
        except ValueError:
            print("ROC AUC score could not be calculated.")

        # === Save CSV of predictions ===
        pred_df = pd.DataFrame(y_pred, columns=self.classes)
        true_df = pd.DataFrame(y_true, columns=[f"True_{c}" for c in self.classes])
        prob_df = pd.DataFrame(y_prob, columns=[f"Prob_{c}" for c in self.classes])
        result_df = pd.concat([true_df, prob_df, pred_df], axis=1)
        os.makedirs("results", exist_ok=True)
        result_df.to_csv("results/predictions.csv", index=False)
        print("Saved predictions to results/predictions.csv")

        # === Confusion Matrix ===
        self.plot_multilabel_confusion_matrix(y_true, y_pred)

        # === ROC Curve ===
        self.plot_roc_curves(y_true, y_prob)

    def plot_multilabel_confusion_matrix(self, y_true, y_pred):
        cm = multilabel_confusion_matrix(y_true, y_pred)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for i, (ax, label) in enumerate(zip(axes, self.classes)):
            tn, fp, fn, tp = cm[i].ravel()
            sns = np.array([[tp, fn], [fp, tn]])
            ax.matshow(sns, cmap=plt.cm.Blues)
            ax.set_title(f'{label}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            for (j, k), val in np.ndenumerate(sns):
                ax.text(k, j, f'{val}', ha='center', va='center', color='red')
        plt.tight_layout()
        plt.savefig("results/confusion_matrix.png")
        plt.show()
        print("Confusion matrix saved as results/confusion_matrix.png")

    def plot_roc_curves(self, y_true, y_prob):
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(self.classes):
            try:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
            except ValueError:
                print(f"ROC could not be calculated for class {label}")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Per-Class ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig("results/roc_curve.png")
        plt.show()
        print("ROC curve saved as results/roc_curve.png")
