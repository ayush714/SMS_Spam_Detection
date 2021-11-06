from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class EvaluateModel:
    def __init__(self, x_test, y_test, model):
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

    def evaluate_model(self):
        print("Evaluating the model:- ")
        y_pred = self.model.predict(self.x_test)
        print("Accuracy Score:- ", accuracy_score(self.y_test, y_pred))
        print("Precision Score:- ", precision_score(self.y_test, y_pred))
        print("Recall Score:- ", recall_score(self.y_test, y_pred))
        print("F1 Score:- ", f1_score(self.y_test, y_pred))
        print(
            "Log Loss:- ", log_loss(self.y_test, self.model.predict_proba(self.x_test))
        )
        print("Completed evaluating the model")

    def plot_confusion_matrix(self, test_y, predict_y):
        confusion = confusion_matrix(test_y, predict_y)

        Recall = ((confusion.T) / (confusion.sum(axis=1))).T
        # divide each element of the confusion matrix with the sum of elements in that column

        Precision = confusion / confusion.sum(axis=0)
        # divide each element of the confusion matrix with the sum of elements in that row

        plt.figure(figsize=(20, 4))

        labels = [0, 1]
        cmap = sns.light_palette("blue")
        plt.subplot(1, 3, 1)
        sns.heatmap(
            confusion,
            annot=True,
            cmap=cmap,
            fmt=".3f",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("Original Class")
        plt.title("Confusion matrix")

        plt.subplot(1, 3, 2)
        sns.heatmap(
            Precision,
            annot=True,
            cmap=cmap,
            fmt=".3f",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("Original Class")
        plt.title("Precision matrix")

        plt.subplot(1, 3, 3)
        sns.heatmap(
            Recall,
            annot=True,
            cmap=cmap,
            fmt=".3f",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("Original Class")
        plt.title("Recall matrix")

        plt.show()

    def plot_roc_curve(self, test_y, predict_y):
        auroc = roc_auc_score(test_y, predict_y)
        print("AUROC Score:- ", auroc)
        fpr, tpr, _ = roc_curve(test_y, predict_y)
        plt.plot(
            fpr, tpr, linestyle="--", label="Prediction_for_lr (AUROC = %0.3f)" % auroc
        )
        plt.title("ROC Plot")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

