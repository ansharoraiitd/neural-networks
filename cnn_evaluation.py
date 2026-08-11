# cnn_evaluation.py
"""
WHAT THIS DOES:
Proper multi-class evaluation of Thursday's trained CNN —
confusion matrix, per-class precision/recall, and pulling up
actual misclassified images to inspect by eye.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def evaluate_multiclass(model, data_loader, class_names, device="cpu"):
    """
    Collect all predictions + true labels across a full data loader,
    then compute confusion matrix and per-class report — same
    pattern as Week 1 Thursday's full_report(), extended to N classes.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []  # kept for the misclassification inspection below

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_images.extend(images.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("Per-class report (precision/recall/f1, one row per class):")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    return cm, all_preds, all_labels, np.array(all_images)


def plot_confusion_matrix(cm, class_names, filename="cnn_confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")


def find_top_confusions(cm, class_names, top_n=5):
    """
    Which off-diagonal (true != predicted) cell has the most
    errors? This turns the confusion matrix's visual pattern into
    a ranked, explicit list — the "which two classes get mixed up
    most" answer, stated directly rather than eyeballed from a grid.
    """
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((class_names[i], class_names[j], cm[i, j]))

    confusions.sort(key=lambda x: -x[2])
    print(f"\nTop {top_n} most common confusions (true -> predicted, count):")
    for true_cls, pred_cls, count in confusions[:top_n]:
        print(f"  {true_cls:>15} confused as {pred_cls:<15} : {count} times")

    return confusions[:top_n]


def show_misclassified_examples(images, preds, labels, class_names, n=8,
                                 filename="misclassified_examples.png"):
    """
    Pull up actual misclassified images to inspect by eye — the
    qualitative check that raw metrics can't give you.
    """
    wrong_idx = np.where(preds != labels)[0]
    chosen = np.random.RandomState(42).choice(wrong_idx, size=min(n, len(wrong_idx)), replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, idx in zip(axes.ravel(), chosen):
        img = images[idx].squeeze()  # remove channel dim for grayscale display
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: {class_names[labels[idx]]}\nPred: {class_names[preds[idx]]}",
                     fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")