# day5_week4_cnn_evaluation.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_basics import SimpleCNN
from cnn_evaluation import (
    evaluate_multiclass, plot_confusion_matrix,
    find_top_confusions, show_misclassified_examples
)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Reload Thursday's trained model — no retraining needed, this is
# exactly why torch.save() at the end of Thursday's script mattered
model = SimpleCNN(n_classes=10)
model.load_state_dict(torch.load("cnn_fashion_mnist.pth"))

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
test_subset = torch.utils.data.Subset(test_dataset, range(1000))
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

print("=" * 60)
print("FULL MULTI-CLASS EVALUATION")
print("=" * 60)
cm, preds, labels, images = evaluate_multiclass(model, test_loader, class_names)

plot_confusion_matrix(cm, class_names)

print("\n" + "=" * 60)
print("TOP CONFUSIONS")
print("=" * 60)
find_top_confusions(cm, class_names)

print("\n" + "=" * 60)
print("MISCLASSIFIED EXAMPLES — inspecting actual failures")
print("=" * 60)
show_misclassified_examples(images, preds, labels, class_names)