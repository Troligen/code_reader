import torch
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Ensure these are accessible by converting them to numpy arrays
mean = np.array(mean)
std = np.array(std)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_dataset = datasets.ImageFolder("./bird/train", transform=transform)
valid_dataset = datasets.ImageFolder("./bird/valid", transform=transform)
test_dataset = datasets.ImageFolder('./bird/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet50(pretrained=True)
model = nn.DataParallel(model)

for param in model.parameters():
    param.requires_grad = False

num_features = model.module.fc.in_features
num_classes = len(train_dataset.classes)
model.module.fc = nn.Linear(num_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.module.fc.parameters(), lr=0.001)

def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        # Avoid division by zero
        if total > 0:
            accuracy = 100 * valid_correct / total
        else:
            accuracy = 0

        print(f'Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f}, '
              f'Validation Loss: {valid_loss/len(valid_loader):.4f}, '
              f'Validation Accuracy: {accuracy:.2f}%')
    torch.save(model.state_dict(), 'model_state_dict.pth')


train_model(model, criterion, optimizer, train_loader, valid_loader)

def test_model(model, test_loader, classes):
    model.eval()
    actuals, predictions = [], []
    correct_imgs, incorrect_imgs = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            actuals.extend(labels.view_as(predicted))
            predictions.extend(predicted)
            
            # Collect correct and incorrect examples
            matches = predicted == labels
            for i in range(images.size(0)):
                if len(correct_imgs) < 3 and matches[i]:
                    correct_imgs.append((images[i], predicted[i], labels[i]))
                elif len(incorrect_imgs) < 3 and not matches[i]:
                    incorrect_imgs.append((images[i], predicted[i], labels[i]))

    # Plotting correct and incorrect images
    def plot_images(imgs, title):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for idx, (img, pred, actual) in enumerate(imgs):
            ax = axes[idx]
            img = img.cpu().numpy().transpose((1, 2, 0))
            img = std * img + mean
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f'Pred: {classes[pred]} \n Actual: {classes[actual]}')
            ax.axis('off')
        plt.suptitle(title)
        plt.show()

    plot_images(correct_imgs, 'Correctly Classified Images')
    plot_images(incorrect_imgs, 'Incorrectly Classified Images')
    
    actuals = torch.stack(actuals)  # Convert list of tensors to a single tensor
    predictions = torch.stack(predictions)

    # Confusion Matrix
    cm = confusion_matrix(actuals.cpu(), predictions.cpu())
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Analyzing which species model has the hardest time with
    errors = cm - np.diag(np.diag(cm))
    hardest_class_index = np.argmax(errors.sum(axis=0))
    print(f"The model has the hardest time classifying: {classes[hardest_class_index]}")

# Running the testing function
test_model(model, test_loader, train_dataset.classes)
