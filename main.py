from unittest.mock import DEFAULT
import torch
import torch.nn as nn
import torch.optim as optim
from sympy.plotting.backends.matplotlibbackend import matplotlib
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ImageFolder(root='dog_dataset', transform=transforms)
total_size = len(dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.batch = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.batch2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 54 * 54, 120)
        self.drop = nn.Dropout2d()
        self.fc2 = nn.Linear(120, 5)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.batch(self.conv1(x))))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
custom_loss_history = []
model.to(device)
for epoch in range(15):
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        y_pred = model(images)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    custom_loss_history.append(epoch_loss)
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}:.4f')
model.eval()
with torch.no_grad():
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        print(f'Accuracy: {correct / total}')
names = dataset.classes
print(classification_report(all_labels, all_predictions, target_names=names))
print(confusion_matrix(all_labels, all_predictions))
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(all_labels, all_predictions), annot=True, fmt='d' , cmap="Blues",xticklabels=names, yticklabels=names)
plt.xlabel('Предсказанная порода')
plt.ylabel('Реальная порода')
plt.title('Confusion Matrix')
plt.show()
ab = resnet18(weights="DEFAULT")
ab = ab.to(device)
for param in ab.parameters():
    param.requires_grad = False
num_classes = 5
ab.fc = nn.Linear(in_features=ab.fc.in_features, out_features=num_classes)
ab = ab.to(device)
optimizer1 = torch.optim.Adam(ab.fc.parameters(), lr=0.001)
resnet_loss_history = []
for epoch in range(15):
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = ab(images)
        loss = criterion(outputs, labels)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
    epoch_loss = running_loss / len(train_loader)
    resnet_loss_history.append(epoch_loss)
ab.eval()
with torch.no_grad():
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = ab(images)
        predicted = torch.argmax(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        print(f'Accuracy: {correct / total}')

plt.figure(figsize=(10, 10))
plt.plot(custom_loss_history, label='Custom CNN', color='blue', marker='o')
plt.plot(resnet_loss_history, label='ResNet18', color='orange', marker='s')
plt.title("Сравнение")
plt.xlabel("Ось X")
plt.ylabel("Ось Y")
plt.legend()
plt.show()
torch.save(model.state_dict(), 'dog_breed_model.pth')
model.eval()
example_input = torch.rand(1, 3,224,224).to(device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('dog_breed_model_traced.pt')

loaded = torch.jit.load('dog_breed_model_traced.pt')
outputs = loaded(example_input)