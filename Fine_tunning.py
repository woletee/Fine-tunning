import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Common Transform for Image Models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load CIFAR-10 for CNN, ResNet, AlexNet, ViT
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Visualization Function
def plot_accuracy(train_acc, test_acc, model_name):
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Training Function
def train_model(model, criterion, optimizer, num_epochs=5):
    train_acc, test_acc = [], []
    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc.append(correct / total)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_acc.append(correct / total)
        print(f'Epoch {epoch+1}, Train Acc: {train_acc[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}')
    return train_acc, test_acc

# 1. Fine-Tuning CNN
cnn_model = models.resnet18(pretrained=True)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 10)
cnn_model = cnn_model.to(device)
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_acc, test_acc = train_model(cnn_model, criterion, optimizer)
plot_accuracy(train_acc, test_acc, 'CNN (ResNet18)')

# 2. Fine-Tuning AlexNet
alexnet_model = models.alexnet(pretrained=True)
alexnet_model.classifier[6] = nn.Linear(alexnet_model.classifier[6].in_features, 10)
alexnet_model = alexnet_model.to(device)
optimizer = optim.Adam(alexnet_model.parameters(), lr=0.001)
train_acc, test_acc = train_model(alexnet_model, criterion, optimizer)
plot_accuracy(train_acc, test_acc, 'AlexNet')

# 3. Fine-Tuning U-Net
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channels, kernel_size=2, stride=2)
        )
    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.decoder(x)
        return x

unet_model = UNet(3, 1).to(device)
optimizer = optim.Adam(unet_model.parameters(), lr=0.001)
train_acc, test_acc = train_model(unet_model, criterion, optimizer)
plot_accuracy(train_acc, test_acc, 'U-Net')

# 4. Fine-Tuning Vision Transformer (ViT)
from torchvision.models import vit_b_16
vit_model = vit_b_16(pretrained=True)
vit_model.heads.head = nn.Linear(vit_model.heads.head.in_features, 10)
vit_model = vit_model.to(device)
optimizer = optim.Adam(vit_model.parameters(), lr=0.001)
train_acc, test_acc = train_model(vit_model, criterion, optimizer)
plot_accuracy(train_acc, test_acc, 'ViT')

# 5. Fine-Tuning BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# 6. Fine-Tuning GPT-2 (LLM)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

print("Fine-tuning for CNN, AlexNet, U-Net, ViT, BERT, and GPT-2.")
#this repository is for fine-tunning cnn, alznet, transformer, and other models too
