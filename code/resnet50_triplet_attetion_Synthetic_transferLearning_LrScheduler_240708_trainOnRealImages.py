import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
from PIL import Image
import os
import numpy as np
from torchsummary import summary
import torch.cuda.amp as amp
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# Define AttentionModule and ResNet50Attention classes
class AttentionModule(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.3):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        query = self.bn1(self.conv1(x))
        key = self.bn2(self.conv2(x))
        attention_map = self.sigmoid(query + key)
        x = x * attention_map
        x = self.dropout(x)
        return x

class ResNet50Attention(nn.Module):
    def __init__(self, dropout_rate=0.5, attention_dropout_rate=0.3):
        super(ResNet50Attention, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Load weights from the internet
        self.attention1 = AttentionModule(256, dropout_rate=attention_dropout_rate)
        self.attention2 = AttentionModule(512, dropout_rate=attention_dropout_rate)
        self.attention3 = AttentionModule(1024, dropout_rate=attention_dropout_rate)
        self.attention4 = AttentionModule(2048, dropout_rate=attention_dropout_rate)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.resnet.fc.in_features, 256)
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.attention1(x)
        x = self.resnet.layer2(x)
        x = self.attention2(x)
        x = self.resnet.layer3(x)
        x = self.attention3(x)
        x = self.resnet.layer4(x)
        x = self.attention4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

# Define TripletLoss class
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# Function to load the trained model
def load_trained_model(model_path, device):
    model = ResNet50Attention(dropout_rate=0.5, attention_dropout_rate=0.5)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

# Setup Dataset
train_dir = r'C:\Users\obus\Desktop\finTrack\classification_training_dataset\transferTraining_train'
val_dir = r'C:\Users\obus\Desktop\finTrack\classification_training_dataset\transferTraining_val'
img_height, img_width = 224, 224  # Resize to 224x224 to match ResNet50 input size
batch_size = 32

# Data Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms['val'])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

train_dataset_size = len(train_dataset)
val_dataset_size = len(val_dataset)
class_names = train_dataset.classes

print("Classes:", class_names)
print("Training dataset size:", train_dataset_size)
print("Validation dataset size:", val_dataset_size)

# Initialize device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r'C:\Users\obus\Desktop\finTrack\classification_training_dataset\resnet50-0676ba61.pth'  # Update with the actual path to the single model file
model = load_trained_model(model_path, device).to(device)

# Generate Random Triplets
def generate_random_triplets(dataset, num_triplets):
    triplets = []
    labels = np.array([s[1] for s in dataset.samples])
    unique_labels = np.unique(labels)
    
    for _ in range(num_triplets):
        anchor_label = np.random.choice(unique_labels)
        negative_label = np.random.choice(unique_labels[unique_labels != anchor_label])
        
        anchor_indices = np.where(labels == anchor_label)[0]
        negative_indices = np.where(labels == negative_label)[0]
        
        if len(anchor_indices) > 1:
            anchor_idx = np.random.choice(anchor_indices)
            positive_idx = np.random.choice(anchor_indices)
            while positive_idx == anchor_idx:
                positive_idx = np.random.choice(anchor_indices)
        else:
            continue  # Skip if there aren't enough positive samples
        
        negative_idx = np.random.choice(negative_indices)
        
        anchor = dataset.samples[anchor_idx][0]
        positive = dataset.samples[positive_idx][0]
        negative = dataset.samples[negative_idx][0]
        
        triplets.append((anchor, positive, negative))
    
    return triplets

# Create triplet loaders with random mining
num_train_triplets = len(train_dataset)  # You can adjust this number
num_val_triplets = len(val_dataset)  # You can adjust this number
train_triplets = generate_random_triplets(train_dataset, num_train_triplets)
val_triplets = generate_random_triplets(val_dataset, num_val_triplets)
train_triplet_loader = DataLoader(train_triplets, batch_size=batch_size, shuffle=True)
val_triplet_loader = DataLoader(val_triplets, batch_size=batch_size, shuffle=False)

# Function to apply data transforms and handle grayscale images
def load_and_transform_image(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return data_transforms['train'](image)

# Initialize optimizer and learning rate scheduler
num_epochs = 100  # Set the number of epochs for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataloader), epochs=num_epochs)

# Train the Model with Mixed Precision, Learning Rate Scheduling, and Gradient Clipping
def train_model_with_clipping(model, criterion, optimizer, train_loader, val_loader, scheduler, num_epochs=25, clip_value=1.0):
    scaler = amp.GradScaler()
    
    log = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0

            for i, (anchor, positive, negative) in enumerate(loader):
                anchor = torch.stack([load_and_transform_image(img) for img in anchor])
                positive = torch.stack([load_and_transform_image(img) for img in positive])
                negative = torch.stack([load_and_transform_image(img) for img in negative])

                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with amp.autocast():
                        anchor_output = model(anchor)
                        positive_output = model(positive)
                        negative_output = model(negative)
                        loss = criterion(anchor_output, positive_output, negative_output)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

                running_loss += loss.item() * anchor.size(0)

            epoch_loss = running_loss / len(loader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            log[f'{phase}_loss'].append(epoch_loss)

            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    fine_tuned_model_save_path = f'C:\\Users\\obus\\Desktop\\finTrack\\classification_training_dataset\\train_results\\trainOn35way16shotRealimg\\resnet50_attention_realImg_35W16S_{timestamp}.pth'
                    torch.save(model.state_dict(), fine_tuned_model_save_path)
                    print(f'New best validation loss: {best_val_loss:.4f}. Model saved to {fine_tuned_model_save_path}.')

    return model, log

cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

criterion = TripletLoss(margin=1)

# Fine-tune the model on real image data with OneCycleLR scheduler
model, log = train_model_with_clipping(model, criterion, optimizer, train_triplet_loader, val_triplet_loader, scheduler, num_epochs=num_epochs, clip_value=0.9)

# Save the training log
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_save_path = f'C:\\Users\\obus\\Desktop\\finTrack\\classification_training_dataset\\train_results\\trainOn35way16shotRealimg\\resnet50_attention_realImg_35W16S_{timestamp}.json'
with open(log_save_path, 'w') as f:
    json.dump(log, f)

print(f'Training log saved to {log_save_path}')

# Visualize embeddings using t-SNE
def visualize_embeddings(model, dataset, num_samples=500):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for i, (img_path, label) in enumerate(dataset.imgs):
            if i >= num_samples:
                break
            img = load_and_transform_image(img_path).unsqueeze(0).to(device)
            embedding = model(img).cpu().numpy()
            embeddings.append(embedding)
            labels.append(label)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names)
    plt.title('t-SNE visualization of embeddings')
    plt.show()

visualize_embeddings(model, val_dataset)
