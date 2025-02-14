import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import itertools
from PIL import Image

# =====================
# 🔹 Step 1: Set Dataset Paths and Custom Dataset
# =====================
FER_PATH = "FER2013"
UTKFACE_PATH = "UTKFace"

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            age = float(img_name.split('_')[0])
        except:
            age = 0.0
            
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(age, dtype=torch.float32)

# Check if datasets exist
if not os.path.exists(FER_PATH) or not os.path.exists(UTKFACE_PATH):
    raise FileNotFoundError("Datasets not found in the directory!")

# =====================
# 🔹 Step 2: Define Image Transformations
# =====================
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Lambda(lambda x: x.to(torch.float32))
])

# =====================
# 🔹 Step 3: Load Datasets
# =====================
try:
    fer_dataset = datasets.ImageFolder(root=FER_PATH, transform=transform)
    utk_dataset = UTKFaceDataset(root_dir=UTKFACE_PATH, transform=transform)
    
    batch_size = 32
    fer_loader = DataLoader(fer_dataset, batch_size=batch_size, shuffle=True)
    utk_loader = DataLoader(utk_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"FER2013 Classes: {fer_dataset.classes}")
    print(f"Datasets loaded - FER: {len(fer_dataset)}, UTK: {len(utk_dataset)} images")
except Exception as e:
    print(f"Error loading datasets: {str(e)}")
    raise

# =====================
# 🔹 Step 4: Define CNN Model
# =====================
class MultiTaskCNN(nn.Module):
    def __init__(self, num_sentiments, num_stances):
        super(MultiTaskCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.relu = nn.ReLU()
        
        self.sentiment_out = nn.Linear(512, num_sentiments)
        self.stance_out = nn.Linear(512, num_stances)
        self.age_out = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        
        sentiment_pred = self.sentiment_out(x)
        stance_pred = self.stance_out(x)
        age_pred = self.age_out(x)

        return sentiment_pred, stance_pred, age_pred

# =====================
# 🔹 Step 5: Setup Model, Loss, and Optimizer
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_sentiments = len(fer_dataset.classes)
num_stances = 3

model = MultiTaskCNN(num_sentiments, num_stances).to(device)
model = model.to(torch.float32)

criterion_sentiment = nn.CrossEntropyLoss()
criterion_stance = nn.CrossEntropyLoss()
criterion_age = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Print model parameter types
print(f"Model Parameters Data Type: {next(model.parameters()).dtype}")

# =====================
# 🔹 Step 6: Training Loop
# =====================
num_epochs = 5
print(f"Training on {device}")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Create iterator for UTK dataset
    utk_iterator = iter(utk_loader)
    
    for i, (images_fer, labels_fer) in enumerate(fer_loader):
        # Get UTK batch with proper error handling
        try:
            images_utk, ages_utk = next(utk_iterator)
        except StopIteration:
            utk_iterator = iter(utk_loader)
            images_utk, ages_utk = next(utk_iterator)

        # Ensure batch sizes match
        if images_fer.size(0) != images_utk.size(0):
            continue
            
        # Move to device and set data types
        images_fer = images_fer.to(device, dtype=torch.float32)
        labels_fer = labels_fer.to(device)
        images_utk = images_utk.to(device, dtype=torch.float32)
        ages_utk = ages_utk.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        
        # Forward pass
        sentiment_pred, stance_pred, age_pred = model(images_fer)
        
        # Compute losses ensuring tensor sizes match
        loss_sentiment = criterion_sentiment(sentiment_pred, labels_fer)
        loss_stance = criterion_stance(stance_pred, labels_fer % 3)
        loss_age = criterion_age(age_pred.squeeze(), ages_utk[:age_pred.size(0)])

        total_loss = loss_sentiment + loss_stance + loss_age
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(fer_loader)}], Loss: {total_loss.item():.4f}")
    
    epoch_loss = running_loss / len(fer_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Complete, Avg Loss: {epoch_loss:.4f}")
# =====================
# 🔹 Step 7: Inference Function
# =====================
def predict(image_path):
    model.eval()
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device, dtype=torch.float32)

        with torch.no_grad():
            sentiment_pred, stance_pred, age_pred = model(image)

        sentiment_label = torch.argmax(sentiment_pred, dim=1).item()
        stance_label = torch.argmax(stance_pred, dim=1).item()
        age_value = age_pred.item()

        sentiment_map = fer_dataset.classes
        stance_map = ["Support", "Neutral", "Oppose"]

        print(f"Predicted Sentiment: {sentiment_map[sentiment_label]}")
        print(f"Predicted Stance: {stance_map[stance_label]}")
        print(f"Predicted Age: {int(age_value)}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

# =====================
# 🔹 Step 8: Save Model and Run Inference
# =====================
if __name__ == "__main__":
    try:
        # Save the trained model
        torch.save(model.state_dict(), "multitask_model.pth")
        print("Model saved successfully")
        
        # Run inference on a sample image
        sample_image_path = os.path.join(FER_PATH, "happy", "PrivateTest_95094.jpg")
        if os.path.exists(sample_image_path):
            predict(sample_image_path)
        else:
            print(f"Sample image not found at: {sample_image_path}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise