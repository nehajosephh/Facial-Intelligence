import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import time
import sys
import multiprocessing

# =====================
# 🔹 Step 1: Set Dataset Path and Custom Dataset
# =====================
UTKFACE_PATH = "UTKFace"

def check_gpu_memory():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        print(f"Available GPU memory: {gpu_memory:.2f} GB")
        return gpu_memory
    return 0

def adjust_batch_size(gpu_memory):
    if gpu_memory < 4:  # Less than 4GB GPU memory
        return 16
    elif gpu_memory < 8:  # Less than 8GB GPU memory
        return 32
    return 64  # Default for larger GPUs

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            age = float(img_name.split('_')[0])
        except:
            print(f"Warning: Could not parse age from filename: {img_name}")
            age = 0.0
            
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(age, dtype=torch.float32)

def main():
    # Check if dataset exists and create if needed
    if not os.path.exists(UTKFACE_PATH):
        print(f"Error: Dataset not found at {UTKFACE_PATH}")
        print("Please download the UTKFace dataset and place it in the current directory")
        print("You can download it from: https://susanqq.github.io/UTKFace/")
        sys.exit(1)

    # =====================
    # 🔹 Step 2: Define Image Transformations
    # =====================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # =====================
    # 🔹 Step 3: Load and Split Dataset
    # =====================
    try:
        # Check GPU memory and adjust batch size
        gpu_memory = check_gpu_memory()
        batch_size = adjust_batch_size(gpu_memory)
        num_workers = 0  # Set to 0 for Windows to avoid multiprocessing issues
        
        print(f"Using batch size: {batch_size}, workers: {num_workers}")
        
        # Load full dataset
        full_dataset = UTKFaceDataset(root_dir=UTKFACE_PATH, transform=train_transform)
        
        # Split into train and validation sets
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Apply validation transform to validation set
        val_dataset.dataset.transform = val_transform
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        print(f"Dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)} images")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        sys.exit(1)

    # =====================
    # 🔹 Step 4: Load Pre-trained Model
    # =====================
    def create_model():
        try:
            # Load pre-trained ResNet18
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception as e:
            print(f"Warning: Could not load pre-trained model: {str(e)}")
            print("Using untrained model instead")
            model = models.resnet18(weights=None)
        
        # Freeze all layers except the last few
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last few layers
        for param in model.layer4.parameters():
            param.requires_grad = True
        
        # Modify the final layer for age prediction
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        return model

    # =====================
    # 🔹 Step 5: Setup Model, Loss, and Optimizer
    # =====================
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = create_model().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    except Exception as e:
        print(f"Error setting up model: {str(e)}")
        sys.exit(1)

    # =====================
    # 🔹 Step 6: Training Loop with Validation
    # =====================
    num_epochs = 10
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            start_time = time.time()
            
            for i, (images, ages) in enumerate(train_loader):
                try:
                    images = images.to(device)
                    ages = ages.to(device)

                    optimizer.zero_grad()
                    
                    age_pred = model(images)
                    loss = criterion(age_pred.squeeze(), ages)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    if (i + 1) % 50 == 0:
                        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: out of memory")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, ages in val_loader:
                    try:
                        images = images.to(device)
                        ages = ages.to(device)
                        
                        age_pred = model(images)
                        loss = criterion(age_pred.squeeze(), ages)
                        val_loss += loss.item()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print("WARNING: out of memory during validation")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
            
            # Calculate average losses
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                try:
                    torch.save(model.state_dict(), "best_age_model.pth")
                except Exception as e:
                    print(f"Warning: Could not save model: {str(e)}")
            else:
                patience_counter += 1
            
            # Print epoch statistics
            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{num_epochs}] Complete")
            print(f"Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
                
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)

    # =====================
    # 🔹 Step 7: Inference Function
    # =====================
    def predict_age(image_path):
        model.eval()
        
        try:
            if not os.path.exists(image_path):
                print(f"Error: Image not found at {image_path}")
                return
                
            image = Image.open(image_path).convert('RGB')
            image = val_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                age_pred = model(image)

            age_value = age_pred.item()
            print(f"Predicted Age: {int(age_value)}")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise

    # =====================
    # 🔹 Step 8: Run Inference
    # =====================
    try:
        # Run inference on a sample image
        sample_image_path = os.path.join(UTKFACE_PATH, "8_1_0_20170109201705598.jpg.chip.jpg")
        if os.path.exists(sample_image_path):
            predict_age(sample_image_path)
        else:
            print(f"Sample image not found at: {sample_image_path}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()