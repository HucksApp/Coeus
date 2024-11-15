import torch
import torch.nn as nn
from torchvision import models
from PIL import Image  # For handling images
import json
import os

from torchvision import datasets
import torchvision.transforms as transforms  # Image transformations
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset  # Data handling classes
import torch.optim as optim

class CoeusClassification(nn.Module):  # Inheriting from nn.Module

    def __init__(self, training=False, dataset_path=None, save_dir=None):
        super(CoeusClassification, self).__init__()
        
        # Use ResNet-50 pre-trained weights if not training
        self.resnet = models.resnet50(weights=None if training else models.ResNet50_Weights.DEFAULT)
        
        self.training = training
        self.save_dir = save_dir
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # If in training mode, set up dataset, modify final layer, and set optimizer/loss
        if training:
            self.brand_dataset = self.create_dataset()
            self.num_classes = len(self.brand_dataset.classes)
            self.update_settings_file("num_classes", self.num_classes)
             # Save class-to-index mapping
            class_to_idx = self.brand_dataset.class_to_idx
            self.update_settings_file("class_to_idx", class_to_idx)

            self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=self.num_classes)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        else:
            self.num_classes = self.get_setting("num_classes")
            path_to_trained = self.get_setting("path_to_trained")
            self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=self.num_classes)
            self.load_state_dict(torch.load(path_to_trained))
            self.to(self.device)
            self.eval()

    def update_settings_file(self, key, value):
        file_path = os.path.join(self.save_dir, "coeus_classify_settings.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
        else:
            data = {}
        data[key] = value
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def get_setting(self, key):
        file_path = os.path.join(self.save_dir, "coeus_classify_settings.json")
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data.get(key)
    
    # Custom loader to handle conversion of 'P' mode images
    def pil_loader_with_transparency(path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                # Convert any non-RGB images (including palette and RGBA) to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return img.copy()  # Ensure persistence of the image file
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None


    def create_dataset(self):
        if self.training and self.dataset_path:
            brand_dataset = datasets.ImageFolder(
                root=self.dataset_path,
                transform=self.transform,
                is_valid_file=lambda x: x.lower().endswith(('.png', '.jpg', '.jpeg'))
            )
            brand_dataset.loader = CoeusClassification.pil_loader_with_transparency  # Set custom loader for handling transparency
            return brand_dataset

    def train_in_progressive(self, epochs_per_run=3):
        # Prepare dataset loaders
        train_size = int(0.8 * len(self.brand_dataset))
        val_size = len(self.brand_dataset) - train_size
        train_dataset, val_dataset = random_split(self.brand_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, persistent_workers=True)


        # Check for existing checkpoint
        checkpoint_path = os.path.join(self.save_dir, "progress_checkpoint.pth")
        start_epoch = 0

        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, start_epoch + epochs_per_run):
            self.train()  # Set to training mode
            running_loss = 0.0
            print(f"Starting Epoch {epoch+1}/{start_epoch + epochs_per_run}...")

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.resnet(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * images.size(0)

                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{start_epoch + epochs_per_run}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # Calculate average loss
            train_loss = running_loss / train_size
            print(f"Epoch [{epoch+1}/{start_epoch + epochs_per_run}], Training Loss: {train_loss:.4f}")

            # Validation phase
            self.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.resnet(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = correct / total * 100
            print(f"Epoch [{epoch+1}/{start_epoch + epochs_per_run}], Validation Accuracy: {val_accuracy:.2f}%")

            # Save checkpoint after each epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': train_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

        # Save final model weights
        self.save_trained()

    def save_trained(self):
        trained_path = os.path.join(self.save_dir, "trained_model.pth")
        self.update_settings_file("path_to_trained", trained_path)
        torch.save(self.state_dict(), trained_path)
        print(f"Model saved to {trained_path}")

    def predict_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Load the class-to-index mapping
        class_to_idx = self.get_setting("class_to_idx")
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        with torch.no_grad():
            output = self.resnet(image)
            _, predicted = torch.max(output, 1)
        
        # Get predicted class using the loaded mapping
        predicted_class = idx_to_class[predicted.item()]
        print(f"Predicted Class: {predicted_class}")
        return predicted_class

