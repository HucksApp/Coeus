import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from PIL import Image
import os
import json


class CoeusClassification(nn.Module):
    def __init__(self, training=False, dataset_path=None, save_dir=None, title=None):
        super(CoeusClassification, self).__init__()

        # Title-based settings for the model
        self.title = title
        self.save_dir = os.path.join(save_dir, title) if title else save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Load the base model (ResNet-50)
        self.training = training
        self.dataset_path = dataset_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet50(
            weights=None if training else models.ResNet50_Weights.DEFAULT)
        self.reference_models = {}  # To store other models

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        # Training setup
        if training:
            self.brand_dataset = self.create_dataset()
            self.num_classes = len(self.brand_dataset.classes)
            self.update_settings_file("num_classes", self.num_classes)

            # Save class-to-index mapping
            class_to_idx = self.brand_dataset.class_to_idx
            self.update_settings_file("class_to_idx", class_to_idx)

            # Replace the final layer for classification
            self.resnet.fc = nn.Linear(
                self.resnet.fc.in_features, self.num_classes)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        else:
            self.num_classes = self.get_setting("num_classes")
            path_to_trained = self.get_setting("path_to_trained")
            self.resnet.fc = nn.Linear(
                self.resnet.fc.in_features, self.num_classes)
            self.load_state_dict(torch.load(path_to_trained))
            self.to(self.device)
            self.eval()

        # Load referenced classification models
        referenced_models = self.get_setting("referenced_models") or {}
        for key, model_path in referenced_models.items():
            self.reference_models[key] = self.load_other_model(
                model_path, "resnet50")

    ### SETTINGS MANAGEMENT ###
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
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data.get(key)
        return None

    ### DATASET CREATION ###
    @staticmethod
    def pil_loader_with_transparency(path):
        """Handles loading of images with transparency."""
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return img.copy()
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def create_dataset(self, selected_classes=None):
        if self.training and self.dataset_path:
            # Filter the dataset classes based on selected_classes
            all_classes = os.listdir(self.dataset_path)
            if selected_classes:
                all_classes = [
                    cls for cls in all_classes if cls in selected_classes]

            brand_dataset = datasets.ImageFolder(
                root=self.dataset_path,
                transform=self.transform,
                is_valid_file=lambda x: x.lower().endswith(('.png', '.jpg', '.jpeg'))
            )
            brand_dataset.loader = CoeusClassification.pil_loader_with_transparency
            # Adjust the class list based on selected_classes
            brand_dataset.classes = all_classes
            brand_dataset.class_to_idx = {
                cls: idx for idx, cls in enumerate(all_classes)}

            return brand_dataset

    ### TRAINING ###

    def train_in_progressive(self, epochs_per_run=3):
        train_size = int(0.8 * len(self.brand_dataset))
        val_size = len(self.brand_dataset) - train_size
        train_dataset, val_dataset = random_split(
            self.brand_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        checkpoint_path = os.path.join(
            self.save_dir, "progress_checkpoint.pth")
        start_epoch = 0

        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, start_epoch + epochs_per_run):
            self.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.resnet(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * images.size(0)

            # Validation
            self.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(
                        self.device), labels.to(self.device)
                    outputs = self.resnet(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = correct / total * 100
            print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.2f}%")

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, checkpoint_path)

        self.save_trained()

    def save_trained(self):
        trained_path = os.path.join(self.save_dir, "trained_model.pth")
        self.update_settings_file("path_to_trained", trained_path)
        torch.save(self.state_dict(), trained_path)

        # Save TorchScript serialized model
        scripted_path = os.path.join(
            self.save_dir, "trained_model_scripted.pth")
        scripted_model = torch.jit.script(self.resnet)
        scripted_model.save(scripted_path)
        self.update_settings_file("path_to_scripted", scripted_path)

    ### REFERENCED MODELS ###

    def load_other_model(self, model_path, model_type, selected_classes=None, detection_classes=None):
        if model_type.lower() == "resnet50":
            # Classification model
            other_model = models.resnet50(weights=None)
            # If selected_classes are provided, set the number of classes to the length of selected_classes
            num_classes = len(
                selected_classes) if selected_classes else self.get_setting("num_classes")
            other_model.fc = nn.Linear(other_model.fc.in_features, num_classes)
            other_model.load_state_dict(torch.load(
                model_path, map_location=self.device))

            # Store the selected classes for later filtering during inference
            other_model.selected_classes = selected_classes

            return other_model.to(self.device)

        elif model_type.lower() == "fasterrcnn":
            # Object detection model (e.g., Faster R-CNN)
            other_model = fasterrcnn_resnet50_fpn(weights=None)
            other_model.load_state_dict(torch.load(
                model_path, map_location=self.device))
            other_model.eval()  # Detection models are often used in eval mode
            # Store selected classes for detection, if provided
            other_model.selected_classes = selected_classes
            # Store detection classes mapping
            other_model.detection_classes = detection_classes

            return other_model.to(self.device)

        elif model_type.lower() == "mobilenetv2":
            # MobileNetV2 model for classification
            other_model = mobilenet_v2(weights=None)
            # If selected_classes are provided, set the number of classes to the length of selected_classes
            num_classes = len(
                selected_classes) if selected_classes else self.get_setting("num_classes")
            other_model.classifier[1] = nn.Linear(
                other_model.classifier[1].in_features, num_classes)
            other_model.load_state_dict(torch.load(
                model_path, map_location=self.device))

            # Store the selected classes for later filtering during inference
            other_model.selected_classes = selected_classes

            return other_model.to(self.device)

        else:
            raise NotImplementedError(
                f"Model type {model_type} is not supported.")

    def predict_with_selected_classes(self, model, image, selected_classes):
        # Get model predictions (classification)
        model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            # Model output, typically (batch_size, num_classes)
            outputs = model(image)

        # Filter predictions based on selected classes
        if selected_classes:
            # Get the predicted class index
            predicted_class = outputs.argmax(dim=1).item()

            if predicted_class not in selected_classes:
                return None  # No valid class prediction

            return predicted_class  # Return the predicted class within selected classes
        else:
            # No filtering, return any prediction
            return outputs.argmax(dim=1).item()

    def detections_by_selected_classes(self, detections, selected_classes):
        # Filter out detections not belonging to selected classes
        filtered_detections = []
        for detection in detections:
            # Assuming detection['labels'] is a tensor of class labels for each detected object
            mask = torch.isin(detection['labels'],
                              torch.tensor(selected_classes))
            filtered_detections.append({
                'boxes': detection['boxes'][mask],
                'labels': detection['labels'][mask],
                'scores': detection['scores'][mask],
            })
        return filtered_detections

    def load_detection_classes(self):
        detection_classes_path = os.path.join(
            self.save_dir, "detection_classes.json")
        if os.path.exists(detection_classes_path):
            with open(detection_classes_path, 'r') as file:
                return json.load(file)
        return {}

    def save_detection_classes(self, detection_classes):
        detection_classes_path = os.path.join(
            self.save_dir, "detection_classes.json")
        with open(detection_classes_path, 'w') as file:
            json.dump(detection_classes, file, indent=4)

    def add_referenced_model(self, key, model_path, model_type="resnet50", detection_classes=None, selected_classes=None):
        self.reference_models[key] = self.load_other_model(
            model_path, model_type, selected_classes)

        if model_type.lower() == "fasterrcnn" and detection_classes:
            # Save detection classes if applicable
            self.save_detection_classes(detection_classes)

        # Update settings to include the new model reference
        referenced_models = self.get_setting("referenced_models") or {}
        referenced_models[key] = {"path": model_path, "type": model_type}
        self.update_settings_file("referenced_models", referenced_models)

    def save_torchscript_model(self):
        trained_path = os.path.join(
            self.save_dir, "trained_model_scripted.pth")
        # or torch.jit.trace for tracing
        scripted_model = torch.jit.script(self.resnet)
        scripted_model.save(trained_path)
        self.update_settings_file("path_to_scripted", trained_path)

    def load_torchscript_model(self):
        """Load the TorchScript model for inference."""
        scripted_model_path = self.get_setting("path_to_scripted")
        if scripted_model_path and os.path.exists(scripted_model_path):
            self.resnet = torch.jit.load(scripted_model_path)
            self.resnet.to(self.device)
            self.resnet.eval()

    ### PREDICTION ###

    def predict_image(self, image_path):
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            self.load_state_dict(torch.load(self.get_setting(
                "path_to_trained"), map_location=self.device))
            self.model_loaded = True  # Ensure we don't load it again

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.resnet(image)
            _, predicted = torch.max(output, 1)

        idx_to_class = {v: k for k, v in self.get_setting(
            "class_to_idx").items()}
        return idx_to_class[predicted.item()]

    def predict_image_with_references(self, image_path, selected_classes=None):
        primary_prediction = self.predict_image(image_path)
        references = {}
        detection_classes = self.load_detection_classes()

        image = Image.open(image_path).convert('RGB')
        transformed_image = self.transform(image).unsqueeze(0).to(self.device)

        for key, model in self.reference_models.items():
            model_info = self.get_setting("referenced_models")[key]
            model_type = model_info.get("type", "resnet50").lower()

            if model_type == "resnet50":
                predicted_class = self.predict_with_selected_classes(
                    model, transformed_image, selected_classes)
                if predicted_class is not None:
                    idx_to_class = {v: k for k, v in self.get_setting(
                        "class_to_idx").items()}
                    references[key] = idx_to_class[predicted_class]

            elif model_type == "fasterrcnn":
                from torchvision.transforms.functional import to_tensor
                detection_image = to_tensor(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    detections = model(detection_image)[0]

                predictions = []
                for box, label, score in zip(detections["boxes"], detections["labels"], detections["scores"]):
                    if score > 0.5:  # Confidence threshold
                        # Only include detections in the selected classes
                        if label.item() in selected_classes:
                            predictions.append({
                                "box": box.cpu().tolist(),
                                "label": detection_classes.get(label.item(), f"Unknown ({label.item()})"),
                                "score": score.item()
                            })
                references[key] = predictions

        return {"primary": primary_prediction, "references": references}
