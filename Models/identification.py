from PIL import Image
import os
import json
import torch
import random
import math
import torch.nn as nn
from torchvision.ops import nms
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
import torchvision.transforms as T
from sklearn.metrics import average_precision_score, precision_recall_curve
from Models.coeus_base import CoeusBase


class CoeusIdentification(nn.Module, CoeusBase):
    def __init__(self, title, training=False, dataset_path=None, save_dir=None):
        super(CoeusIdentification, self).__init__()
        CoeusBase.__init__(self)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.title = title

        self.save_dir = os.path.join(self.save_dir, "identify")
        os.makedirs(self.save_dir, exist_ok=True)

        # Use title to organize model-specific settings
        self.save_dir = os.path.join(save_dir, title)
        self.dataset_path = dataset_path
        os.makedirs(self.save_dir, exist_ok=True)

        # Load pre-trained Faster R-CNN with ResNet50 backbone
        self.fasterrcnn = fasterrcnn_resnet50_fpn(pretrained=True)

        if training:
            # Fine-tune the classifier head (bounding box predictor)
            in_features = self.fasterrcnn.roi_heads.box_predictor.cls_score.in_features
            num_classes = self.get_num_classes()

            # Modify the classifier head based on selected classes
            self.fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes)

            self.dataset = self.create_dataset()
            self.class_to_idx = self.dataset.class_to_idx
            self.save_class_mappings()

        # Move model to device (GPU/CPU)
        self.fasterrcnn.to(self.device)

        # Load referenced models
        referenced_models = self.get_setting("referenced_models") or {}
        self.create_reference_models(referenced_models)

    def save_torchscript_model(self):
        """ Save the TorchScript model for deployment. """
        trained_path = os.path.join(
            self.save_dir, "trained_model_scripted.pth")
        # Convert the Faster R-CNN model to TorchScript
        # or torch.jit.trace(self.fasterrcnn) for tracing
        scripted_model = torch.jit.script(self.fasterrcnn)
        scripted_model.save(trained_path)
        self.update_settings_file("path_to_scripted", trained_path)
        print(f"TorchScript model saved to {trained_path}")

    def load_torchscript_model(self):
        """ Load the TorchScript model for inference. """
        scripted_model_path = self.get_setting("path_to_scripted")
        if scripted_model_path and os.path.exists(scripted_model_path):
            self.fasterrcnn = torch.jit.load(scripted_model_path)
            self.fasterrcnn.to(self.device)
            self.fasterrcnn.eval()
            print(f"TorchScript model loaded from {scripted_model_path}")
        else:
            raise FileNotFoundError("TorchScript model not found!")

    def filter_classes(self, class_to_idx):
        """ Filter classes to include only the selected ones. """
        if self.class_selection is not None:
            filtered_classes = {
                cls: idx for cls, idx in class_to_idx.items() if cls in self.class_selection}
            return filtered_classes
        return class_to_idx

    def save_class_mappings(self):
        """ Save detection class mappings (class_to_idx) in a JSON file for portability. """
        file_path = os.path.join(self.save_dir, 'detection_classes.json')
        with open(file_path, 'w') as file:
            json.dump(self.class_to_idx, file, indent=4)

    def load_class_mappings(self):
        """ Load detection class mappings from a JSON file. """
        file_path = os.path.join(self.save_dir, 'detection_classes.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                self.class_to_idx = json.load(file)
        else:
            raise FileNotFoundError("Detection class mappings not found!")


    def create_dataset(self):
        """ Create the dataset object. """
        if self.dataset_path:
            annotations_file = os.path.join(
                self.dataset_path, "annotations.json")
            transform = CustomTransform()
            dataset = CustomDataset(
                root=self.dataset_path,
                annotations_file=annotations_file,
                transform=transform
            )
            return dataset
        else:
            raise ValueError("Dataset path not provided.")

    def get_num_classes(self):
        """ Get number of classes for detection model. """
        return len(self.create_dataset().classes) if self.dataset_path else 91

    def get_setting(self, key):
        """ Retrieve setting from the settings file. """
        file_path = os.path.join(self.save_dir, "coeus_identify_settings.json")
        with open(file_path, 'r') as file:
            settings = json.load(file)
        return settings.get(key)

    def update_settings_file(self, key, value):
        """ Update settings file for model-specific configurations. """
        file_path = os.path.join(self.save_dir, "coeus_identify_settings.json")
        settings = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                settings = json.load(file)
        settings[key] = value
        with open(file_path, 'w') as file:
            json.dump(settings, file, indent=4)

    def save_trained(self):
        """ Save the trained Faster R-CNN model and its settings. """
        trained_path = os.path.join(self.save_dir, "fasterrcnn_trained.pth")
        torch.save(self.fasterrcnn.state_dict(), trained_path)
        self.update_settings_file("path_to_trained", trained_path)
        print(f"Model saved to {trained_path}")

    def train_in_progressive(self, epochs_per_run=3):
        """ Train the model incrementally with checkpointing. """
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(
            self.dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        optimizer = optim.SGD(self.fasterrcnn.parameters(
        ), lr=0.005, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1)

        checkpoint_path = os.path.join(
            self.save_dir, "progress_checkpoint.pth")
        start_epoch = 0

        # Load checkpoint if exists
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            self.fasterrcnn.load_state_dict(checkpoint['model_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

        # Training loop
        for epoch in range(start_epoch, start_epoch + epochs_per_run):
            self.fasterrcnn.train()
            running_loss = 0.0

            for images, targets in train_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device)
                            for k, v in target.items()} for target in targets]

                optimizer.zero_grad()
                loss_dict = self.fasterrcnn(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                scheduler.step()
                optimizer.step()
                running_loss += losses.item()

            print(
                f"Epoch [{epoch + 1}], Loss: {running_loss / len(train_loader):.4f}")

            # Validation
            self.fasterrcnn.eval()
            with torch.no_grad():
                all_labels, all_scores = [], []
                for images, targets in val_loader:
                    images = [image.to(self.device) for image in images]
                    predictions = self.fasterrcnn(images)

                    for target, pred in zip(targets, predictions):
                        labels = target['labels'].cpu().numpy()
                        scores = pred['scores'].cpu().numpy()
                        all_labels.extend(labels)
                        all_scores.extend(scores)

                # Calculate precision-recall curve and average precision
                precision, recall, _ = precision_recall_curve(
                    all_labels, all_scores)
                ap = average_precision_score(all_labels, all_scores)
                print(f"Precision: {
                      precision[-1]:.4f}, Recall: {recall[-1]:.4f}, AP: {ap:.4f}")

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.fasterrcnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)

        # Save the trained model explicitly
        self.save_trained()

    def compute_mean_ap(self, all_labels, all_scores):
        # Compute mAP for multi-class
        # The `average_precision_score` expects binary labels per class, so you need to compute for each class
        num_classes = self.num_classes
        aps = []
        for class_id in range(num_classes):
            binary_labels = [1 if label ==
                             class_id else 0 for label in all_labels]
            binary_scores = [score for i, score in enumerate(
                all_scores) if all_labels[i] == class_id]
            if len(binary_scores) > 0:
                aps.append(average_precision_score(
                    binary_labels, binary_scores))
        return sum(aps) / len(aps) if aps else 0.0

    def load_trained_model(self):
        trained_path = self.get_setting("path_to_trained")
        if not trained_path or not os.path.exists(trained_path):
            raise FileNotFoundError(
                "Trained model not found! Ensure training is complete and the model is saved.")
        self.fasterrcnn.load_state_dict(torch.load(
            trained_path, map_location=self.device))
        self.fasterrcnn.to(self.device)
        self.fasterrcnn.eval()
        print(f"Model loaded from {trained_path}")

    def predict_image(self, image_path, selected_classes=None, threshold=0.5, iou_threshold=0.5):
        """Predict objects in an image and optionally filter by selected classes."""
        # Load the trained model if not already loaded
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            self.load_trained_model()
            self.model_loaded = True

        # Ensure class mappings are loaded
        if not hasattr(self, 'class_to_idx'):
            self.load_class_mappings()

        # Convert selected class names to indices
        selected_class_indices = None
        if selected_classes:
            selected_class_indices = set(
                idx for cls, idx in self.class_to_idx.items() if cls in selected_classes
            )

        # Set the model to evaluation mode
        self.fasterrcnn.eval()

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get raw predictions
            raw_predictions = self.fasterrcnn(image_tensor)[0]

            # Apply NMS
            filtered_indices = nms(
                raw_predictions['boxes'], raw_predictions['scores'], iou_threshold
            )
            predictions = {key: val[filtered_indices].cpu()
                           for key, val in raw_predictions.items()}

            # Filter by confidence threshold
            confidence_mask = predictions['scores'] > threshold
            predictions = {
                key: val[confidence_mask] for key, val in predictions.items()
            }

            # Filter by selected classes, if provided
            if selected_class_indices is not None:
                class_mask = torch.tensor(
                    [label in selected_class_indices for label in predictions['labels']],
                    dtype=torch.bool
                )
                predictions = {
                    key: val[class_mask] for key, val in predictions.items()
                }

        # Handle empty predictions
        if len(predictions['boxes']) == 0:
            return {"boxes": [], "labels": [], "scores": []}

        # Convert class indices to names
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        predictions['labels'] = [
            idx_to_class.get(label.item(), "Unknown") for label in predictions['labels']
        ]

        return predictions


class CustomDataset(Dataset):
    def __init__(self, root, annotations_file, transform=None):
        self.root = root
        self.annotations = json.load(open(annotations_file))
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.annotations[idx]['file_name'])
        image = Image.open(img_path).convert("RGB")
        boxes = self.annotations[idx]['annotations']

        # Extract bounding boxes, labels, and model references
        boxes = [box['bbox'] for box in boxes]
        labels = [box['category_id'] for box in boxes]
        model_refs = [box['model_ref'] for box in boxes]

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'model_refs': model_refs
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.annotations)


class CustomTransform:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize((800, 800)),  # Resize to standard size
            transforms.RandomHorizontalFlip(),  # Flip image horizontally
            # Random color adjustments
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # Random rotation within a range
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(800, scale=(0.8, 1.0), ratio=(
                0.75, 1.33)),  # Random crop with scaling
        ])

    def __call__(self, image, target):
        # Apply image transformations
        image = self.transforms(image)

        # Adjust bounding boxes for transformation
        boxes = target['boxes']

        # Apply random rotation logic
        angle = random.randint(-10, 10)
        image = F.rotate(image, angle)
        boxes = self.rotate_boxes(boxes, angle, image.size)

        # For flip, use the helper flip_boxes function
        if random.random() < 0.5:  # Random Horizontal Flip
            image = F.hflip(image)
            boxes = self.flip_boxes(boxes, image.width)

        target['boxes'] = boxes
        return image, target

    def rotate_boxes(self, boxes, angle, image_size):
        """ Rotate bounding boxes based on the image's rotation angle. """
        angle_rad = math.radians(angle)
        width, height = image_size
        center_x, center_y = width / 2, height / 2  # Rotate around the center

        rotated_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            # Get the box's center and size
            box_center_x = (xmin + xmax) / 2
            box_center_y = (ymin + ymax) / 2
            box_width = xmax - xmin
            box_height = ymax - ymin

            # Apply rotation to the box's center
            new_center_x = center_x + (box_center_x - center_x) * math.cos(
                angle_rad) - (box_center_y - center_y) * math.sin(angle_rad)
            new_center_y = center_y + (box_center_x - center_x) * math.sin(
                angle_rad) + (box_center_y - center_y) * math.cos(angle_rad)

            # Convert back to box corners
            new_xmin = new_center_x - box_width / 2
            new_ymin = new_center_y - box_height / 2
            new_xmax = new_center_x + box_width / 2
            new_ymax = new_center_y + box_height / 2

            rotated_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])

        return torch.tensor(rotated_boxes, dtype=torch.float32)

    def flip_boxes(self, boxes, image_width):
        """ Flip bounding boxes horizontally. """
        flipped_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            flipped_boxes.append(
                [image_width - xmax, ymin, image_width - xmin, ymax])
        return torch.tensor(flipped_boxes, dtype=torch.float32)
