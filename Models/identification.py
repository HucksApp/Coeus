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
from torchvision import models, transforms
from torchvision.transforms import functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import mean_absolute_error


class CoeusIdentification(torch.nn.Module):
    def __init__(self, title, training=False, dataset_path=None, save_dir=None, class_selection=None):
        super(CoeusIdentification, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.title = title
        # Use title to organize model-specific settings
        self.save_dir = os.path.join(save_dir, title)
        self.dataset_path = dataset_path
        self.class_selection = class_selection  # Store class selection

        os.makedirs(self.save_dir, exist_ok=True)

        # Load pre-trained Faster R-CNN with ResNet50 backbone
        self.fasterrcnn = fasterrcnn_resnet50_fpn(pretrained=True)

        if training:
            # Fine-tune the classifier head (bounding box predictor)
            in_features = self.fasterrcnn.roi_heads.box_predictor.cls_score.in_features
            num_classes = self.get_num_classes()

            # Modify the classifier head based on selected classes
            self.fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, len(self.class_selection) or num_classes)

            # Save class mappings to persistent JSON file
            self.dataset = self.create_dataset()
            self.class_to_idx = self.filter_classes(self.dataset.class_to_idx)
            # Check for missing classes in dataset and raise an error if found
            available_classes = set(self.dataset.class_to_idx.values())
            missing_classes = set(self.class_selection) - available_classes
            if missing_classes:
                raise ValueError(f"Selected classes not found in dataset: {
                                 missing_classes}")

            self.save_class_mappings()

        # Move model to device (GPU/CPU)
        self.fasterrcnn.to(self.device)

        # Load other models for classification (optional)
        self.reference_models = {
            "car_models_id": self.load_other_model(f"{self.save_dir}/classify/path_to_trained.pth", "resnet50"),
            # Add more models if necessary...
        }

    def save_torchscript_model(self):
        """ Save the TorchScript model for deployment. """
        trained_path = os.path.join(self.save_dir, "trained_model_scripted.pth")
        # Convert the Faster R-CNN model to TorchScript
        scripted_model = torch.jit.script(self.fasterrcnn)  # or torch.jit.trace(self.fasterrcnn) for tracing
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

    def load_other_model(self, model_path, model_type="resnet50"):
        """ Load a classification model (e.g., ResNet50, VGG16). """
        if model_type == "resnet50":
            model = models.resnet50(pretrained=True)
        elif model_type == "vgg16":
            model = models.vgg16(pretrained=True)
        model.load_state_dict(torch.load(model_path))
        model.to(self.device).eval()
        return model

    def apply_classification(self, image, model_ref):
        """ Classify car part using the specified classification model. """
        if model_ref in self.reference_models:
            model = self.reference_models[model_ref]
            return self.classify_part(image, model)
        else:
            print(f"Model reference '{model_ref}' not found.")
            return None

    def classify_part(self, image, model):
        """ Classify a part of the car using the classification model. """
        image_tensor = T.to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        return torch.argmax(probabilities, dim=1)

    def apply_nms(self, predictions, iou_threshold):
        boxes = predictions['boxes']
        scores = predictions['scores']
        keep = nms(boxes, scores, iou_threshold)
        return [predictions[idx] for idx in keep]

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

    def load_trained_model(self):
        """ Load the trained Faster R-CNN model. """
        trained_path = self.get_setting("path_to_trained")
        self.fasterrcnn.load_state_dict(torch.load(
            trained_path, map_location=self.device))
        # Ensure model is in eval mode after loading
        self.fasterrcnn.to(self.device).eval()
        for model in self.reference_models.values():
            model.eval()  # Set classification models to eval mode
        print(f"Model loaded from {trained_path}")

    def get_num_classes(self):
        """ Get number of classes for detection model. """
        return len(self.create_dataset().classes) if self.dataset_path else 91

    def get_setting(self, key):
        """ Retrieve setting from the settings file. """
        file_path = os.path.join(self.save_dir, "settings.json")
        with open(file_path, 'r') as file:
            settings = json.load(file)
        return settings.get(key)

    def update_settings_file(self, key, value):
        """ Update settings file for model-specific configurations. """
        file_path = os.path.join(self.save_dir, "settings.json")
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

    def predict_image(self, image_path, threshold=0.5, iou_threshold=0.5):
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            self.load_trained_model()
            self.model_loaded = True

        self.fasterrcnn.eval()

        image = Image.open(image_path).convert('RGB')
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_prediction = self.fasterrcnn(image_tensor)[0]

            # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
            predictions = self.apply_nms(
                raw_prediction, iou_threshold=iou_threshold)

            # Filter predictions by selected classes
            filtered_predictions = [
                pred for pred in predictions if pred['labels'].item() in self.class_selection]

            # Filter predictions by score threshold
            for box, label, score, model_ref in zip(filtered_predictions['boxes'], filtered_predictions['labels'], filtered_predictions['scores'], filtered_predictions['model_refs']):
                if score > threshold:
                    print(f"Detected: Label={label.item()}, Score={
                          score:.2f}, Box={box.cpu().numpy()}")

                    # Check if model_ref is valid
                    if model_ref in self.reference_models:
                        referenced_model = self.reference_models[model_ref]
                        referenced_result = self.apply_classification(
                            image, referenced_model)
                        print(f"Classified as: {referenced_result.item()}")
                    else:
                        print(f"Model reference '{model_ref}' not found!")


class CustomDataset(Dataset):
    def __init__(self, root, annotations_file, transform=None, class_selection=None):
        self.root = root
        self.annotations = json.load(open(annotations_file))
        self.transform = transform
        self.class_selection = class_selection  # Store selected classes

        # Validate class selection
        self.class_selection = set(self.class_selection or [])

        # Filter annotations to keep only the selected classes
        self.annotations = [
            ann for ann in self.annotations if any(
                box['category_id'] in self.class_selection for box in ann['annotations']
            )
        ]

        # Ensure all selected classes are part of the dataset
        available_classes = set(
            box['category_id'] for ann in self.annotations for box in ann['annotations'])
        missing_classes = self.class_selection - available_classes
        if missing_classes:
            raise ValueError(f"Selected classes not found in dataset: {
                             missing_classes}")

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.annotations[idx]['file_name'])
        image = Image.open(img_path).convert("RGB")
        boxes = self.annotations[idx]['annotations']

        # Filter boxes based on class selection
        boxes = [box['bbox']
                 for box in boxes if box['category_id'] in self.class_selection]
        labels = [box['category_id'] for box in boxes]

        # Model references for classification
        model_refs = [box['model_ref']
                      for box in boxes if box['category_id'] in self.class_selection]

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'model_refs': model_refs  # Add model references for classification
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


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
