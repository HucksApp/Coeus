import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from PIL import Image
import os
import json


class CoeusGenerative(nn.Module):
    def __init__(self, training=False, dataset_path=None, save_dir=None, title=None):
        super(CoeusGenerative, self).__init__()

        # Title-based settings for the model
        self.title = title
        self.save_dir = os.path.join(save_dir, title) if title else save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # GPT-4-like model setup (using GPT-2 as a base)
        self.training = training
        self.dataset_path = dataset_path
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.to(self.device)

        # Training setup
        if training:
            self.optimizer = optim.Adam(self.parameters(), lr=5e-5)
            self.criterion = nn.CrossEntropyLoss()
        else:
            # Load settings for inference
            path_to_trained = self.get_setting("path_to_trained")
            self.load_state_dict(torch.load(path_to_trained))
            self.eval()

        # Referenced models for classification or detection
        self.referenced_models = {}
        referenced_models = self.get_setting("referenced_models") or {}
        for key, model_info in referenced_models.items():
            self.referenced_models[key] = self.load_other_model(model_info)

    ### SETTINGS MANAGEMENT ###
    def update_settings_file(self, key, value):
        file_path = os.path.join(self.save_dir, "coeus_generative_settings.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
        else:
            data = {}
        data[key] = value
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    def get_setting(self, key):
        file_path = os.path.join(self.save_dir, "coeus_generative_settings.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
            return data.get(key)
        return None

    ### DATASET CREATION ###
    class TextDataset(Dataset):
        def __init__(self, tokenizer, texts, max_length=512):
            self.tokenizer = tokenizer
            self.texts = texts
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encodings = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].squeeze()
            attention_mask = encodings["attention_mask"].squeeze()
            return input_ids, attention_mask

    def create_text_dataset(self):
        if self.training and self.dataset_path:
            with open(self.dataset_path, "r") as file:
                texts = json.load(file)
            return self.TextDataset(self.tokenizer, texts)

    ### TRAINING ###
    def train_in_progessive(self, epochs_per_run=3, batch_size=8):
        dataset = self.create_text_dataset()
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        checkpoint_path = os.path.join(self.save_dir, "progress_checkpoint.pth")
        start_epoch = 0

        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, start_epoch + epochs_per_run):
            self.train()
            total_loss = 0
            for input_ids, attention_mask in train_loader:
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                checkpoint_path,
            )

        self.save_trained()

    def save_trained(self):
        trained_path = os.path.join(self.save_dir, "trained_model.pth")
        self.update_settings_file("path_to_trained", trained_path)
        torch.save(self.state_dict(), trained_path)

    ### REFERENCED MODELS ###
    def load_other_model(self, model_info):
        model_path = model_info.get("path")
        model_type = model_info.get("type").lower()
        model_name = model_info.get("model_name").lower()
        selected_classes = model_info.get("selected_classes", None)
        detection_classes = model_info.get("detection_classes", None)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load classification models
        if model_type == "classification":
            if model_name == "resnet50":
                model = models.resnet50(weights=None)
                num_classes = len(selected_classes) if selected_classes else self.get_setting("num_classes")
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.selected_classes = selected_classes
            elif model_name == "mobilenetv2":
                model = mobilenet_v2(weights=None)
                num_classes = len(selected_classes) if selected_classes else self.get_setting("num_classes")
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.selected_classes = selected_classes
            else:
                raise ValueError(f"Unsupported classification model name: {model_name}")

        # Load identification models
        elif model_type == "identification":
            if model_name == "fasterrcnn":
                model = fasterrcnn_resnet50_fpn(weights=None)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()  # Set to evaluation mode
                model.detection_classes = detection_classes
            else:
                raise ValueError(f"Unsupported identification model name: {model_name}")

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Move the model to the appropriate device and return it
        return model.to(self.device)

    ### INFERENCE ###
    def generate_answer(self, question, max_length=100):
        self.model.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(question, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(
                input_ids, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id
            )
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer

    def answer_with_references(self, question, input_data=None):
        primary_answer = self.generate_answer(question)
        references = {}

        for key, model in self.referenced_models.items():
            if isinstance(model, models.ResNet):
                # Example classification inference
                input_tensor = input_data.to(self.device)  # Input image tensor
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                    predicted_class = torch.argmax(output, dim=1).item()
                references[key] = f"Predicted class: {predicted_class}"
            elif isinstance(model, fasterrcnn_resnet50_fpn):
                # Example object detection inference
                input_tensor = [input_data.to(self.device)]  # List of image tensors
                model.eval()
                with torch.no_grad():
                    detections = model(input_tensor)
                references[key] = detections  # Return raw detections or format results
            else:
                references[key] = f"Model type for {key} is not implemented."

        return {"primary_answer": primary_answer, "references": references}



# usage:
# coeus = CoeusGenerative(training=True, dataset_path="path/to/texts.json", save_dir="./Models", title="GenerativeModel")
# coeus.train_model(epochs_per_run=3)
# answer = coeus.generate_answer("How do I troubleshoot an engine fault?")
# print(answer)
