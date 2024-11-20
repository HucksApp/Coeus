import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from PIL import Image
import os
import json
from Models.coeus_base import CoeusBase


class CoeusGenerative(nn.Module, CoeusBase):
    def __init__(self, training=False, dataset_path=None, save_dir=None, title=None):
        super(CoeusGenerative, self).__init__()
        CoeusBase.__init__(self)
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

        # Referenced models
        # Load referenced models
        referenced_models = self.get_setting("referenced_models") or {}
        self.create_reference_models(referenced_models)

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

    def create_text_dataset(self):
        if self.training and self.dataset_path:
            with open(self.dataset_path, "r") as file:
                texts = json.load(file)
            return TextDataset(self.tokenizer, texts)

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

# usage:
# coeus = CoeusGenerative(training=True, dataset_path="path/to/texts.json", save_dir="./Models", title="GenerativeModel")
# coeus.train_model(epochs_per_run=3)
# answer = coeus.generate_answer("How do I troubleshoot an engine fault?")
# print(answer)
