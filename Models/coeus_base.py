import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import mobilenet_v2
from diffusers import StableDiffusionPipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import mobilenet_v2
from torchvision.transforms.functional import to_tensor
from PIL import Image
from abc import abstractmethod

class CoeusBase:

    def __init__(self) :
        self.reference_models = {}

    @abstractmethod
    def predict_image(self, image_path, selected_classes=None, **kwargs):
        """implement in child class"""
        pass

    @abstractmethod
    def get_setting(self, key):
        """implement in child class"""
        pass

    @abstractmethod
    def update_settings_file(self, key, value):
        """implement in child class"""
        pass


    def create_reference_models(self, referenced_models):
        for key, model_props in referenced_models.items():
            self.reference_models[key] = self.__load_other_model__(
                model_props['path'], model_props['model_type'] or "resnet50", model_props['selected_classes'], model_props['detection_classes'])


    def __load_other_model__(self, model_options, model_type, selected_classes=None, detection_classes=None):
        if model_type.lower() == "resnet50":
            # Classification model
            other_model = models.resnet50(weights=None)
            # If selected_classes are provided, set the number of classes to the length of selected_classes
            num_classes = len(
                selected_classes) if selected_classes else self.get_setting("num_classes")
            other_model.fc = nn.Linear(other_model.fc.in_features, num_classes)
            other_model.load_state_dict(torch.load(
                model_options, map_location=self.device))

            # Store the selected classes for later filtering during inference
            other_model.selected_classes = selected_classes

            return other_model.to(self.device)

        elif model_type.lower() == "fasterrcnn":
            # Object detection model (e.g., Faster R-CNN)
            other_model = fasterrcnn_resnet50_fpn(weights=None)
            other_model.load_state_dict(torch.load(
                model_options, map_location=self.device))
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
                model_options, map_location=self.device))
            # Store the selected classes for later filtering during inference
            other_model.selected_classes = selected_classes
            return other_model.to(self.device)

        elif model_type.lower() == "gpt-4-like":
            # Load GPT-4-like model and tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(model_options)
            other_model = GPT2LMHeadModel.from_pretrained(
                model_options).to(self.device)
            other_model.tokenizer = tokenizer  # Attach tokenizer for convenience
            other_model.eval()  # Set the model to eval mode by default

            return other_model

        elif model_type.lower() == "stable-diffusion":
            # Stable Diffusion image generation model
            other_model = StableDiffusionPipeline.from_pretrained(
                model_options, torch_dtype=torch.float16).to(self.device)
            other_model.eval()  # Stable Diffusion is often used in eval mode
            return other_model

        else:
            raise NotImplementedError(
                f"Model type {model_type} is not supported.")
        
    def manage_referenced_model(self, key, model_options=None, model_type=None, rm=False, detection_classes=None, selected_classes=None):
        # Load current referenced models from settings
        referenced_models = self.get_setting("referenced_models") or {}

        if rm:
            # Remove the specified model if it exists
            if key in self.reference_models:
                del self.reference_models[key]  # Remove from in-memory models
            if key in referenced_models:
                del referenced_models[key]  # Remove from saved settings
                self.update_settings_file("referenced_models", referenced_models)
                print(f"Removed referenced model: {key}")
            else:
                print(f"Referenced model '{key}' does not exist.")
            return

        # Add or update a referenced model
        self.reference_models[key] = self.load_other_model(
            model_options, model_type, selected_classes, detection_classes
        )

        # Update settings to include the new model reference
        referenced_models[key] = {
            "path": model_options,
            "model_type": model_type,
            "detection_classes": detection_classes,
            "selected_classes": selected_classes,
        }
        self.update_settings_file("referenced_models", referenced_models)
        print(f"Added or updated referenced model: {key}")


    def predict_image_with_references(self, image_path, selected_classes=None, **kwargs):
        primary_prediction = self.predict_image(image_path, selected_classes, **kwargs)
        references = {}
        image = Image.open(image_path).convert('RGB')
        transformed_image = self.transform(image).unsqueeze(0).to(self.device)

        for key, model in self.reference_models.items():
            model_info = self.get_setting("referenced_models")[key]
            model_type = model_info.get("model_type", "resnet50").lower()

            if model_type == "resnet50":
                predicted_class = self.predict_with_selected_classes(
                    model, transformed_image, selected_classes
                )
                if predicted_class is not None:
                    idx_to_class = {v: k for k, v in self.get_setting(
                        "class_to_idx").items()}
                    references[key] = idx_to_class[predicted_class]

            elif model_type == "fasterrcnn":
                detection_classes = model_info.get("detection_classes", {})
                detection_image = to_tensor(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    detections = model(detection_image)[0]

                predictions = []
                for box, label, score in zip(detections["boxes"], detections["labels"], detections["scores"]):
                    if score > 0.5:  # Confidence threshold
                        if label.item() in selected_classes:  # Include only selected classes
                            predictions.append({
                                "box": box.cpu().tolist(),
                                "label": detection_classes.get(label.item(), f"Unknown ({label.item()})"),
                                "score": score.item()
                            })
                references[key] = predictions

            elif model_type == "gpt-4-like":
                with torch.no_grad():
                    prompt = f"Analyze the given image: {image_path} and provide insights."
                    inputs = model.tokenizer(
                        prompt, return_tensors="pt").to(self.device)
                    output = model.generate(
                        **inputs, max_length=150, num_beams=5, early_stopping=True)
                    references[key] = model.tokenizer.decode(
                        output[0], skip_special_tokens=True)

            elif model_type == "stable-diffusion":
                with torch.no_grad():
                    # Generating an image or variation based on the input image
                    sd_prompt = f"Create a detailed artistic rendition of {image_path}."
                    generated_images = model(
                        prompt=sd_prompt, num_inference_steps=50, guidance_scale=7.5).images
                    # Assuming you need the first image
                    references[key] = generated_images[0]

            else:
                references[key] = f"Unsupported model type: {model_type}"

        return {"primary": primary_prediction, "references": references}
