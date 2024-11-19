import os
import json
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPTokenizer
from PIL import Image
from torch.utils.data import DataLoader

class CoeusImageGenerative:
    def __init__(self, model_name="stable-diffusion", save_dir=None, title=None, training=False, scheduler_name="DDIM"):
        # Initialize model settings
        self.title = title
        self.save_dir = os.path.join(save_dir, title) if title else save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model loading
        self.model_name = model_name
        self.pipeline = None
        self.controlnet = None
        self.training = training
        self.scheduler_name = scheduler_name
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        if training:
            # Load the model in training mode
            self.load_pipeline(training=True)
            self.optimizer = torch.optim.AdamW(self.pipeline.unet.parameters(), lr=5e-6)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        else:
            # Load pretrained model for inference
            self.load_pipeline()

    ### SETTINGS MANAGEMENT ###
    def _settings_file_path(self):
        return os.path.join(self.save_dir, "coeus_generate_settings.json")

    def update_settings_file(self, key, value):
        file_path = self._settings_file_path()
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
        else:
            data = {}
        data[key] = value
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    def get_setting(self, key):
        file_path = self._settings_file_path()
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
            return data.get(key)
        return None

    ### MODEL LOADING ###
    def load_pipeline(self, training=False):
        try:
            if self.model_name == "stable-diffusion":
                model_id = "CompVis/stable-diffusion-v1-4" if training else "runwayml/stable-diffusion-v1-5"
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16
                ).to(self.device)
            elif self.model_name == "stable-diffusion-controlnet":
                self.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
                ).to(self.device)
                self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
                ).to(self.device)
                self.pipeline.scheduler = self.scheduler_name
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Error loading the pipeline: {e}")

    ### TRAINING ###
    def train_progressive(self, dataset, epochs=1, save_checkpoints=True):
        """
        Fine-tune the pipeline on a custom dataset (text-image pairs).
        Dataset should be a PyTorch DataLoader.
        """
        self.pipeline.train()
        data_loader = DataLoader(dataset, batch_size=8, shuffle=True)  # Adjust batch size as needed

        for epoch in range(epochs):
            for batch_idx, (texts, images) in enumerate(data_loader):
                try:
                    images = images.to(self.device)
                    latents = self.pipeline.vae.encode(images).latent_dist.sample()
                    text_inputs = self.tokenizer(
                        texts, padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.device)
                    noise = torch.randn_like(latents)
                    latents_noisy = latents + noise
                    predicted_noise = self.pipeline.unet(latents_noisy, text_inputs).sample
                    loss = torch.nn.functional.mse_loss(predicted_noise, noise)

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                except Exception as e:
                    print(f"Error during training step {batch_idx}: {e}")
                    continue

            self.lr_scheduler.step()

            if save_checkpoints:
                self.save_checkpoint(epoch)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            "model_state_dict": self.pipeline.unet.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "epoch": epoch,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

    ### INFERENCE ###
    def generate_image(self, prompt, negative_prompt=None, guidance_scale=7.5, num_inference_steps=50):
        self.pipeline.eval()
        with torch.no_grad():
            images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images
        return images

    def generate_with_controlnet(self, prompt, canny_image, guidance_scale=7.5, num_inference_steps=50):
        if not self.controlnet:
            raise ValueError("ControlNet is not loaded. Use a ControlNet model.")
        self.pipeline.eval()
        with torch.no_grad():
            images = self.pipeline(
                prompt=prompt,
                image=canny_image,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images
        return images

    ### SETTINGS EXPORT ###
    def save_generated_image(self, image, file_name):
        save_path = os.path.join(self.save_dir, file_name)
        image.save(save_path)
        print(f"Image saved to {save_path}")

    def save_torchscript_model(self):
        scripted_path = os.path.join(self.save_dir, "trained_model_scripted.pth")
        scripted_model = torch.jit.script(self.pipeline.unet)
        scripted_model.save(scripted_path)
        self.update_settings_file("path_to_scripted", scripted_path)

    def load_torchscript_model(self):
        scripted_path = self.get_setting("path_to_scripted")
        if scripted_path and os.path.exists(scripted_path):
            self.pipeline.unet = torch.jit.load(scripted_path)
            self.pipeline.unet.to(self.device)
        else:
            print(f"Scripted model not found at {scripted_path}. Make sure the path is correct.")
