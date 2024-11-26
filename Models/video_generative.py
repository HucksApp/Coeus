import torch
import os
import json
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
from torchvision.transforms import functional as F
from PIL import Image
from Models.traits.base import CoeusBase
from Models.traits.model_keys import CoeusModelKeys
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import models



class CoeusVideoCreator(nn.Module, CoeusModelKeys):
    
    def __init__(self, title, training=False, dataset_path=None, save_dir=None, keys=[], checkpoint_path=None):
        super(CoeusVideoCreator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.title = title

        # Directory paths
        self.save_dir = os.path.join(save_dir, title)
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize the ModelKey class to handle word prompts
        self.dataset_path = dataset_path
        self.video_size = (128, 128)
        CoeusModelKeys.__init__(self, self.save_dir,
                                "coeus_generate_settings.json", training, keys)

        # Initialize the video generation model (now with 3D convolutions)
        self.model, self.discriminator = self.initialize_video_model()

        # If not training, load the saved model
        if not training and checkpoint_path:
            self.load_trained_model(checkpoint_path)

        # Load dataset and create dataset object for training or testing
        if dataset_path:
            self.dataset = self.create_video_dataset()

        # Optimizer and scheduler for training
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Load pre-trained VGG19 model for perceptual loss
        self.vgg = models.vgg19(pretrained=True).features.to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters
        
    def load_trained_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.model.to(self.device)
        self.discriminator.to(self.device)
        self.model.eval()
        print(f"Trained model loaded from {checkpoint_path}")


    def initialize_video_model(self):
        model = VideoGANGenerator3D().to(self.device)
        discriminator = VideoDiscriminator3D().to(self.device)  # Initialize discriminator
        return model, discriminator

    def create_video_dataset(self):
        if self.dataset_path:
            dataset = VideoDataset(self.dataset_path, transform=VideoTransform())
            return dataset
        else:
            raise ValueError("Dataset path not provided.")

    def train(self, epochs=10, batch_size=4):
        self.model.train()
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, (video_frames, labels) in enumerate(dataloader):
                video_frames = video_frames.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                generated_video = self.model(video_frames)
                loss = self.compute_loss(generated_video, video_frames)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

            # Save model checkpoint
            self.save_checkpoint(epoch)

    def compute_loss(self, generated_video, real_video):
        pixel_loss = nn.MSELoss()(generated_video, real_video)

        total_loss = pixel_loss

        # Include adversarial loss
        adv_loss = self.adversarial_loss(generated_video)
        total_loss += adv_loss

        # Include perceptual loss
        perceptual_loss = self.perceptual_loss(generated_video, real_video)
        total_loss += perceptual_loss

        # Include temporal consistency loss
        temporal_loss = self.temporal_consistency_loss(generated_video)
        total_loss += temporal_loss

        return total_loss

    def adversarial_loss(self, generated_video):
        batch_size, seq_len, c, h, w = generated_video.size()

        generated_video_flat = generated_video.view(-1, c, h, w)
        real_video_flat = generated_video.view(-1, c, h, w)

        real_preds = self.discriminator(real_video_flat)
        fake_preds = self.discriminator(generated_video_flat)

        real_loss = F.binary_cross_entropy(real_preds, torch.ones_like(real_preds).to(self.device))
        fake_loss = F.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds).to(self.device))

        return (real_loss + fake_loss) / 2

    def temporal_consistency_loss(self, generated_video):
        batch_size, seq_len, c, h, w = generated_video.size()

        temporal_loss = 0.0
        for t in range(1, seq_len):
            prev_frame = generated_video[:, t-1]
            curr_frame = generated_video[:, t]
            frame_diff = F.mse_loss(curr_frame, prev_frame)
            temporal_loss += frame_diff

        return temporal_loss / (seq_len - 1)

    def perceptual_loss(self, generated_video, real_video):
        real_features = self.extract_vgg_features(real_video)
        generated_features = self.extract_vgg_features(generated_video)

        perceptual_loss = F.mse_loss(generated_features, real_features)
        return perceptual_loss

    def extract_vgg_features(self, video):
        batch_size, seq_len, c, h, w = video.size()
        video = video.view(batch_size * seq_len, c, h, w)
        features = self.vgg(video)
        return features.view(batch_size, seq_len, -1)

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def generate_video_from_prompt(self, prompt, input_frames):
        if prompt in self.keys:
            print(f"Generating video for prompt '{prompt}' with labels {self.keys}")
        else:
            print(f"Prompt '{prompt}' not found in keys. Using default labels.")
        
        self.model.eval()
        with torch.no_grad():
            input_frames = input_frames.to(self.device)
            generated_video = self.model(input_frames)
        return generated_video

    def save_generated_video(self, video_frames, filename="generated_video.mp4"):
        self.convert_frames_to_video(video_frames, filename)

    def convert_frames_to_video(self, frames, filename):
        height, width = frames[0].shape[1], frames[0].shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))

        for frame in frames:
            frame = frame.permute(1, 2, 0).cpu().numpy()
            frame = np.uint8(frame * 255)
            out.write(frame)

        out.release()
        print(f"Generated video saved as {filename}")

    def evaluate(self, validation_dataset):
        self.model.eval()
        dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=False)
        total_loss = 0.0
        with torch.no_grad():
            for video_frames, _ in dataloader:
                video_frames = video_frames.to(self.device)
                generated_video = self.model(video_frames)
                loss = self.compute_loss(generated_video, video_frames)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Evaluation Loss: {avg_loss}")

class VideoGANGenerator3D(nn.Module):
    """
    Video generation model using 3D convolutions.
    """
    def __init__(self):
        super(VideoGANGenerator3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 16 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 3 * 128 * 128)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), 3, 128, 128)
        return x

class VideoDiscriminator3D(nn.Module):
    def __init__(self):
        super(VideoDiscriminator3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class VideoDataset(Dataset):
    def __init__(self, video_folder, transform=None):
        self.video_folder = video_folder
        self.transform = transform
        self.video_files = [f for f in os.listdir(video_folder)]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_folder, self.video_files[idx])
        video_frames = self.load_video(video_path)

        if self.transform:
            video_frames = self.transform(video_frames)

        return video_frames, self.video_files[idx]

    def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return torch.tensor(np.array(frames)).permute(0, 3, 1, 2).float() / 255  # Normalize


class VideoTransform:
    """
    Augmentation for video frames.
    """
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

    def __call__(self, frames):
        transformed_frames = [self.transform(frame) for frame in frames]
        return torch.stack(transformed_frames)
