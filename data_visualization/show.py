import torch
import matplotlib.pyplot as plt
import numpy as np

# Function to display images
def imshow(img):
    # Unnormalize the image
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img = torch.clamp(img, 0, 1)  # Ensure values are within [0, 1] for display
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')  # Remove axis for cleaner display
    plt.show()