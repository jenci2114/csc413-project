import lpips
import torch
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

# Load the image using PIL
def load_image(path):
    
    image = Image.open(image_path).convert("RGB")

    # Define the transformation pipeline
    transforms = Compose([
        ToTensor(),  # Converts the PIL image to a PyTorch tensor in the range [0, 1]
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizes the tensor to the range [-1, 1]
    ])

    # Apply the transformations to the image
    normalized_image = transforms(image)

generated_path = "../textual_inversion/images/clock_small/4.jpeg"
generated_path = "../textual_inversion/images/clock_small/4.jpeg"

breakpoint()

img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,64,64)
d = loss_fn_alex(img0, img1)
print(d)