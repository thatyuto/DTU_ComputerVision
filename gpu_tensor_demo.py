import torch
from PIL import Image
from torchvision import transforms

# -- GPU check 
print(f"Is CUDA available? {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("CUDA is not available. Exiting...")
    exit()

# -- Device setup
device = torch.device("cuda")
print(f"Using device: {device}")

# -- read image
img = Image.open("profilePicture.jpg").convert("RGB") # converts image to RGB mode
x = transforms.ToTensor()(img) # converts image to tensor
print("after ToTensor:", x.shape, x.dtype, x.device)

# -- move to GPU
x_gpu = x.to(device) # move tensor to GPU
print("after to(device):", x_gpu.shape, x_gpu.dtype, x_gpu.device)

# -- save to disk
# saving a CUDA tensor records GPU storage; other machines may fail to load it.
# official recipe: save a cpu copy for protability
torch.save(x_gpu.cpu(), "x_gpu.pt")
print("saved to x_gpu.pt")

