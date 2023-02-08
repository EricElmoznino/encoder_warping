from torchvision import transforms
from PIL import Image
import glob

image_list = []
for filename in glob.glob('yourpath/*.gif'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)


def PreprocessGS(images):
    size = 96
    transform = transforms.Compose([
         transforms.Resize((size,size)),
         transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(mean=0.5, std=0.5)])
    
    try:
        return np.stack([transform(Image.open(i).convert('RGB')) for i in images])
    except TypeError: 
        return torch.stack([transform(Image.open(i).convert('RGB')) for i in images])