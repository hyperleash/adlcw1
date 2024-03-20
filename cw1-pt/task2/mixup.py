import torch
from torch.utils.data import Dataset 
import torchvision
import torchvision.transforms as transforms

from PIL import Image

import numpy as np

def one_hot_encode(y: int, num_classes: int):
    """Method to convert a label to one-hot encoding

    Args:
        y (int): label
        num_classes (int): numnber of classes

    Returns:
        torch.tensor: One-hot encoded label
    """
    one_hot_encoded = torch.zeros(1, num_classes)
    return one_hot_encoded.scatter_(1, torch.tensor([y]).unsqueeze(1), 1)

class MixupDataset(Dataset):
    def __init__(self, dataset, alpha=0.1, max=1.0, min=0.0, sampling_method = 1, num_classes=10, demo = False):
        self.dataset = dataset  
        self.alpha = alpha 
        self.max = max
        self.min = min
        self.sampling_method = sampling_method
        self.num_classes = num_classes
        self.demo = demo

    def __getitem__(self, index):

        x, y = self.dataset[index]
        y = one_hot_encode(y, self.num_classes)
        if(self.sampling_method == 1):
            lmbda = np.random.beta(self.alpha, self.alpha) # Sampling like in the paper
        elif(self.sampling_method == 2):
            lmbda = np.random.uniform(self.min, self.max) # Uniform sampling
        
        index2 = torch.randperm(len(self.dataset))[0] 
        x2, y2 = self.dataset[index2]
        y2 = one_hot_encode(y2, self.num_classes)

        #For demonstation purposes only (generating example images for better visibility)
        if(self.demo):
            lmbda = 0.5

        x = lmbda * x + (1 - lmbda) * x2
        y = lmbda * y.squeeze() + (1 - lmbda) * y2.squeeze()

        return x, y

    def __len__(self):
        return len(self.dataset) 
    
def mixup_demo():
    """SAVES A DEMO EXAMPLE OF MIXUP IMAGES
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Apply Mixup
    mixup_trainset = MixupDataset(trainset, alpha=0.4, sampling_method=1, demo=True)  
    mixup_loader = torch.utils.data.DataLoader(mixup_trainset, batch_size=batch_size, shuffle=True)

    # Get a Mixup batch
    mixup_dataiter = iter(mixup_loader)
    mixup_images, mixup_labels = next(mixup_dataiter)

    im = Image.fromarray((torch.cat(mixup_images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("mixup_pt_images.png")
    print('mixum_pt_images.png saved.')
  