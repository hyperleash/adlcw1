import torch
from torch.utils.data import Dataset 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from torchvision.models import vit_b_16
from torchvision.models import vision_transformer
import numpy as np



class MixupDataset(Dataset):
    def __init__(self, dataset, alpha=0.1, max=1.0, min=0.0, sampling_method = 1):
        self.dataset = dataset  
        self.alpha = alpha 
        self.max = max
        self.min = min
        self.sampling_method = sampling_method

    def __getitem__(self, index):
        x, y = self.dataset[index]

        if(self.sampling_method == 1):
            lmbda = np.random.beta(self.alpha, self.alpha) # Sampling like in the paper
        elif(self.sampling_method == 2):
            lmbda = np.random.uniform(self.min, self.max)
        
        index2 = torch.randperm(len(self.dataset))[0]  # Sample random second index
        x2, y2 = self.dataset[index2]
        lmbda = 0.5
        x = lmbda * x + (1 - lmbda) * x2
        y = lmbda * y + (1 - lmbda) * y2

        return x, y

    def __len__(self):
        return len(self.dataset) 
    
if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16# as mentioned by the authors of the paper
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Apply Mixup
    mixup_trainset = MixupDataset(trainset, alpha=0.4, sampling_method=1)  
    mixup_loader = torch.utils.data.DataLoader(mixup_trainset, batch_size=batch_size, shuffle=True)

    # Get a Mixup batch
    mixup_dataiter = iter(mixup_loader)
    mixup_images, mixup_labels = next(mixup_dataiter)
    # example images
    #dataiter = iter(trainloader)
    #images, labels = next(dataiter) # note: for pytorch versions (<1.14) use dataiter.next()

    im = Image.fromarray((torch.cat(mixup_images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("mixup_pt_images.jpg")
    print('mixum_pt_images.jpg saved.')
    print('Ground truth labels:' + ' '.join('%5s' % classes[int(mixup_labels[j])] for j in range(batch_size)))