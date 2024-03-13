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
from mixup import MixupDataset


class Vit(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vision_transformer.VisionTransformer(
            image_size=32, #CIFAR-10 has 32x32 images
            patch_size=4, #16
            num_layers=6, #12
            num_heads=4, #12
            hidden_dim=16, #768
            mlp_dim=64, # 3072
            num_classes=10, 
        )

    def forward(self, x):
        x = self.vit(x)
        return x

def train_model(sampling_method: int, name: str, epochs: int, batches: int):

    loss_filename = name+'.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"  Device {i}: {device_name}")
    else:
        print("CUDA is not available. Using CPU.")


    vit_model_mixup = Vit() #vit_b_16(weights = "DEFAULT")
    transform = transforms.Compose(
            [ 
            #transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    batch_size = batches # as mentioned by the authors of the paper
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    mixup_trainset = MixupDataset(trainset, alpha=0.2, sampling_method=sampling_method)  
    mixup_loader = torch.utils.data.DataLoader(mixup_trainset, batch_size=batch_size, shuffle=True)
    
    vit = vit_model_mixup.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(vit.parameters(), lr=0.001, momentum=0.9)

    ## train
    print("Starting training")
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch: {epoch}")    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
                
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device) 
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            outputs = vit(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                with open(loss_filename, "a") as f:  # Open file in append mode
                    f.write(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.3f}\n")
                running_loss = 0.0

    print('Training done.')

        # save trained model
    torch.save(vit.state_dict(), name+'.pt')
    print('Model saved.')

def test_model(saved_model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"  Device {i}: {device_name}")
    else:
        print("CUDA is not available. Using CPU.")

    transform = transforms.Compose(
        [
        #transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    batch_size = 4

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    dataiter = iter(testloader)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    vit = Vit() #vit_b_16(weights = "DEFAULT")

    vit.load_state_dict(torch.load(saved_model_path))
    vit = vit.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.to(device) 
            labels = labels.to(device)

            outputs = vit(inputs)

            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (100 * correct / total)
    print(f"Accuracy: {accuracy:.2f}%")