import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torchvision.models import vision_transformer
from mixup import MixupDataset

import time
import json

class Vit(nn.Module):
    """Class implementing Vision Transformer model
    """
    def __init__(self):
        super().__init__()
        self.vit = vision_transformer.VisionTransformer(
            image_size=32, #CIFAR-10 has 32x32 images
            patch_size=4, #16
            num_layers=2, #12
            num_heads=2, #12
            hidden_dim=16, #768
            mlp_dim=64, # 3072
            num_classes=10, 
        )

    def forward(self, x):
        x = self.vit(x)
        return x

def get_train_val(sampling_method: int, val_size: float = 0.1):
    """Method to get train and validation datasets

    Args:
        sampling_method (int): for MixUp, 1 or 2 as described in the task
        val_size (float, optional): Proportion of the validation set. Defaults to 0.1.

    Returns:
        tuple: train and validation datasets
    """
    transform = transforms.Compose(
            [ 
            #transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    batch_size = 20 # as mentioned by the authors of the paper
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    dataset_size = len(dataset)  
    val_size = int(val_size * dataset_size)
    train_size = dataset_size - val_size

    mixup_dataset = MixupDataset(dataset, alpha=0.2, sampling_method=sampling_method, num_classes=10)
    train_dataset, val_dataset = torch.utils.data.random_split(mixup_dataset, [train_size, val_size])
    
    return train_dataset, val_dataset



def test_model(saved_model_path: str, test):
    """Method for testing a saved model

    Args:
        saved_model_path (str): _description_
        test (_type_): _description_
        batches (int): _description_

    Returns:
        _type_: _description_
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device_name}")
    else:
        print("CUDA is not available. Using CPU.")

    vit = Vit() 

    vit.load_state_dict(torch.load(saved_model_path))
    vit = vit.to(device)

    correct = 0
    total = len(test)

    testloader = torch.utils.data.DataLoader(test, batch_size=20, shuffle=False, num_workers=2)

    criterion1 = torch.nn.BCEWithLogitsLoss()
    criterion2 = torch.nn.CrossEntropyLoss()

    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.to(device) 
            labels = labels.to(device)

            outputs = vit(inputs)
            _, predicted = torch.max(outputs, 1)
            
            if(labels.ndim > 1):
                loss = criterion1(outputs, labels)
            else:
                loss = criterion2(outputs, labels)
                
            
            if(labels.ndim > 1):
                _, labels = torch.max(labels, 1)
            
            running_loss += loss.item()
            correct += (predicted == labels).sum().item()

    accuracy = (100 * correct / total)
    total_loss = running_loss / total
    return accuracy, total_loss

def test_all_checkpoints(name: str, n_epochs: int, phase: str, test):
    """Method to test all checkpoints of a given model and save results to a file

    Args:
        name (str): name of the model
        n_epochs (int): number of epochs of the final checkpoint
    """
    results = {} 

    for epoch in range(0, n_epochs + 1, 4):
        if(epoch == 0):
            continue

        filename = f"{name}_{epoch}.pt"
        test_results = {}
        print(f"Testing checkpoint: {filename}")
        test_results["acc"], test_results["loss"] = test_model(filename, test) 
        results[epoch] = test_results
        
    with open(f"{name}_{phase}_results.json", "w") as outfile:
        json.dump(results, outfile, indent=4) 
    
    print(f"RESULTS SUMMARY IN {name}_{phase}_results.json")

def train_model(train: MixupDataset, name: str, epochs: int, batches: int):
    """Method to train and save a model

    Args:
        sampling_method (int): for MixUp, 1 or 2 as described in the task
        name (str): name of the model
        epochs (int): number of epochs
        batches (int): number of datapoints per batch
    """

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


    vit_model_mixup = Vit() 
    
    mixup_loader = torch.utils.data.DataLoader(train, batch_size=batches, shuffle=True)
    
    vit = vit_model_mixup.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(vit.parameters(), lr=0.001, momentum=0.9)

    ## train
    print("Starting training")
    start_time = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch: {epoch}")    
        running_loss = 0.0
        for i, data in enumerate(mixup_loader, 0):
                
            inputs, labels = data
            inputs = inputs.to(device) 
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = vit(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                with open(loss_filename, "a") as f: 
                    f.write(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 2000:.3f}\n")
                running_loss = 0.0

        if (epoch + 1) % 4 == 0:  # Save checkpoint every 2 epochs
            torch.save(vit.state_dict(),f"{name}_{epoch + 1}.pt")
            print(f'Checkpoint at {epoch + 1} epochs saved.')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Training done in {total_time:.2f} seconds.')
               

