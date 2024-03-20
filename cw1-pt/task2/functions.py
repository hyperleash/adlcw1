import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import vision_transformer
from mixup import MixupDataset
import json

#import matplotlib.pyplot as plt


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

def train_model(sampling_method: int, name: str, epochs: int, batches: int):
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
    transform = transforms.Compose(
            [ 
            #transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    batch_size = batches
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    mixup_trainset = MixupDataset(trainset, alpha=0.2, sampling_method=sampling_method, num_classes=10)  
    mixup_loader = torch.utils.data.DataLoader(mixup_trainset, batch_size=batch_size, shuffle=True)
    
    vit = vit_model_mixup.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(vit.parameters(), lr=0.001, momentum=0.9)

    ## train
    print("Starting training")
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

        if (epoch + 1) % 2 == 0:  # Save checkpoint every 2 epochs
            torch.save(vit.state_dict(),f"{name}_{epoch + 1}.pt")
            print(f'Checkpoint at {epoch + 1} epochs saved.')

    print('Training done.')

def test_model(saved_model_path: str):
    """Method to test a specific saved model

    Args:
        saved_model_path (str): path to model file

    Returns:
       int : accuracy on test set
    """
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

    vit = Vit() 

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
    return accuracy

def test_all_checkpoints(name: str, n_epochs: int):
    """Method to test all checkpoints of a given model and save results to a file

    Args:
        name (str): name of the model
        n_epochs (int): number of epochs of the final checkpoint
    """
    results = {} 

    for epoch in range(0, n_epochs + 1, 2):
        if(epoch == 0):
            continue

        filename = f"{name}_{epoch}.pt"
        
        print(f"Testing checkpoint: {filename}")
        test_results = test_model(filename) 
        results[epoch] = test_results
        
    with open(f"{name}_results.json", "w") as outfile:
        json.dump(results, outfile, indent=4)  

def save_plot(results_file : str):
    """Method to save a .png of a plot of the results

    Args:
        results_file (str): name of the results file
    """
    with open(f"{results_file}_results.json", 'r') as f:
        data = json.load(f)

    epochs = [int(epoch) for epoch in data.keys()]
    accuracy = list(data.values())

    plt.figure(figsize=(8, 6))  
    plt.plot(epochs, accuracy, marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{results_file} Accuracy vs. Epochs')
    plt.grid(True)

    plt.savefig(f'{results_file}_plot.png')

    plt.close()

    print(f"Plot saved as {results_file}_plot.png")

def save_example(model_name: str, epochs: int):
    """Method to save an example of the model's predictions

    Args:
        model_name (str): name of the model
        epochs (int): number of epochs of the checkpoint being tested
    """
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
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 36

    testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    dataiter = iter(testloader)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
   
    images, labels = next(dataiter)
    vit = Vit() 

    vit.load_state_dict(torch.load(f"{model_name}_{epochs}.pt"))
    vit = vit.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        
        inputs = images.to(device)
        targets = labels.to(device)
            

        outputs = vit(inputs)

        _, predicted = torch.max(outputs, 1)
          
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = (100 * correct / total)
    print(f"Accuracy on 36 example images: {accuracy}%")

    #Save image
    im_tensor = (torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8')
    num_cols = 6  
    label_font_size = 6
    label_padding = 5

    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', label_font_size)

    num_images = 36
    img_width, img_height = 32, 32
    num_rows = (num_images + num_cols - 1) // num_cols 

    grid_width = num_cols * img_width
    grid_height = num_rows * (img_height + label_padding + label_font_size + 15)

    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid_img)

    for i in range(num_images):
        img_array = im_tensor[0:img_height, i*32:i*32 + img_width, :]
        img = Image.fromarray(img_array)

        x = (i % num_cols) * img_width
        y = (i // num_rows) * (img_height + label_padding + 3*label_font_size)
        
        grid_img.paste(img, (x, y))

        draw.text((x, y + img_height + label_padding), f"T: {classes[labels[i]]}", font=font, fill='black')
        draw.text((x, y + img_height + label_padding + label_font_size), f"P: {classes[predicted[i]]}", font=font, fill='black')

    grid_img.save(f"{model_name}_results.png")
    print(f'{model_name}_results.png saved.')

    print('Ground truth labels:' + ' '.join('%5s' % classes[int(labels[j])] for j in range(batch_size)))
    print('Predicted labels:' + ' '.join('%5s' % classes[int(predicted[j])] for j in range(batch_size)))