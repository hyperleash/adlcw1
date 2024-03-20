from functions import train_model, test_model, test_all_checkpoints
import torch
import torchvision
import torchvision.transforms as transforms
from mixup import MixupDataset
from torch.utils.data import random_split 



if __name__ == '__main__':

    epochs = 20
    transform = transforms.Compose(
        [
        #transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    batch_size = 20

    #TRAIN SET
    trainset = torchvision.datasets.CIFAR10(root='../task2/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    dataiter = iter(trainset)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #TEST SET (REGULAR AND MIXED UP WITH BOTH SAMPLING METHODS)
    testset = torchvision.datasets.CIFAR10(root='../task2/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    mixup_testset_s1 = MixupDataset(testset, alpha=0.4, sampling_method=1, demo=False)  
    mixup_testset_s2 = MixupDataset(testset, alpha=0.4, sampling_method=2, demo=False)  

    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    
    #TRAINING SAMPLING METHOD 1 (Look at vit_s1.txt for logs)
    mixup_trainset_s1 = MixupDataset(trainset, alpha=0.4, sampling_method=1, demo=False)  
    train_s1, val_s1 = random_split(mixup_trainset_s1, [train_size, val_size])
    train_model(train_s1, "vit_s1", 20, batch_size)

    #TRAINING SAMPLING METHOD 2 (Look at vit_s1.txt for logs)
    mixup_trainset_s2 = MixupDataset(trainset, alpha=0.4, sampling_method=2, demo=False)  
    train_s2, val_s2 = random_split(mixup_trainset_s2, [train_size, val_size])
    train_model(train_s2, "vit_s2", 20, batch_size)

    #RESULTS (Look at corresponding .json files)
    test_all_checkpoints("vit_s1", epochs, "train", train_s1)
    test_all_checkpoints("vit_s1", epochs, "val", val_s1)
    test_all_checkpoints("vit_s1", epochs, "test", testset)
    test_all_checkpoints("vit_s1", epochs, "test_mixup_s1", mixup_testset_s1)

    test_all_checkpoints("vit_s2", epochs, "train", train_s2)
    test_all_checkpoints("vit_s2", epochs, "val", val_s2)
    test_all_checkpoints("vit_s2", epochs, "test", testset)
    test_all_checkpoints("vit_s2", epochs, "test_mixup_s2", mixup_testset_s2)

    ###########################---COMPARISON---###########################

    #For both methods the validation shows slightly higher loss and smaller accuracy than the training set. 
    #This is expected as the model is trained on the training set and the validation set is used to check the performance of the model.

    #Strangely, testing performance is noticably better than the development performance. 
    #This is likely due to the fact that testing images aren't mixed up and the model is more confident in its predictions.
    #This is further supported by the fact that the mixup test set has a lower accuracy than the regular test set and the mixup development set.
