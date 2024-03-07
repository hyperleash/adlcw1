import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def polynomial_fun(w: np.ndarray,x: np.ndarray) -> np.ndarray:
    """Computes a polynomial function output
    Args:
        w (np.ndarray): weight vector of size (M+1)
        x (np.ndarray): vector of scalar x input values

    Returns:
        numpy.ndarray: outputs of the polynomial function
    """
    powers = np.arange(len(w)).reshape(1, -1)

    y = np.sum(w * (x[:, np.newaxis] ** powers), axis=1)

    return y

def fit_polynomial_ls(x: np.ndarray, t: np.ndarray, M: int) -> np.ndarray:
    """Computes optimal least-squares solution for a linear model
    Args:
        x (np.ndarray): vector of inputs x
        t (np.ndarray): vector of 
        M (int): degree of polynomial

    Returns:
        np.ndarray: optimal weights vector
    """
    x_tensor = torch.from_numpy(x)
    
    t_tensor = torch.from_numpy(t)
    
    powers = torch.arange(M + 1, dtype=x_tensor.dtype, device=x_tensor.device).reshape(1, -1) 
    
    design_matrix = x_tensor.unsqueeze(1) ** powers

    w = torch.linalg.lstsq(design_matrix, t_tensor).solution

    return w

def fit_polynomial_sgd(x: np.ndarray, t: np.ndarray, M: int, lr: float, batch_size: int) -> np.ndarray:
    """Computes optimal weights for a linear model using SGD

    Args:
        x (np.ndarray): input vector
        t (np.ndarray): target output vector
        M (int): degree of polynomial
        lr (float): learning rate
        batch_size (int): batch size

    Returns:
        np.ndarray: optimal weights vecto
    """
    #Convert to tensors
    x_tensor = torch.from_numpy(x).to(torch.float32)
    t_tensor = torch.from_numpy(t).to(torch.float32)

    #Create x features
    powers = torch.arange(1,M + 1, dtype=x_tensor.dtype, device=x_tensor.device).reshape(1, -1) 
    design_matrix = x_tensor.unsqueeze(1) ** powers

    #Could use polynomial_fun here, but since there is a PyTorch version of it, I will use that
    torch.manual_seed(42)
    model = torch.nn.Linear(M, 1)

    #Loss and optimizer
    mse_loss = torch.nn.MSELoss()  
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 

    #Create dataset and loader
    dataset = torch.utils.data.TensorDataset(design_matrix, t_tensor) 
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    ##Train
    for epoch in range(1000):  
        for b_num, (x_batch, t_batch) in enumerate(loader):
            optimizer.zero_grad()
            
            pred = model(x_batch)

            loss = mse_loss(pred.flatten(), t_batch) 
            
            loss.backward()
            optimizer.step()
            
        if(epoch % 100 == 0):
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    #concatinate bias and weights to get the full weight vector
    w_tensor = torch.cat((model.bias.detach().unsqueeze(0), model.weight.detach()), dim=1)
    w_numpy = w_tensor.numpy().flatten()

    return w_numpy

def fit_polynomial_sgd_lasso(x: np.ndarray, t: np.ndarray, M: int, lr: float, batch_size: int, lmbda: float) -> np.ndarray:
    """Computes optimal weights for a linear model using SGD

    Args:
        x (np.ndarray): input vector
        t (np.ndarray): target output vector
        M (int): degree of polynomial
        lr (float): learning rate
        batch_size (int): batch size

    Returns:
        np.ndarray: optimal weights vecto
    """
    #Convert to tensors
    x_tensor = torch.from_numpy(x).to(torch.float32)
    t_tensor = torch.from_numpy(t).to(torch.float32)

    #Create x features
    powers = torch.arange(1,M + 1, dtype=x_tensor.dtype, device=x_tensor.device).reshape(1, -1) 
    design_matrix = x_tensor.unsqueeze(1) ** powers

    #Could use polynomial_fun here, but since there is a PyTorch version of it, I will use that
    torch.manual_seed(42)
    model = torch.nn.Linear(M, 1)

    #Loss and optimizer
    mse_loss = torch.nn.MSELoss()  
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 

    #Create dataset and loader
    dataset = torch.utils.data.TensorDataset(design_matrix, t_tensor) 
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    ##Train
    for epoch in range(10000):  
        for b_num, (x_batch, t_batch) in enumerate(loader):
            optimizer.zero_grad()
            
            pred = model(x_batch)

            loss = mse_loss(pred.flatten(), t_batch) 
            reg_loss = lmbda * torch.sum(torch.abs(model.weight))

            total_loss = loss + reg_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        if(epoch % 100 == 0):
            print(f"Epoch: {epoch}, Total Loss: {total_loss.item()}, Loss: {loss.item()}, Reg Loss: {reg_loss.item()}")

    #concatinate bias and weights to get the full weight vector
    w_tensor = torch.cat((model.bias.detach().unsqueeze(0), model.weight.detach()), dim=1)
    w_numpy = w_tensor.numpy().flatten()

    return w_numpy

def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    """Computes root-mean-squared-error (RMSE) between true and predicted values

    Args:
        true (np.ndarray): ground truth vector
        pred (np.ndarray): predicted values vector

    Returns:
        float: RMSE value
    """
    sqrd_err = np.square(true - pred)
    mse = np.mean(sqrd_err)
    return np.sqrt(mse)



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x