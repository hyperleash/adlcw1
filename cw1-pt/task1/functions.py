import torch
import numpy as np


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
    x_train, x_val, t_train, t_val = train_test_split(x,t,42)
    #Convert to tensors
    x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
    t_train_tensor = torch.from_numpy(t_train).to(torch.float32)

    x_val_tensor = torch.from_numpy(x_val).to(torch.float32)
    t_val_tensor = torch.from_numpy(t_val).to(torch.float32)

    #Create x features
    powers = torch.arange(1,M + 1, dtype=x_train_tensor.dtype, device=x_train_tensor.device).reshape(1, -1) 
    design_matrix = x_train_tensor.unsqueeze(1) ** powers

    design_matrix_val = x_val_tensor.unsqueeze(1) ** powers

    #Could use polynomial_fun here, but since there is a PyTorch version of it, I will use that
    torch.manual_seed(42)
    model = torch.nn.Linear(M, 1)

    #Loss and optimizer
    mse_loss = torch.nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    #Create dataset and loader
    dataset = torch.utils.data.TensorDataset(design_matrix, t_train_tensor) 
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    best_validation_loss = float('inf')  
    epochs_without_improvement = 0

    ##Training loop
    for epoch in range(10000):  
        for b_num, (x_batch, t_batch) in enumerate(loader):
            optimizer.zero_grad()
            
            pred = model(x_batch)

            loss = mse_loss(pred.flatten(), t_batch) 
            
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_pred = model(design_matrix_val)
            val_loss = mse_loss(val_pred.flatten(), t_val_tensor)

        if(epoch % 1000 == 0):
            print(f"Epoch: {epoch}, Train Loss: {loss.item()}, Val Loss {val_loss.item()}")

        # Early stopping condition
        improvement = best_validation_loss - val_loss.item()
        if improvement > 0.01:
            best_validation_loss = val_loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 5:
            print(f"Stopping early at epoch {epoch}")
            break

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


def train_test_split(x: np.ndarray, t: np.ndarray, np_seed: int, test_size: float = 0.2):
    """Method for performing train/test split

    Args:
        x (np.ndarray): inputs
        t (np.ndarray): targets
        np_seed (int): Numpy seed
        test_size (float, optional): Proportion of the test set. Defaults to 0.2.

    Returns:
        np.ndarray: inputs training set
        np.ndarray: inputs test set
        np.ndarray: targets training set
        np.ndarray: targets test set
    """

    num_datapoints = x.shape[0]
    indices = np.arange(num_datapoints)

    np.random.seed(np_seed)
    np.random.shuffle(indices) 

    split_idx = int(num_datapoints * (1 - test_size))

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    x_train, x_val = x[train_idx], x[val_idx]
    t_train, t_val = t[train_idx], t[val_idx]

    return x_train, x_val, t_train, t_val

def fit_polynomial_sgd_cv(x: np.ndarray, t: np.ndarray, M_list: list, lr_list: float, batch_size: int, k: int = 5) -> int:
    """Method for performing cross-validated training of a polynomial model using SGD and selecting the best model

    Args:
        x (np.ndarray): input values
        t (np.ndarray): target values
        M_list (list): List of polynomial degrees to try
        lr_list (float): List of learning rates to try
        batch_size (int): batch size
        k (int, optional): number of CV folds. Defaults to 5.

    Returns:
        int: best polynomial degree, best weights
    """
    best_M = None
    best_validation_error = float('inf')

    for l_i, M in enumerate(M_list):
        print(f"\nTraining {k} folds with M = {M}\n")
        fold_losses = []
        num_datapoints = x.shape[0]
        indices = np.arange(num_datapoints)
        np.random.shuffle(indices)  

        fold_size = num_datapoints // k

        for i in range(k):  
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size  
            val_idx = indices[start_idx:end_idx]
            train_idx = np.concatenate([indices[:start_idx], indices[end_idx:]])

            x_train, x_val = x[train_idx], x[val_idx]
            t_train, t_val = t[train_idx], t[val_idx]

            weights = fit_polynomial_sgd(x_train, t_train, M, lr_list[l_i], batch_size) 
            val_pred = polynomial_fun(weights, x_val)
            fold_losses.append(rmse(t_val, val_pred)) 

        avg_validation_error = np.mean(fold_losses)

        if avg_validation_error < best_validation_error:
            best_validation_error = avg_validation_error
            best_M = M
            best_lr = lr_list[l_i]

    print("\nRe-training on best M and all training data\n")
    best_weights = fit_polynomial_sgd(x, t, best_M, best_lr, batch_size)

    return best_M, best_weights


