import torch
import numpy as np
import time
from functions import polynomial_fun, fit_polynomial_ls, fit_polynomial_sgd, rmse, fit_polynomial_sgd_lasso

if __name__ == "__main__":
    print("\nTask 1a\n")

    np.random.seed(42)
    x_train = np.random.uniform(low=-20, high=20, size=1000)
    np.random.seed(43)
    x_test = np.random.uniform(low=-20, high=20, size=10)

    np.random.seed(44)
    noise_train = np.random.normal(0, 0.5, 1000)
    np.random.seed(45)
    noise_test = np.random.normal(0, 0.5, 10)

    weights = [1,2,3]

    y_train = polynomial_fun(weights,x_train)
    y_test = polynomial_fun(weights,x_test)

    t_train = y_train + noise_train
    t_test = y_test + noise_test

    print("\nTraining with SGD M = 10\n")
    
    w_sgd_lasso = fit_polynomial_sgd_lasso(x_train, t_train, 4, 1e-3, 64, 1000)
   
    print("Resulting weights:")
    print(w_sgd_lasso)