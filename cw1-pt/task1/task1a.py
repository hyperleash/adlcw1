import torch
import numpy as np
import time
from functions import polynomial_fun, fit_polynomial_ls, fit_polynomial_sgd, rmse, fit_polynomial_sgd_cv

if __name__ == "__main__":
    print("\nTask 1a\n")

    np.random.seed(42)
    x_train = np.random.uniform(low=-20, high=20, size=1000)
    np.random.seed(43)
    x_test = np.random.uniform(low=-20, high=20, size=200)

    np.random.seed(44)
    noise_train = np.random.normal(0, 0.5, 1000)
    np.random.seed(45)
    noise_test = np.random.normal(0, 0.5, 200)

    weights = [1,2,3]

    y_train = polynomial_fun(weights,x_train)
    y_test = polynomial_fun(weights,x_test)

    t_train = y_train + noise_train
    t_test = y_test + noise_test

    print("\nFinding best M with k-fold cross validation\n")
    
    best_M, best_weights = fit_polynomial_sgd_cv(x_train, t_train, [2,3,4], [1e-2, 1e-3, 1e-3], 20, 5)
   
    print(f"\nBest weights: {best_weights}")
    print(f"Best M: {best_M}")

    print("\nDifference between predicted values and true polynomial curve:\n")
    y_sgd_train = polynomial_fun(best_weights, x_train)
    y_sgd_test = polynomial_fun(best_weights, x_test)

    diff_train = np.abs(y_sgd_train - y_train)
    diff_test = np.abs(y_sgd_test - y_test)

    print(f"train difference mean: {np.mean(diff_train):.2f} std: {np.std(diff_train):.2f}")
    print(f"test difference mean : {np.mean(diff_test):.2f} std: {np.std(diff_test):.2f}")
