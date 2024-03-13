import torch
import numpy as np
import time

from functions import polynomial_fun, fit_polynomial_ls, fit_polynomial_sgd, rmse

if __name__ == "__main__":

    np.random.seed(42)
    x_train = np.random.uniform(low=-20, high=20, size=20)
    np.random.seed(43)
    x_test = np.random.uniform(low=-20, high=20, size=10)

    np.random.seed(44)
    noise_train = np.random.normal(0, 0.5, 20)
    np.random.seed(45)
    noise_test = np.random.normal(0, 0.5, 10)

    weights = [1,2,3]

    y_train = polynomial_fun(weights,x_train)
    y_test = polynomial_fun(weights,x_test)

    t_train = y_train + noise_train
    t_test = y_test + noise_test

    diff_y_train = np.abs(y_train - t_train)

    print(f"\nPart a: Observed train and true values difference mean and std\n")
    print(f"t_train and y_train difference mean: {np.mean(diff_y_train):.2f} std: {np.std(diff_y_train):.2f}")

    print("\nPart b: LS-predicted values and true values difference mean and std\n")

    for m in [2,3,4]:
        w_train = fit_polynomial_ls(x_train, t_train, m)
        y_ls_train = polynomial_fun(w_train.numpy(), x_train)
        y_ls_test = polynomial_fun(w_train.numpy(), x_test)

        diff_train = np.abs(y_ls_train - y_train)
        diff_test = np.abs(y_ls_test - y_test)

        print(f"M={m}")

        print(f"train difference mean: {np.mean(diff_train):.2f} std: {np.std(diff_train):.2f}")
        print(f"test difference mean : {np.mean(diff_test):.2f} std: {np.std(diff_test):.2f}")

    print("\nTraining with SGD M = 2\n")
    sgd_start = time.time()
    w_sgd_M_2_train = fit_polynomial_sgd(x_train, t_train, 2, 1e-2, 20)
    sgd_end = time.time()
    print("Resulting weights:")
    print(w_sgd_M_2_train)

    print("\nTraining with SGD M = 3\n")
    w_sgd_M_3_train = fit_polynomial_sgd(x_train, t_train, 3, 1e-3, 20) #2e-8
    print("Resulting weights:")
    print(w_sgd_M_3_train)

    print("\nTraining with SGD M = 4\n")
    w_sgd_M_4_train = fit_polynomial_sgd(x_train, t_train, 4, 1e-3, 20) #5e-11
    print("Resulting weights:")
    print(w_sgd_M_4_train)

    y_sgd_M_2_train = polynomial_fun(w_sgd_M_2_train, x_train)
    y_sgd_M_3_train = polynomial_fun(w_sgd_M_3_train, x_train)
    y_sgd_M_4_train = polynomial_fun(w_sgd_M_4_train, x_train)

    y_sgd_M_2_test = polynomial_fun(w_sgd_M_2_train, x_test)
    y_sgd_M_3_test = polynomial_fun(w_sgd_M_3_train, x_test)
    y_sgd_M_4_test = polynomial_fun(w_sgd_M_4_train, x_test)

    diff_sgd_M_2_train = np.abs(y_sgd_M_2_train - y_train)
    diff_sgd_M_3_train = np.abs(y_sgd_M_3_train - y_train)
    diff_sgd_M_4_train = np.abs(y_sgd_M_4_train - y_train)

    diff_sgd_M_2_test = np.abs(y_sgd_M_2_test - y_test)
    diff_sgd_M_3_test = np.abs(y_sgd_M_3_test - y_test)
    diff_sgd_M_4_test = np.abs(y_sgd_M_4_test - y_test)

    print("\nPart c: SGD-predicted values and true values difference mean and std\n")

    print(f"SGD M=2 train difference mean: {np.mean(diff_sgd_M_2_train):.2f} std: {np.std(diff_sgd_M_2_train):.2f}")
    print(f"SGD M=3 train difference mean: {np.mean(diff_sgd_M_3_train):.2f} std: {np.std(diff_sgd_M_3_train):.2f}")
    print(f"SGD M=4 train difference mean: {np.mean(diff_sgd_M_4_train):.2f} std: {np.std(diff_sgd_M_4_train):.2f}")
    print()
    print(f"SGD M=2 test difference mean: {np.mean(diff_sgd_M_2_test):.2f} std: {np.std(diff_sgd_M_2_test):.2f}")
    print(f"SGD M=3 test difference mean: {np.mean(diff_sgd_M_3_test):.2f} std: {np.std(diff_sgd_M_3_test):.2f}")
    print(f"SGD M=4 test difference mean: {np.mean(diff_sgd_M_4_test):.2f} std: {np.std(diff_sgd_M_4_test):.2f}")

    
    print("\nPart d: Comparing accuracy of LS and SDG\n")

    ls_start = time.time()
    w_train = fit_polynomial_ls(x_train, t_train, 2)
    ls_end = time.time()

    y_ls_test = polynomial_fun(w_train.numpy(), x_test)

    rmse_ls_y_test = rmse(y_ls_test, y_test)
    rmse_ls_w = rmse(w_train.numpy(), weights)

    rmse_sgd_y_test = rmse(y_sgd_M_2_test, y_test)
    rmse_sgd_w = rmse(w_sgd_M_2_train, weights)

    print(f"LS RMSE y_test : {rmse_ls_y_test:.2f} weights: {rmse_ls_w:.2f}")
    print(f"SGD RMSE y_test: {rmse_sgd_y_test:.2f} weights: {rmse_sgd_w:.2f}")
   
    ls_time = ls_end - ls_start
    sgd_time = sgd_end - sgd_start
    print("\nPart e: Comparing time of LS and SDG\n")
    print(f"LS time : {ls_time:.4f}")
    print(f"SGD time: {sgd_time:.2f}")


    


    