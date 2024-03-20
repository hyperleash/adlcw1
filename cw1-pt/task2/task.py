from functions import train_model, test_all_checkpoints, save_example
from mixup import mixup_demo


if __name__ == '__main__':

    #save demo images for mixup. Demo version with lambda = 0.5 for better visibility
    mixup_demo()

    epochs = 20

    train_model(1, "vit_s1", epochs, 20)
    test_all_checkpoints("vit_s1", epochs)
    #Uses matplotlib, can't be used in the environment, corresponding PNGs are in the task directory
    #save_plot("vit_s1")
    save_example("vit_s1", epochs)

    train_model(2, "vit_s2", epochs, 20)
    test_all_checkpoints("vit_s2", epochs)
    #Uses matplotlib, can't be used in the environment, corresponding PNGs are in the task directory
    #save_plot("vit_s2")
    save_example("vit_s2", epochs)

