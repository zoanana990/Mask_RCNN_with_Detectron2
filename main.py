from trainer import *

import glob
import torch

if __name__ == '__main__':
    """
    Here are some example for this project
    """

    trainer = Detector()
    trainer.train()
    test_dataset = "COCO"
    image_path = "./Data/Fold/" + test_dataset + "/JPEGImages/*.jpg"
    total = int(len(glob.glob(image_path)))
    for i, image in enumerate(glob.glob(image_path)):
        progress_bar(i, total, color=colorama.Fore.YELLOW)
        trainer.Save_Prediction(image, "./output/" + test_dataset + "/Result", "model_final.pth.pth")
        trainer.Save_Mask(image, "./output/" + test_dataset + "/Mask", "model_final.pth.pth")

    ground_truth = "./Data/Mask/" + test_dataset + "/"
    prediction = "./output/" + test_dataset + "/Mask/*.jpg"
    dice = dice_folder(ground_truth, prediction)
    print(dice)