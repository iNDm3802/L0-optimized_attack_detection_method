from math import sqrt
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def amplify(image, mask, cutoff: float = 0):
    img = image

    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            if mask[i][j] == 0:
                for c in range(0, 3):
                    img[i][j][c] = 0
            else:
                for c in range(0, 3):
                    img[i][j][c] = image[i][j][c]

    return img


def normalize(a):
    a_min = a.min(axis=(0, 1), keepdims=True)
    a_max = a.max(axis=(0, 1), keepdims=True)
    return (a - a_min) / (a_max - a_min)


def compute_accuracy(TP: int, FP: int, TN: int, FN: int):
    try:
        return (TP + TN) / (TP + FP + TN + FN)
    except BaseException:
        return 0


def compute_precision(TP: int, FP: int, FN: int):
    try:
        return TP / (TP + FP)
    except BaseException:
        return 0


def compute_recall(TP: int, FP: int, FN: int):
    try:
        return TP / (TP + FN)
    except BaseException:
        return 0


def compute_F(precision: float, recall: float, beta: float = 1):
    try:
        return (1 + pow(beta, 2)) * precision * recall / (pow(beta, 2) * precision + recall)
    except BaseException:
        return 0


def compute_MCC(TP: int, FP: int, TN: int, FN: int):
    try:
        return (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    except BaseException:
        return -1


def get_clear_images(count: int, path: str): #todo DELETE
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=True)

    numder = 0
    for _ in range(0, count % 10000 + 1):
        cln_data, true_label = next(iter(loader))
        for i in range(cln_data.shape[0]):
            cln = cln_data[i].cpu().detach().numpy().transpose(1, 2, 0)
            numder += 1
            n = f'{path}c_{numder}.png'

            plt.imsave(n, cln)
            if numder >= count:
                return
    return
