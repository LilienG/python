#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/28 11:04
# @File     : train.py

"""
import argparse

import torch
import torchvision
from matplotlib import pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from models.lenet import LeNet
from utils import pre_process
import utils.visualizer as vis
import cv2


def get_data_loader(batch_size):
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=pre_process.data_augment_transform(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=pre_process.normal_transform())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader


# Todo:output five wrong pictures & analyst(maybe use util.visualizer.py?)
def evaluate(model, test_loader, device):
    model.eval()
    fault = []

    with torch.no_grad():
        correct = 0
        total = 0
        err_sum = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Todo: Output fault images and analyst reason——task seven
            # method:https://blog.csdn.net/qq_24815615/article/details/105208982?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162533108116780269879519%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=162533108116780269879519&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-3-105208982.first_rank_v2_pc_rank_v29_1&utm_term=LeNet%E5%A6%82%E4%BD%95%E8%BE%93%E5%87%BA%E8%AE%AD%E7%BB%83%E9%94%99%E8%AF%AF%E7%9A%84%E6%A8%A1%E5%9E%8B&spm=1018.2226.3001.4187
            '''
            for index in range(4):
                if predicted[index] != labels[index]:
                    # print(index)
                    err_sum = err_sum+1

                    img = np.empty((28,28), dtype=np.float32)
                    img[:,:] = images[index].cpu().numpy()/2+0.5
                    img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    plt.imshow(img2)
                    plt.show()
'''
        print('Test Accuracy of the model is: {} %'.format(100 * correct / total))

    return fault


def save_model(model, save_path='lenet.pth'):
    ckpt_dict = {
        'state_dict': model.state_dict()
    }
    torch.save(ckpt_dict, save_path)


def train(epochs, batch_size, learning_rate, num_classes):

    # fetch data
    train_loader, test_loader = get_data_loader(batch_size)

    # Loss and optimizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LeNet(num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # Todo: change Adam to SGD——task four
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start train
    total_step = len(train_loader)
    X_train = []
    Y_train = []
    for epoch in range(epochs):
        loss_avg = 0
        count = 0
        for i, (images, labels) in enumerate(train_loader):

            count += 1

            # get image and label
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

            loss_avg += loss.item()

        loss_avg = loss_avg / count

        # evaluate after epoch train
        fault = evaluate(model, test_loader, device)
        for img in fault:
            print(type(img["images"]))
            vis.demo_display_single_image()

        # Todo: record Loss data——X:epoch + 1, Y:loss_avg
        X_train.append(epoch + 1)
        Y_train.append(loss_avg)

    # end for

    # Todo: draw Loss Graph——task three
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)

    pl.plot(X_train, Y_train, 'b-', label=u'Add Convolution')
    pl.legend()
    pl.xlabel(u'iters')
    pl.ylabel(u'loss')
    plt.title('Loss Graph')
    plt.savefig("loss graph 1.png")
    pl.show()

    # save the trained model
    save_model(model, save_path='lenet.pth')

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train(args.epochs, args.batch_size, args.lr, args.num_classes)


