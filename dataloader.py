import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, transforms
from torch import utils
import numpy as np



def data_preparation():
    dataset = dataprocessing(train_data_path, test_data_path, label_data_path)
    training_data, test_data, target = dataset.data_loader() # Load the train, label
    training_size = len(training_data)
    train_data = training_data[:int(training_size*0.8)]
    train_label = target[:int(training_size*0.8)]
    val_data = training_data[int(training_size * 0.8):]
    val_label = target[int(training_size * 0.8):]

    return train_data, val_data, test_data, train_label, val_label
    # dataloader = DataLoader(train, batch_size=10, shuffle=False, num_workers=2)
    # Using existing Pytorch method to load the data



def data_loader(args, train_batch_size, test_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_set =datasets.MNIST('./dataset',
                               train=True,
                               download=True,
                               transform=transform)
    test_set =datasets.MNIST('./dataset',
                             train=False,
                             download=True,
                             transform=transform)
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(args.split_fraction * num_train))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    valid_indices, train_indices = indices[:split], indices[split:]
    train_sampler= utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = utils.data.sampler.SubsetRandomSampler(valid_indices)

    train_loader = utils.data.DataLoader(dataset=train_set,
                                         batch_size=train_batch_size,
                                         sampler=train_sampler,
                                         **kwargs)

    val_loader = utils.data.DataLoader(dataset=train_set,
                                     batch_size=train_batch_size,
                                     sampler=val_sampler,
                                     **kwargs)

    test_loader = utils.data.DataLoader(dataset=test_set,
                                        batch_size=test_batch_size,
                                        shuffle=False,
                                        **kwargs)
    return train_loader, val_loader, test_loader
