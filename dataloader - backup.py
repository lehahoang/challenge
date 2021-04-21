import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, transforms
from torch import utils
import numpy as np



label_data_path = 'train_labels.txt'
train_data_path = 'train_input.txt'
test_data_path = 'test_input.txt'
# input_data_path = 'dataset/train_input.txt'

class dataprocessing(Dataset):
    def __init__(self, train_data_dir, test_data_dir, label_data_dir ):
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.label_dir = label_data_dir

    def data_loader(self):
        label_f = self.label_dir
        train_f = self.train_data_dir
        test_f =  self.test_data_dir
        train = []
        test = []
        with open(label_f) as f:
            lines = f.readlines()
            label = [int(i) for i in lines]

        with open(train_f) as f:
            lines = f.readlines()
            for l in lines:
                elements = l.split(',')
                temp = [float(i) for i in elements]
                train.append(temp)

        with open(test_f) as f:
            lines = f.readlines()
            for l in lines:
                elements = l.split(',')
                temp = [float(i) for i in elements]
                test.append(temp)

        ######### Normalizing the dataset based on min-max method ###
        scaler = preprocessing.MinMaxScaler()
        train_norm = scaler.fit_transform(train)
        test_norm =scaler.fit_transform(test)

        ######## Perform one-hot encoding for the target ############
        target = [i + 1 for i in label]  # No negative values
        target_t = torch.tensor(target, dtype=torch.int64)
        target = F.one_hot(target_t, num_classes=3)

        return train_norm, test_norm, target




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



def mnist_loader(args, train_batch_size, test_batch_size):
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


# def data_loader(input_dir, label_dir):
#     label_f = label_dir
#     input_f = input_dir
#     label = []
#     input = []
#
#     with open(label_f) as f:
#         lines = f.readlines()
#         label = [int(i) for i in lines]
#
#     with open(input_f) as f:
#         lines = f.readlines()
#         for l in lines:
#             elements = l.split(',')
#             temp = [float(i) for i in elements]
#             input.append(temp)
#
#     scaler = preprocessing.StandardScaler().fit(input)
#     input_norm = scaler.transform(input)
#     return input_norm, label
# if __name__ == "__main__":
    # training_data, label = data_loader(input_data_path, label_path)
    # plt.figure(figsize=(20,20))
    # plt.hist(training_data, label='Fancy labels', density=True)
    # plt.show()

