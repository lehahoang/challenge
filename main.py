from dataloader import data_preparation
from models import WizardNet
from pipelines import *
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--net', type=str, help='Class of DNN model')
    parser.add_argument('--train-batch-size', type=int, default=256,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='Batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs for train process')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--out-dir', type=str, default='trained_models/',
                        help='output directory to save the model')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Showing loging status after a certain number of batches ')
    parser.add_argument('--split-fraction', type=float, default =0.1,
                        help='fraction between train/validation set')

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    train_data, val_data, test_data, train_label, val_label = data_preparation()
    model = WizardNet
    #Training
    print("|| Starting the training phase")
    print("|| Number of epochs:", args.epochs)
    if use_cuda: print("|| Training model on GPU\n")
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(1, args.epochs + 1):
        train(args.log_interval, model, device, train_data, optimizer, epoch)
        val(model, device, val_set)
    test(model, device, val_data)
    torch.save(model.state_dict(), args.out_dir +  "model.pt")
#evaluating
    model.train()
    model.test()

class model(self):
    def __int__(self):
        pass
    def __train__(self):

    def __test__(self):
        pass