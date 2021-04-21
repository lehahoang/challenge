
##v1
import argparse
from network import wizard
from pipelines import *

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
    args = parser.parse_args()
    
    train_data, train_label, test_data, test_label = data_loader()

   
    model = wizard(split_ratio=args.split_fraction)
    model.training(train_set, train_label)
    model.evaluate(test_data, test_label)



