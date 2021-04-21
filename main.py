
##v1

from model import wizard
from preprocessor import data_loader

if __name__ == "__main__":
    proof=wizard()
    train_set, test_set = data_loader()
    proof.running(train_set, test_set)



