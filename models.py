import torch.nn as nn
import torch.nn.functional as F

class wizardNet(nn.Module):
    def __init__(self):
        super(WizardNet, self).__init__()
        self.fc1   = nn.Linear(100, 80) # Converting matrix with 16*5*5 features to 1D matrix
        self.fc2   = nn.Linear(80, 30)
        self.fc3   = nn.Linear(30, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def WizardNet():
    return wizardNet()

