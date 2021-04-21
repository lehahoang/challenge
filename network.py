import numpy as np
from pipelines import data_loader


class wizard():
    def __init__(self,
                 input_nodes=97,
                 hidden_nodes=100,
                 output_nodes=3,
                 learning_rate=0.01,
                 epochs=1,
                 train_batch_size=1):
                 self.input_nodes = input_nodes
                 self.hidden_nodes = hidden_nodes
                 self.output_nodes = output_nodes
                 self.learning_rate = learning_rate
                 self.epochs = epochs
                 self.train_batch_size = train_batch_size                    
                 self.w1 = []
                 self.w2 = []
                 self.b1 = []
                 self.b2 = []

    def weight_initialization(self):
        
        self.w1 = np.random.uniform(0,1, (self.input_nodes, self.hidden_nodes))
        self.b1 = np.zeros((self.hidden_nodes, 1))
        self.w2 = np.random.uniform(0,1, (self.hidden_nodes, self.output_nodes))
        self.b2 = np.zeros((self.output_nodes, 1))
        

    def softmax(self,x):
        tmp = np.exp(x - np.max(x, axis = 0, keepdims = True))
        y = tmp / tmp.sum(axis = 0)
        return y

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
        
    def sigmoid_der(self,x):
        return sigmoid(x)*(1-sigmoid(x))
    
    # cost or loss function
    # def cost(target, pred):
    #     return -np.sum(target*np.log(pred))/target.shape[1]

    def convert_label(self,x, num_classes):        
        x = np.arange(num_classes) == x.reshape(x.size, 1) # Convert to one-hot coding
        x = x.astype(np.float)
        return x

    def training(self, train_set, train_label):       
        
        self.weight_initialization()
        print(train_label)
        label = self.convert_label(train_label, 3)

        for epoch in range(self.epochs): 
            for train_input, target in zip(train_set, label):     
                print('epoch:',epoch)                   
                ## Forward pass
                x = train_input.reshape(train_input.size, 1) 
                z1 = self.sigmoid(np.dot(self.w1.T, x) + self.b1)
                z2 = self.softmax(np.dot(self.w2.T, z1) + self.b2)
                
                ## Backward pass
                error = z2 - target
                theta2 = error*z2*(1-z2)
                self.w2 += self.learning_rate * np.dot(z2.T, theta2)
                self.b2 +=self.learning_rate * np.dot(theta2, z2.T)
                ## backpropagation
                tmp = np.dot(self.w2.T, error)                               
                tmp = tmp*z1*(1-z1)
                self.w1 += self.learning_rate * np.dot(tmp, x.T)
                self.b1 += self.learning_rate * np.dot(tmp, x.T)
       

    def evaluate(self, test_set, test_label):
        for test_input, target in enumerate(test_set, label):            
            x = train_input.reshape(test_input.size, 1) 
            z1 = sigmoid(np.dot(self.w1.T, x) + self.b1)
            z2 = softmax(np.dot(self.w2.T, z1) + self.b2)
            predicted_class = np.argmax(z2, axis=0)

if __name__ == "__main__":
    proof=wizard()           
    train_set, train_label, test_set, test_label = data_loader()
    proof.training(train_set, train_label)
    # wizard.evaluate(test_set, test_label)