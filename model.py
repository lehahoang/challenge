##v1
import numpy as np
from preprocessor import data_loader
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


# fit the keras model on the dataset

# evaluate the keras model



class wizard():
    def __init__(self,
                 input_nodes=97,
                 output_nodes=3,
                 learning_rate=0.1,
                 train_batch_size=100,
                 split_ratio=0.3):
                 self.input_nodes = input_nodes
                 self.output_nodes = output_nodes
                 self.learning_rate = learning_rate
                 self.train_batch_size = train_batch_size
                 self.split_ratio = split_ratio

    def modelConfigure(self):
        model = Sequential()
        model.add(Dense(120, input_dim=self.input_nodes, activation='relu'))
        model.add(Dense(70, input_dim=120, activation='relu'))
        model.add(Dense(30, input_dim=70, activation='relu'))
        model.add(Dense(self.output_nodes, activation='softmax'))
        # compile the keras model
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        return model

    def running(self, train_set, test_set):
        model = self.modelConfigure()
        train_data = []
        train_label =[]
        val_data = []
        val_label = []

        split = int(self.split_ratio* train_set.shape[0])
        train_data = train_set[:split, 1:]
        train_label = train_set[:split, 0]
        val_data = train_set[split :,1:]
        val_label = train_set[split :,0]
        test_data = test_set
        delta =0
        epoch =0

        while delta<=10 and epoch<10:
            epoch+=1
            print("Epoch:", epoch)
            model.fit(train_data, to_categorical(train_label), epochs=1, batch_size=self.train_batch_size)
            _, accuracy_train = model.evaluate(train_data, to_categorical(train_label), verbose=0)
            print('Training f1-score: %.2f' % (accuracy_train * 100))
            _, accuracy_val = model.evaluate(val_data, to_categorical(val_label), verbose=0)
            print('Validation accuracy: %.2f' % (accuracy_val * 100))
            delta = abs(accuracy_val - accuracy_train)


        print("Training done*****************************")
        pred = model.predict_classes(test_data, verbose=0)
        print('Test', pred )
        output = np.array(list(map(lambda x: int(x)-1, pred)), dtype=np.int8)
        np.savetxt('out.txt', output, delimiter = ',', fmt='%s')

if __name__ == "__main__":
    proof=wizard(epochs=1)
    train_set, test_set = data_loader()
    proof.running(train_set, test_set)
