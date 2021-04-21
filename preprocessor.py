##v1
import numpy as np
from sklearn import preprocessing



label_data_path = 'dataset/train_labels.txt'
train_data_path = 'dataset/train_input.txt'
test_data_path = 'dataset/test_input.txt'
# input_data_path = 'dataset/train_input.txt'

label_f = label_data_path
train_f = train_data_path 
test_f =  test_data_path 
train_label = []
train_data = []
test_data = []
test_label = []
n_classes = 3

def data_loader():
    global train_data
    global test_data
    with open(train_f) as f:
        lines = f.readlines()        
        for l in lines:
            elements = l.split(',')
            temp = [float(i) for i in elements]
            train_data.append(temp)
        train_data = np.array(train_data)
        scaler = preprocessing.StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)

    with open(label_f) as f:
        lines = f.readlines()
        train_label = np.array(list(map(lambda x: int(x)+1, lines)))

    train_t = np.array(train_data).T # Transpose
    merge = np.vstack((train_label,train_t)) # stack them togheter
    train_data = merge.T # Transpose again

    # output = np.array(list(map(lambda x: int(x)-1, train_label)), dtype=np.int8)
    # np.savetxt('out.txt', output, delimiter = ',', fmt='%s')
    
    with open(test_f) as f:
        lines = f.readlines()        
        for l in lines:
            elements = l.split(',')
            temp = [float(i) for i in elements]
            test_data.append(temp)
        test_data = np.array(test_data)
        scaler = preprocessing.StandardScaler().fit(test_data)
        test_data = scaler.transform(test_data)

    return train_data, test_data


if __name__=="__main__":
    train,test = data_loader()
    print(test.shape)










# label = np.array(label).T # Transpose
# label = np.arange(n_classes) == label.reshape(label.size, 1) # Convert to one-hot coding
# label = label.astype(np.float)


# with open(train_f) as f:
#     lines = f.readlines()
#     for l in lines:
#         elements = l.split(',')
#         temp = [float(i) for i in elements]
#         train.append(temp)
# train_t = np.array(train).T # Transpose
# merge = np.vstack((label,train_t)) # stack them togheter
# out = merge.T # Transpose again

# # out_str = list(map(str, out)) # map them back to string

# np.savetxt('merge.txt', np.matrix(out), delimiter = ',') # save them to the input data file.


# with open(test_f) as f:
#     lines = f.readlines()
#     for l in lines:
#         elements = l.split(',')
#         temp = [float(i) for i in elements]
#         test.append(temp)





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

