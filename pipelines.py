import numpy as np



label_data_path = 'train_labels.txt'
train_data_path = 'train_input.txt'
test_data_path = 'test_input.txt'
# input_data_path = 'dataset/train_input.txt'

label_f = label_data_path
train_f = train_data_path 
test_f =  test_data_path 
train_label = []
train_data = []
n_classes = 3

def data_loader():
    global train_data
    with open(label_f) as f:
        lines = f.readlines()
        train_label = np.array(list(map(lambda x: int(x)+1, lines)))

    with open(train_f) as f:
        lines = f.readlines()        
        for l in lines:
            elements = l.split(',')
            temp = [float(i) for i in elements]
            train_data.append(temp)
        train_data = np.array(train_data)

    return train_data, train_label, None, None

def convert_label(x, num_classes):
    # x = np.arange(num_classes) == x.reshape(label.size, 1) # Convert to one-hot coding
    # x = x.astype(np.float)
    x = np.arange(num_classes) == x.reshape(x.size, 1) # Convert to one-hot coding
    x = x.astype(np.float)
    return x
if __name__=="__main__":
    y,x,_,_ = data_loader()
    print(x)
    x = convert_label(x,4)
    print(x)








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

