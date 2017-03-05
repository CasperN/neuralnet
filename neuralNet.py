import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_training_data(xfile = 'TrainDigitX.csv',yfile = 'TrainDigitY.csv'):
    '''
    Reads data, makes it useable for the NN later
    '''
    xs = pd.read_csv(xfile,header=None)
    # Convert to numpy array. Add extra all 1 feature for bias parameter
    xs = np.insert(xs.values,0,1,1)

    y = pd.read_csv(yfile,header=None)
    # Convert to a matrix
    ys = np.zeros((y.values.shape[0],10))
    for row,col in enumerate(y.values):
        ys[row,col] = 1

    return xs,ys

# Neural Net, Fully connected between layers, Sigmoid transfer function

def sigmoid(t):
    # Our transfer function
    return 1/(1+np.exp(-t))

def der_sigmoid(t):
    # Derivative of sigmoid
    return sigmoid(t) * (1-sigmoid(t))

def backpropagate(x,y,weights,learnRate):
    '''
    Updates weights by training on input x with label y
    This should be rewritten with as a for loop for arbitrary depth
    '''
    # Hidden layer (x is output of input layers)
    h_in  = x.dot(weights[0]) / len(x)
    h_out = sigmoid(h_in)
    # Output layer
    o_in  = h_out.dot(weights[1]) / len(h_out)
    o_out = sigmoid(o_in)
    error = o_out - y
    # Back Propagate last layer
    delta1 = error * der_sigmoid(o_in)
    weights[1] -= learnRate * np.outer(h_out,delta1)
    # Back Propagate hidden layer
    delta0 = der_sigmoid(h_in) * weights[1].dot(delta1)
    weights[0] -= learnRate * np.outer(x,delta0)

def train(xs,ys,N_hidden=200,learnRate = 10):
    '''
    trains the neural net on xs and ys
    '''
    num_inputs = xs.shape[1]
    num_hidden = N_hidden
    num_output = 10
    # Randomly initialize weights
    # weights[0] is a matrix of weights from layer 0 to layer 1
    weights = []
    weights.append(np.random.random((num_inputs,num_hidden)))
    weights.append(np.random.random((num_hidden,num_output)))
    idx = np.random.randint(0,10,len(xs))
    for x,y in zip(xs,ys):
        backpropagate(x,y,weights,learnRate)
    return weights

def classifier(x,weights):
    # Hidden layer (x is output of input layers)
    h_in  = x.dot(weights[0]) / len(x) # This seems to help numerical error...
    h_out = sigmoid(h_in)
    # Output layer
    o_in  = h_out.dot(weights[1]) / len(h_out)
    o_out = sigmoid(o_in)
    return np.argmax(o_out)

def test(xs,ys,weights):
    '''
    Counts errors in the neural net
    '''
    n_cor = sum(classifier(x,weights)==np.argmax(y) for x,y in zip(xs,ys)) + .0
    return 1 - n_cor/len(xs)

def k_cross_validate(xs,ys,k = 10,n_hidden = 200, learnRate = 10):
    idx = np.random.randint(0,k,len(xs))
    errs = []
    for i in range(k):
        xtr = xs[idx != i]
        ytr = ys[idx != i]
        xte = xs[idx == i]
        yte = ys[idx == i]
        weights = train(xtr,ytr,n_hidden,learnRate)
        errs.append(test(xte,yte,weights))
    print('{} +/- {}'.format(np.mean(errs),np.std(errs)))
    return np.mean(errs)

def test_parameters():
    xs,ys = get_training_data()
    res = ''
    nHiddens = np.round(2**(np.arange(4,9,.25)))
    learnRates = np.exp(np.arange(-1,3,.2))
    err = []
    for lr in learnRates:
        for nh in nHiddens:
            print(lr,nh)
            err.append(
                k_cross_validate(xs,ys, k=10, n_hidden = nh, learnRate = lr))
    err = np.reshape(err,(len(learnRates),len(nHiddens)))

    # Plot
    plt.imshow(1-er[::-1,],
      extent = (nHiddens.min(),nHiddens.max(),learnRates.min(),learnRates.max())
      interpolation='none',cmap = 'Greys',aspect=30)
    plt.savefid('lr_nh_fig')
    return nHiddens, learnRates, err
