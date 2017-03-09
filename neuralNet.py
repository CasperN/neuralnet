import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import multiprocessing
import itertools

def get_training_data(xfile = 'TrainDigitX.csv',yfile = 'TrainDigitY.csv'):
    '''
    Reads data, makes it useable for the NN later
    '''
    xs = pd.read_csv(xfile,header=None)
    # Add extra all 1 feature for bias parameter
    xs = np.insert(xs.values,0,1,1)
    y = pd.read_csv(yfile,header=None)
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

def initialize_weights( nodesPerLayer ):
    '''
    Returns a list of matrices
    Currently assuming 1 hidden layer but should be rewritten to generalize
    '''
    nInputs, nHidden, nOutputs = nodesPerLayer
    # weights[0] is a matrix of weights from layer 0 to layer 1
    weights = []
    # Randomly initialize weights w_ij in [-1,1)
    weights.append(2 * np.random.random((nInputs,nHidden)) - 1)
    weights.append(2 * np.random.random((nHidden,nOutputs)) - 1)
    return weights

def backpropagate(x,y,weights,learnRate):
    '''
    Updates weights by training on input x with label y
    This should be rewritten with as a for loop for arbitrary depth
    '''
    # Hidden layer (x is output of input layers)
    h_in  = x.dot(weights[0])
    h_out = sigmoid(h_in)
    # Output layer
    o_in  = h_out.dot(weights[1])
    o_out = sigmoid(o_in)
    error = o_out - y
    # Calculate change
    delta1 = error * der_sigmoid(o_in)
    delta0 = der_sigmoid(h_in) * weights[1].dot(delta1)
    # Back Propagate
    weights[1] -= learnRate * np.outer(h_out,delta1)
    weights[0] -= learnRate * np.outer(x,delta0)

def test(xs,ys,weights):
    '''
    Counts errors in the neural net
    '''
    n_cor = sum(classifier(x,weights)==np.argmax(y) for x,y in zip(xs,ys)) + .0
    return 1 - n_cor/len(xs)

def train_dynamic(xs,ys,nHidden=200,learnRate = 10,rounds=15):
    '''
    trains the neural net on xs and ys
    '''
    nodesPerLayer = [int(xs.shape[1]), int(nHidden), int(ys.shape[1])]
    weights = initialize_weights(nodesPerLayer)
    # make a test set and a training set, test with 10% of data
    k = np.random.randint(0,10,len(xs))
    xtr = xs[k > 0];  xte = xs[k == 0]
    ytr = ys[k > 0];  yte = ys[k == 0]
    # Train until error stops decreasing
    prev_err = 1.1;  err = 1
    while err < prev_err:
        prev_err = err
        for x,y in zip(xtr,ytr):
            backpropagate(x,y,weights,learnRate)
        err = test(xte,yte,weights)
    return weights

def classifier(x,weights):
    '''
    Classify x by the neural net defined by weights
    '''
    # Hidden layer (x is output of input layers)
    h_in  = x.dot(weights[0])
    h_out = sigmoid(h_in)
    # Output layer
    o_in  = h_out.dot(weights[1])
    o_out = sigmoid(o_in)
    return np.argmax(o_out)

def k_cross_validate(xs,ys,k = 10,nHidden = 200, learnRate = 10):
    '''
    Finds error of parameters over input data via k wise cross validation
    '''
    idx = np.random.randint(0,k,len(xs))
    errs = []
    for i in range(k):
        # cross validation training / testing sets
        xtr = xs[idx != i];  xte = xs[idx == i]
        ytr = ys[idx != i];   yte = ys[idx == i]
        # Train and get error
        weights = train_dynamic(xtr,ytr,nHidden,learnRate)
        errs.append(test(xte,yte,weights))
    print('nHidden: {}\tlearnRate: {} \tC.V.Error: {} +/- {}'.format(
       nHidden, learnRate, np.round(np.mean(errs),5), np.round(np.std(errs),5)))
    return np.mean(errs)

def kcsv_mp_wrapper(args):
    # k_cross_validate multiprocessing wrapper
    nHidden, learnRate, xs, ys = args
    err = k_cross_validate(xs,ys,k=10,nHidden=nHidden,learnRate=learnRate)
    return nHidden, learnRate, err

def test_parameters_pool(xs,ys):
    '''
    Test parameters, graph accuracy
    '''
    nHidden = np.arange(90,141,10)
    learnRates = np.round(np.exp(np.arange(-3,0,.4)),3)
    # Multiprocess and map
    p = multiprocessing.Pool(multiprocessing.cpu_count()-1) # -1 so I can still do things
    params = [(nh,lr,xs,ys) for nh,lr in itertools.product(nHidden,learnRates)]
    error_rates = p.map(kcsv_mp_wrapper, params)
    p.close()
    p.terminate()
    # Put in dataframe then graph
    res = pd.DataFrame(index = learnRates, columns = nHidden)
    for nh,lr, er in error_rates:
        res.loc[lr,nh] = er
    vals = res.values.astype(float)
    # Scale error from 0 - .9 so colors are meaningful and we can see
    heat = 1 - vals
    heat = (heat - heat.min()) *.9 / (heat.max()- heat.min())  + .1
    plt.table(cellText  = vals,
            rowLabels   = res.index,
            colLabels   = res.columns,
            loc         = 'center',
            cellColours = plt.cm.hot(heat) )
    plt.axis('off')
    plt.show()

def train_epochs(xtr, ytr, xte, yte, learnRate, nHidden, maxEpoch):
    '''
    Trains data for some epochs, reports training error for each epoch
    '''
    # initialize_weights
    nodesPerLayer = [int(xtr.shape[1]), int(nHidden), int(ytr.shape[1])]
    weights = initialize_weights(nodesPerLayer)
    # Train over data test weights after each epoch
    epochError = []
    for i in range(maxEpoch):
        for x,y in zip(xtr,ytr):
            backpropagate(x,y,weights,learnRate)
        epochError.append( test(xte,yte,weights))
        print('{} finished {}/{}'.format(
            multiprocessing.current_process(), i+1, maxEpoch ))
    return epochError

def kcv_epoch_grapher(xs,ys, learnRate = .2, nHidden = 110, k = 10, maxEpoch = 100):
    '''
    k wise split then train the net and graph how error changes with epoch
    '''
    # Split training data k-wise
    idx = np.random.randint(0,k,len(xs))
    xte = [ xs[idx == i] for i in range(k) ]
    yte = [ ys[idx == i] for i in range(k) ]
    xtr = [ xs[idx != i] for i in range(k) ]
    ytr = [ ys[idx != i] for i in range(k) ]
    # Map train_epochs over a pool
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    argGen = ( (xtr[i], ytr[i], xte[i], yte[i],
                learnRate, nHidden, maxEpoch) for i in range(k) )
    errors = pool.starmap(train_epochs,argGen)
    pool.close()
    pool.terminate()
    # Plot errors
    for er in errors:
        plt.plot(range(1,maxEpoch+1),er)
    plt.title('Epoch vs Error on {} nodes and learn rate {}'.format(nHidden,learnRate))
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()


if __name__ == '__main__':
    xs,ys = get_training_data()
    # this stops it from crashing somehow, its an OSX bug apparently
    multiprocessing.set_start_method('spawn')
    print('read data')
    #test_parameters_pool(xs,ys)
    kcv_epoch_grapher(xs,ys)
