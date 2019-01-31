import numpy as np

def initialize_parameters(nw_archi, init_method):
    """
    Initialize weights of neural neural_network according to chosen
    method.

    Args:
        nw_archi (list): Position in List corresponds to layer and value
                         corresponds to number of units at that layer.
                         Index 0 is for input layer and last item is for
                         output units.
        method (str): method for initializing weights of neural network.
                      "he" for deep nets with relu activations, xavier for tanh
                      and random for shallow networks.

    Returns:
        parameters (dict): Initial weights of neural network in the form
                           {'Wi': (a_i, a_i-1), 'Bi': (a_i, 1)} for each layer.
    """

    valid_methods = ('random', 'he', 'xavier')
    assert method in valid_methods, "Invalid method entered. \
                                     Parameter initialization fail!"
    parameters = {}

    for i in range(1, len(nw_archi)):
        if method == 'random':
            scale_factor = 0.01
        elif method == 'he':
            scale_factor = np.sqrt(2/nw_archi[i-1])
        elif method == 'xavier':
            scale_factor = np.sqrt(1/nw_archi[i-1])
        parameters['W' + str(layer)] = \
              np.random.randn(nw_archi[i], nw_archi[i-1]) * scale_factor
        parameters['B' + str(layer)] = np.zeros(nw_archi[i], 1)

        assert parameters['W' + str(i)].shape == (nw_archi[i], nw_archi[i-1])
        assert parameters['B' + str(i)].shape == (nw_archi[i], 1)

    return parameters


def relu(Z):
    """"
    Implements relu activation function. For each element z in Z,
    if z >= 0, relu(z) = z
    elif z < 0, relu(z) = 0

    Args:
        Z (np.array): Input to activation function

    Returns:
        A (np.array): Output of relu activation
    """"
    A = np.maximum(0, Z)
    return A


def sigmoid(Z):
    """
    Implements the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-Z))


def single_fw_prop(A_prev, W, b, activation):
    """
    Compute one step of forward propagation consisting
    Z = W * A_prev + b
    A = activation(Z)

    Args:
        A_prev (np.array): (n_i-1, m) input to layer i
        W (np.array): (n_i, n_i-1) weights for layer i
        b (np.array): (n_i, 1) bias for layer i
        activation (func): activation function for layer i

    Returns:
        A (np.array): (n_i, m) output of layer i
        cache (tuple): Containing A_prev(input to layer i),
                       W, b (parameters of layer i) and
                       Z (linear combination of parameters and A_prev) for
                       reuse in backpropagation
    """
    Z = np.dot(W, A_prev) + b
    A = activation(Z)
    cache = (A_prev, W, b, Z)
    return A, cache


def fw_prop(X, parameters):
    """
    Computes full forward propagation thru network to generate outputself.
    Assumes relu activations for layers 1 to L-1, and sigmoid activation for
    layer L. By convention, input is layer 0

    Args:
        X (np.array): (n_0, m) input of m training examples with n_0 features
        parameters (dict): contains the weights and biases for each layer
                           of neural network

    Returns:
        AL (np.array): (n_L, m) output of neural network.
        caches (list): List of tuples storing A_prev, W, b, Z for each layer
    """
    L = len(parameters) // 2  # number of layers of neural network
    caches = []
    A_prev = X

    for i in range(1, L):
        # Relu activation for layers 1 to L-1
        W = parameters['W' + str(i)]
        b = parameters['B' + str(i)]
        A, cache = single_fw_prop(A_prev, W, b, relu)
        A_prev = A
        caches.append(cache)

    # Forward Prop thru Output Layer L using Sigmoid Activation
    WL = parameters['W' + str(L)]
    BL = parameters['B' + str(L)]
    AL, cache = single_fw_prop(A_prev, WL, BL, sigmoid)
    caches.append(cache)

    return AL, caches


def compute_cost(Y, AL):
    """
    Compute cross-entropy loss over m examples for logistic regression
    cost = -np.sum([Y*log(AL) + (1-Y)*log(1-AL)], axis=1)

    Args:
        Y (np.array): (1,m) shape labels
        AL (np.array): (1,m) shape predicted probabilities
    Returns:
        cost (float): loss value over m examples
    """
    cost = -np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL), axis=1)
    return cost


def cross_entropy_backprop(Y, AL):
    """
    Compute the error gradient wrt to output AL i.e. dAL
    dAL = -Y/AL + (1-Y)/(1-AL) for (n_o, m) shape. Should not sum over m
    examples as (n_o, m) rather than (n_o, 1) is required by the activation
    backprop step

    Args:
        Y (np.array): (1, m) shape binary output labels
        AL (np.array): (1, m) shape log prob predictions

    Returns:
        dAL (np.array): (1, m) shape cost gradient wrt AL for logistic
                        regression cost function
    """
    dAL = -Y/AL + (1-Y)/(1-AL)  #(n_o, m)
    return dAL


def sigmoid_backprop(dA, cache):
    """
    Backpropagation over a sigmoid activation function

    Args:
        dA (np.array): (n_i, m) output gradient
        cache (tuple): contains A_prev, W, b, Z for layer i

    Returns:
        dZ (np.array): (n_i, m) gradient wrt Z
    """
    A_prev, W, b, Z = cache  # unpack cache tuple for layer
    A = sigmoid(Z)

    dAdZ = A * (1-A)  #(n_i, m) * (n_i, m)
    dZ = dA * dAdZ  #(n_i, m) * (n_i, m)

    return dZ


def relu_backprop(dA, cache):
    """
    Backpropagation over relu activation function
    relu fw_prop: if z>=0, a=z, if z<0, a=0
    relu bw_prop: if z>=0, dadz=1, if z<0, da/dz=0

    Args:
        dA (np.array): (n_i, m) output gradient
        cache (tuple): contains A_prev, W, b, Z for layer i

    Returns:
        dZ (np.array): (n_i, m) gradient wrt Z
    """
    A_prev, W, b, Z = cache
    dZ = dA.copy()  # Set dZ = dA, equivalent to setting dadz=1 for all elements
    dZ[Z<0] = 0  # Set dadz=0 for Z<0, since dZ = dA*dadz, dZ = dA when z>=0 and
                 # dZ =0 when z<0
    return dZ


def single_linear_backprop(dZ, cache):
    """
    Single Backpropagation from dZ to dA_prev, dW and dB components

    Z = W * A_prev + b
    dA_prev = dZ * dZdA_prev where dA_prev = W
    dW = dZ * dZdW where dZdW = A_prev, averaged over m examples
    dB = dZ * dZdB where dZdB = 1, averaged over m examples

    Args:
        dZ (np.array): (n_i, m) shape dZ gradient
        cache (tuple): contains A_prev, W, B, and Z for layer i

    Returns:
        dA_prev (np.array): (n_i-1, m) shape gradient at input layer
        dW (np.array): (n_i, n_i-1) shape gradient of weights at layer i
        dB (np.array): (n_i, 1) shape gradient of biases at layer i
    """
    A_prev, W, b, Z = cache
    m = A_prev.shape[1]

    dA_prev = np.dot(W.T, dZ)   #(n_i-1, n_i) * (n_i, m) = (n_i, m)
    dW = 1/m * np.dot(dZ, A_prev.T)  #(n_i, m) * (n_i-1, m).T = (n_i, n_i-1)
    dB = 1/m * np.sum(dZ, axis=1)  #(n_i, m)
    return dA_prev, dW, dB


def full_backprop(Y, AL, caches):
    """
    Computes the gradients of a fully connected fit forward neural network
    Assumes cross-entropy cost function, sigmoid output activation and
    relu activation for all prior layers

    Args:
        Y (np.array): (1, m) shaped output binary labels
        AL (np.array): (1, m) shaped output probability of Y=1 predictions
        caches (list): length L list of tuples containing A_prev, W, B, Z for
                       use in backprop calculation of each layer i

    Returns:
        gradients (dict): Gradients of parameters of each layer in the form
                          {'dWi': (a_i, a_i-1), 'dBi': (a_i, 1)} for all layers.
    """
    L = len(caches)
    gradients = {}

    dAL = cross_entropy_backprop(Y, AL)
    cache = caches.pop()
    dZ = sigmoid_backprop(dAL, cache)
    dA_prev, dW, dB = single_linear_backprop(dZ, cache)
    gradients['dW' + str(L)] = dW
    gradients['dB' + str(L)] = dB

    for i in reversed(range(1, L)):
        cache = caches.pop()
        dZ = relu_backprop(dA_prev, cache)
        dA_prev, dW, dB = single_linear_backprop(dZ, cache)
        gradients['dW' + str(i)] = dW
        gradients['dB' + str(i)] = dB

    return gradients


def update_params(parameters, gradients, learning_rate):
    """
    Update the parameters of network using gradient descent

    Args:
        parameters (dict): Original parameters in the form
                           {W1: np.array, B1: np.array ..., WL:..., BL:...}
        gradients (dict): Gradient to adjust parameters in the form
                          {dW1: np.array, dB1: np.array ..., dWL:..., dBL:...}
        learning_rate (float): Update factor hyperparameter

    Returns:
        parameters (dict): Updated parameters in the form
                           {W1: np.array, B1: np.array ..., WL:..., BL:...}
    """
    L = len(parameters) // 2  # Number of layers of neural network

    for i in range(1, L+1):
        # Update W1, B1 ... WL, BL
        parameters['W' + str(i)] -= learning_rate * gradients['dW' + str(i)]
        parameters['B' + str(i)] -= learning_rate * gradients['dB' + str(i)]

    return parameters


def nn_fit(X, Y, nw_archi, wt_init_method='he',
           learning_rate=0.007, num_iterations=10000,
           print_cost=True):
    """
    Fit a neural_network model using gradient descent
    Args:
        X (np.array): Training examples with shape (a_0, m) as inputs to neural
                      network
        Y (np.array): (1,m) output labels by neural network
        nw_archi (list): Position in List corresponds to layer and int value
                         corresponds to number of units at that layer.
                         Index 0 is for input layer and last item is for
                         output units.
        wt_init_method (str): parameters initialization methods. Available ones
                              are 'he' for deep networks using relu, 'xavier'
                              for deep nets using tanh and 'random' for shallow
                              nets. Defaults to 'he'.
        learning_rate (float): scale factor used to update of network weights
        num_iterations (int): Number of times to run gradient descent
        print_cost (boolean): Flag to turn on/off printing of cost curve.

    Returns:
        parameters (dict): Trained weights of neural network in the form
                           {'Wi': (a_i, a_i-1), 'b': (a_i, 1)} for each layer i.
    """
    valid_methods = ('random', 'he', 'xavier')
    assert wt_init_method in valid_methods, "Invalid method entered. \
          Parameter initialization fail!"

    parameters = initialize_parameters(nw_archi, wt_init_method)
    costs = []  # List to store cost for each gradient descent step

    for i in range(num_iterations):
        AL, caches = fw_prop(X, parameters)
        cost = compute_cost(Y, AL)
        gradients = full_backprop(Y, AL, caches)
        parameters = update_params(parameters, gradients, learning_rate)

        if print cost and i % 100==0:
            costs.append(cost)

    # TODO: Implement print cost curve to check cost montonically decreasing

    return parameters


def nn_predict(X, parameters):
    """
    Predict the output labels given input features X

    Args:
        X (np.array): (n_0, m) shaped input features
        parameters (dict): Weights of trained neural network in the form
                           {'Wi': (a_i, a_i-1), 'Bi': (a_i, 1)} for each layer.
    Returns:
        pred (np.array): (1, m) shaped binary output prediction at threshold 0.5
    """
    AL = fw_prop(X, parameters)  # (1, m) log probability
    pred = AL.copy()
    pred[pred<0.5] = 0
    pred[pred>=0.5] = 1

    return pred


def compute_accuracy(actuals, predictions):
    """
    Compute the classification accuracy. Both actuals and predictions must
    have shape (n, m), where n is result (can be one-hot encoding or not) and
    m is number of examples

    Args:
        actuals (n,m array): 2d array containing the correct labels
                             (can be numeric or one-hot encoded)
        predictions (n, m array): 2d array containing the predicted labels
                                  (can be numeric or one-hot encoded)

    Returns:
        results (float): accuracy of classification prediction
    """

    assert actuals.shape == predictions.shape, "shapes of actuals \
                                                and predictions are not equal"
    assert len(predictions.shape) == 2, "shape length of predictions is not 2"

    classification_results = (actuals==predictions).all(axis=0)
    results = np.count_nonzero(classification_results) / actuals.shape[1]
    return results
