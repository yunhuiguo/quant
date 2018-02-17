import numpy as np
import matplotlib.pyplot as plt
import math
import time

import sys
sys.path.append('./python-mnist/')
from mnist import MNIST

def binarize(weight, ave, bina):
    neg = weight < ave
    pos = weight > ave
    bina[neg] = -1
    bina[pos] = 1

def ReLU(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return np.exp(x)/(np.exp(x)+1.0)

def trick_sigmoid(x):
    return 1.7159*np.tanh(2*x/3)

def trick_dsigmoid(x):
    return 1.7159*(1-np.power(np.tanh(2*x/3),2))*2/3

def dReLU(x):
    x[x >0] = 1
    x[x <= 0] = 0
    return x

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def Leaky_Relu(x, leakage = 0.01):
    a = np.copy(x)
    a[ a < 0 ] *= leakage
    return a

def Leaky_dRelu(x, leakage = 0.01):
    return np.clip(x > 0, leakage, 1.0)

def softmax(x):
    # Find the largest a, and subtract it from each a in order to prevent overflow
    x_max = np.max(x,1).reshape(x.shape[0],1)
    sum_exp_x = np.sum(np.exp(x - x_max),1).reshape(x.shape[0],1) 
    pred_y = np.exp(x - x_max) / (sum_exp_x+0.0) 
    return pred_y

def random_init_weights(input_size, output_size):
    return 0.01 * np.random.randn(input_size, output_size)

def zero_init_bias(output_size):
    return  np.zeros((1, output_size))

def random_init_weights_fan_in(input_size, output_size):
    return np.random.normal(0,np.power(input_size,-0.5),(input_size,output_size))

def Xavier_initializtion(input_size, output_size):
    var = 2.0 / (input_size + output_size)
    stddev = math.sqrt(var)
    return np.random.normal(0.0, stddev, (input_size, output_size))

def random_init_bias(output_size):
    return np.random.randn(1, output_size)

def zero_init_delta_w(input_size, output_size):
    return np.zeros((input_size,output_size))

class Network():

    def __init__(self, layers, init_method_weights = random_init_weights, init_method_bias = random_init_bias, init_method_delta_w = zero_init_delta_w, activation_fn = "sigmoid", 
        learning_rate = 0.01, momentum = 0.9, epoches = 60, batch_size = 128, nesterov_momentum = 0):
        self.layers = layers
        self.init_method_weights = init_method_weights
        self.init_method_bias = init_method_bias
        self.init_method_delta_w = init_method_delta_w
        self.nesterov_momentum = nesterov_momentum
        self.momentum = momentum
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.setup_layers()

        if activation_fn == "sigmoid":
            self.activation_fn = sigmoid
            self.activation_dfn = dsigmoid
        elif activation_fn == "ReLU":
            self.activation_fn = ReLU
            self.activation_dfn = dReLU
        elif activation_fn == "trick_sigmoid":
            self.activation_fn = trick_sigmoid
            self.activation_dfn = trick_dsigmoid
        elif activation_fn == "Leaky_Relu":
            self.activation_fn = Leaky_Relu
            self.activation_dfn = Leaky_dRelu

        self.validation_loss = None
        self.best_validation_weights = None
        self.best_validation_biases = None

    def setup_layers(self):
        self.w = [ self.init_method_weights(input_size, output_size) for input_size, output_size in zip(self.layers[:-1], self.layers[1:])]
        self.b = [ self.init_method_bias(output_size) for output_size in self.layers[1:]]

        self.binary_w = [ self.init_method_weights(input_size, output_size) for input_size, output_size in zip(self.layers[:-1], self.layers[1:])]

        # for momentum
        self.delta_w = [ self.init_method_delta_w(input_size, output_size) for input_size, output_size in zip(self.layers[:-1], self.layers[1:])]

        if self.nesterov_momentum == 1:
            # for nesterov momentum
            self.pre_delta_w = [ self.init_method_delta_w(input_size, output_size) for input_size, output_size in zip(self.layers[:-1], self.layers[1:])]

    def forward(self, x):
        for idx, (weight, bias) in enumerate(zip(self.w[:-1], self.b[:-1])):
            '''
            ave = sum(weight) / (weight.shape[0]+0.0)
            for i in range(weight.shape[1]):
                np.apply_along_axis(binarize, 0, weight[:,i], ave[i], self.binary_w[idx][:,i])
            '''
            pos = weight > 0
            neg = weight < 0

            self.binary_w[idx][pos] = 1
            self.binary_w[idx][neg] = -1

            x = self.activation_fn(np.matmul(x, self.binary_w[idx]) + bias)

        pred_y = softmax(np.matmul(x, self.w[-1]) + self.b[-1])
        return pred_y

    def get_activations(self, x):
        activation = x
        activations = [activation] 
        pre_activations = []

        for idx, (weight, bias) in enumerate(zip(self.w[:-1], self.b[:-1])):
            '''
            ave = sum(weight) / (weight.shape[0]+0.0)
            for i in range(weight.shape[1]):
                np.apply_along_axis(binarize, 0, weight[:,i], ave[i], self.binary_w[idx][:,i])
            '''
            pos = weight > 0
            neg = weight < 0

            self.binary_w[idx][pos] = 1
            self.binary_w[idx][neg] = -1

            pre_activation = np.matmul(activation, self.binary_w[idx]) + bias
            pre_activations.append(pre_activation)
            activation = self.activation_fn(pre_activation)
            activations.append(activation)

        pre_activation = np.matmul(activation, self.w[-1]) + self.b[-1]    
        pre_activations.append(pre_activation)    
        activation = softmax(pre_activation)
        activations.append(activation)

        return activations, pre_activations                           

    def momentum_update(self, gradient, delta_w_):
        delta_w_ = self.learning_rate * gradient / (self.batch_size+0.0) + self.momentum * delta_w_ #delta_w has same dimension as w
        return delta_w_  

    def nesterov_momentum_update(self, gradient, delta_w_, pre_delta_w_):
        pre_delta_w_ = delta_w_
        delta_w_ = self.learning_rate * gradient / (self.batch_size+0.0) + self.momentum * delta_w_ 

        return delta_w_, pre_delta_w_ 

    def update_mini_batch(self, train_data_batch, train_label_batch):
        dw = [np.zeros(weight.shape) for weight in self.w]
        db = [np.zeros(bias.shape) for bias in self.b]

        for train_data, train_label in zip(train_data_batch, train_label_batch):
            dw_, db_ = self.backpropagation(train_data, train_label)
            dw = [dweight + dweight_ for dweight, dweight_ in zip(dw, dw_)]
            db = [dbias + dbias_ for dbias, dbias_ in zip(db, db_)]

            # For gradient check
            #self.gradient_check(dw, train_data.reshape(1, train_data.shape[0]), train_label.reshape(1, train_label.shape[0]))
            #self.bias_gradient_check(db, train_data.reshape(1, train_data.shape[0]), train_label.reshape(1, train_label.shape[0]))

        if self.nesterov_momentum == 1:
            #nesterov_momentum
            for idx, (weight, dw_, delta_w_, pre_delta_w_) in enumerate(zip(self.w, dw, self.delta_w, self.pre_delta_w)):
                self.delta_w[idx], self.pre_delta_w[idx] = self.nesterov_momentum_update(dw_, delta_w_, pre_delta_w_)
                weight = weight - self.momentum * self.pre_delta_w[idx] + (1 + self.momentum) * self.delta_w[idx]
                self.w[idx] = weight
        else:
            # momentum
            for idx, (weight, dw_, delta_w_) in enumerate(zip(self.w, dw, self.delta_w)):
                self.delta_w[idx] = self.momentum_update(dw_, delta_w_)
                weight = weight + self.delta_w[idx]
                self.w[idx] = weight

        self.b = [bias + self.learning_rate * db_ / (train_data_batch.shape[0]+0.0)  for bias, db_ in zip(self.b, db)]
        
    def backpropagation(self, train_data, train_label):
        train_data = train_data.reshape(1, train_data.shape[0])
        dw = [np.zeros(weight.shape) for weight in self.w ]
        db = [np.zeros(bias.shape) for bias in self.b ]

        activations, pre_activations = self.get_activations(train_data)
    
        delta = train_label - activations[-1]
        dw[-1] = np.matmul( activations[-2].transpose(), delta)

        # Backpropagation
        for idx in range(2, len(self.layers)):
            pre_activation = pre_activations[-idx]
            activation = activations[-idx-1]
            delta = self.activation_dfn(pre_activation) * np.matmul(delta, self.w[-idx+1].transpose())
            dw[-idx] = np.matmul( activation.transpose(), delta)
            db[-idx] = delta  
        return dw, db

    def loss(self, pred_y, one_hot_labels):
        pred_y[pred_y == 0.0] = 1e-15
        log_pred_y = np.log(pred_y)
        loss_ = -np.sum(one_hot_labels * log_pred_y) / (one_hot_labels.shape[0]+0.0)

        return loss_
 
    def accuracy(self, pred_y, labels):
        pred_class = np.argmax(pred_y, axis=1)
        accuracy_ = np.sum(pred_class == labels)/(pred_class.shape[0]+0.0)

        return accuracy_

    def train(self, training_images, one_hot_train_labels, training_labels, test_images, one_hot_test_labels, test_labels,
     validation_images, validation_labels, one_hot_validation_labels):

        batch_count = training_images.shape[0] / self.batch_size
        self.validation_loss = np.float64("inf")
        training_accuracy_all = []
        test_accuracy_all = []
        validation_accuracy_all = []
        training_loss_all = []
        test_loss_all = []
        validation_loss_all = []

        stop_training = False
        for epoch in range(self.epoches):
            # Shuffle data
            idxs = np.random.permutation(training_images.shape[0]) 
            X_random = training_images[idxs]
            Y_random = one_hot_train_labels[idxs]
            print "Epoch " + str(epoch)

            for i in range(int(batch_count)):
                train_data_batch = X_random[i * self.batch_size: (i+1) * self.batch_size, :]
                train_label_batch = Y_random[i * self.batch_size: (i+1) * self.batch_size, :]   

                # Update mini_batch
                self.update_mini_batch(train_data_batch, train_label_batch)
                '''
                #pred_y_train = self.forward(training_images)
                #pred_y_test = self.forward(test_images)
                #pred_y_validation = self.forward(validation_images)

                #loss_ = self.loss(pred_y_validation, one_hot_validation_labels)

                # Early stopping
                if loss_ <= self.validation_loss:
                    self.validaetion_loss = loss_
                    self.best_validation_weights = [np.array(weight) for weight in self.w]
                    self.best_validation_biases = [np.array(bias) for bias in self.b]

                    # Calculate accuracy and loss
                    #training_accuracy_all.append(self.accuracy(pred_y_train, training_labels))
                    #test_accuracy_all.append(self.accuracy(pred_y_test, test_labels))
                    #validation_accuracy_all.append(self.accuracy(pred_y_validation, validation_labels))

                    #training_loss_all.append(self.loss(pred_y_train, one_hot_train_labels))
                    #test_loss_all.append(self.loss(pred_y_test, one_hot_test_labels))
                    #validation_loss_all.append(self.loss(pred_y_validation, one_hot_validation_labels))
                else:
                    stop_training = True
                    break
            if stop_training == True:
                break
            '''
            pred_y_test = self.forward(test_images)
            print "Test accuracy is: " + str(self.accuracy(pred_y_test, test_labels))
            
        fig1 = plt.figure(1)
        plt.plot(training_accuracy_all,'r-')
        plt.plot(test_accuracy_all, 'b-')
        plt.plot(validation_accuracy_all, 'g-')

        plt.legend(['train accuracy', 'test accuracy', 'validation accuracy'], loc='lower right')
        plt.xlabel('Batches', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.title('Accuracy VS Batches', fontsize=15)
        fig1.show()

        fig2 = plt.figure(2)
        plt.plot(training_loss_all,'r-')
        plt.plot(test_loss_all, 'b-')
        plt.plot(validation_loss_all, 'g-')

        plt.legend(['train loss', 'test loss', 'validation loss_'], loc='lower right')
        plt.xlabel('Batches', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.title('Loss VS Batches', fontsize=15)
        fig2.show()           
        plt.show()
        
if __name__ == '__main__':        
    # Read datasets
    data = MNIST('./python-mnist/data')
    training_images, training_labels = data.load_training()
    test_images, test_labels = data.load_testing()

    training_images = np.array(training_images)
    test_images = np.array(test_images)
    training_labels = np.array(training_labels)
    test_labels = np.array(test_labels)

    # Normalize data
    training_images = training_images / 127.5 - 1
    test_images = test_images / 127.5 - 1

    classes = 10
    one_hot_train_labels = np.eye(classes)[training_labels] 
    one_hot_test_labels = np.eye(classes)[test_labels]  

    # Construct train data and validation data
    training_images, validation_images = training_images[0:50000,:], training_images[50000:,:]
    training_labels, validation_labels = training_labels[0:50000], training_labels[50000:]
    one_hot_train_labels, one_hot_validation_labels = one_hot_train_labels[0:50000,:], one_hot_train_labels[50000:,:]

    #training_images, validation_images = training_images[0:500,:], training_images[-500:,:]
    #training_labels, validation_labels = training_labels[0:500], training_labels[-500:]
    #one_hot_train_labels, one_hot_validation_labels = one_hot_train_labels[0:500,:], one_hot_train_labels[-500:,:]

    nn = Network([784, 64, 10])

    # Train network
    nn.train(training_images, one_hot_train_labels, training_labels, test_images, one_hot_test_labels, test_labels, validation_images, validation_labels, one_hot_validation_labels)


