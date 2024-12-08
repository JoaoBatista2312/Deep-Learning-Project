#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        y_hat = self.predict(x_i)
        
        if y_i != y_hat:
            self.W[y_i, :] += x_i
            self.W[y_hat, :] -= x_i
        
        #raise NotImplementedError # Q1.1 (a)


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """     
        y_pred = self.predict(x_i)
        #x_i = x_i.reshape((-1,1))

        #y_hat = y = np.zeros((len(self.W),1))
        #y[y_i,:] = 1
        #y_hat[y_pred,:] = 1

        #dL_dW = np.matmul((y_hat - y), x_i.T)
        # print(dL_dW[y_i,:])
        if y_i != y_pred:
            self.W[y_i] = (1-learning_rate*l2_penalty)*self.W[y_i] + learning_rate * x_i
            self.W[y_pred] = (1-learning_rate*l2_penalty)*self.W[y_pred] - learning_rate * x_i
        #raise NotImplementedError # Q1.2 (a,b)

class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.W = [np.random.normal(0.1, 0.1**2, size=(hidden_size,n_features)),
                  np.random.normal(0.1, 0.1**2, size=(n_classes,hidden_size))]
        
        self.B = [np.zeros((hidden_size,1)),
                  np.zeros((n_classes,1))]
        # raise NotImplementedError # Q1.3 (a)

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.
        y_hat = []
        for i in range(len(X)):
            y_temp, _ = self.__forward__(X[i],self.W,self.B)
            y_hat.append(y_temp)
        y_hat = np.array(y_hat)
        
        return np.argmax(self.__softmax__(y_hat),axis=1)
        raise NotImplementedError # Q1.3 (a)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, Y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """
        # print("X: ", np.shape(X))
        num_layers = len(self.W)
        total_loss = 0
        # For each observation and target
        for x, y in zip(X, Y):
            # Comoute forward pass
            output, hiddens = self.__forward__(x, self.W, self.B)
            output_temp = np.argmax(self.__softmax__(output), axis=0)
           
            # Compute Loss and Update total loss
            loss = self.__multinomial_logistic_loss__(output_temp, y)
            total_loss += loss
            # Compute backpropagation
            grad_weights, grad_biases = self.__backward__(x, y, output, hiddens, self.W)
            
            # Update weights
           
            for i in range(num_layers):
                self.W[i] -= learning_rate*grad_weights[i]
                self.B[i] -= learning_rate*grad_biases[i]
                
        return total_loss
        #raise NotImplementedError # Q1.3 (a)
    
    def __forward__(self, x, weights, biases):
        num_layers = len(weights)
        g = np.tanh
        hiddens = []
        # compute hidden layers
        for i in range(num_layers):
            h = x[:, None] if i == 0 else hiddens[i-1]
            z = weights[i].dot(h) + biases[i]
    
            if i < num_layers-1:  # Assuming the output layer has no activation.
                hiddens.append(g(z))
        #compute output
        output = z
        output = output.reshape(-1) #To get rid of the (x,1)
        return output, hiddens

    def __backward__(self,x, y, output, hiddens, weights):
        num_layers = len(weights)
        y_OHE = np.zeros(np.shape(output))
        y_OHE[y] = 1

        probs = self.__softmax__(output)
        grad_z = (probs - y_OHE)[:,None]
        
        grad_weights = []
        grad_biases = []
        
        # Backpropagate gradient computations 
        for i in range(num_layers-1, -1, -1):
            # Gradient of hidden parameters.
            h = x[:, None] if i == 0 else hiddens[i-1]
            #print(f'h:{h}')
            #print(f'grad_z:{grad_z}')

            grad_weights.append(grad_z.dot(h.T))
            #print(f'grad_weights shape:{grad_weights[-1]}')
            grad_biases.append(grad_z)
            
            # Gradient of hidden layer below.
            grad_h = weights[i].T.dot(grad_z)

            # Gradient of hidden layer below before activation.
            grad_z = grad_h * np.maximum(0, h)   # Grad of loss wrt z3.

        # Making gradient vectors have the correct order
        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases

    #!Check how this works?
    def __multinomial_logistic_loss__(self,y_pred, y_true):
        """Multinomial logistic loss."""

        log_likelihood = -np.log(y_pred + 1e-15) * y_true
        loss = np.sum(log_likelihood) 
        return loss

    def __softmax__(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

        


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
