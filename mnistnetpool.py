#Import packages
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import to_categorical
# Define variables
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')[:1000] / 255
x_test = x_test.reshape(10000, 784).astype('float32')[:1000] / 255
y_train = to_categorical(y_train, 10)[:1000]
y_test = to_categorical(y_test, 10)[:1000]
classes = 10
batch_size = 64
population = 100
generations = 100
threshold = 0.95
# Create the model and serve it variables
def serve_model(epochs, units1, act1, units2, act2, classes, act3, lossfunction, opt, xtrain, ytrain, summary = False):
    model = Sequential()
    model.add(Dense(units1, input_shape = [784, 1]))
    model.add(Activation(act1))
    model.add(Dense(units2))
    model.add(Activation(act2))
    model.add(Dense(classes))
    model.add(Activation(act3))
    model.compile(loss = lossfunction, optimizer = opt, metrics = ['acc'])
    if summary:
        model.summary()
    model.fit(xtrain, ytrain, batch_size = batch_size, epochs = epochs)
    return model
# Randomly generate a set of hyper-parameters
class Network():
    def __init__(self):
        self.epochs = np.random.randint(1, 15)
        self.units1 = np.random.randint(1, 500)
        self.units2 = np.random.randint(1, 500)
        self.lossfunction = random.choice([
            'categorical_crossentropy',
            'binary_crossentropy',
            'mean_squared_error',
            'mean_absolute_error',
        ])
        self._act1 = random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'])
        self.act2 = random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'])
        self.act3 = random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'])
        self.opt = random.choice(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])
        self.accuracy = 0
# Initialize hyperparams
    def init_hyperparams(self):
        hyperparams = {
            'epochs': self.epochs,
            'units1': self.units1,
            'act1': self.act1,
            'units2': self.units2,
            'act2': self.act2,
            'act3': self.act3,
            'lossfunction': self.loss,
            'opt': self.opt
        }
        return hyperparams
# Initialize the population of networks
def init_networks(population):
    for _ in range(population):
        return Network()
# Calculates the accuracy of the model with the random hyper-parameters
def fitness(networks):
    for network in networks:
        hyperparams = network.init_hyperparams()
        epochs = hyperparams['epochs']
        units1 = hyperparams['units1']
        act1 = hyperparams['act1']
        units2 = hyperparams['units2']
        act2 = hyperparams['act2']
        act3 = hyperparams['act3']
        lossfunction = hyperparams['lossfunction']
        opt = hyperparams['opt']
        # Some model architectures break the code due to dimensionality issues or other errors
        try:
            model = serve_model(epochs, units1, act1, units2, act2, classes, act3, lossfunction, opt, x_train, y_train)
            network.accuracy = model.evaluate(x_test, y_test, verbose = 0)[1]
            print ('Accuracy: {}'.format(network.accuracy))
        except:
            network.accuracy = 0
            print ('Build throws error.')
    return networks
# Select only the best 20% of networks for next gen
def selection(networks):
    networks = sorted(networks, key = lambda network: network.accuracy, reverse = True)

    return networks
# Hyper-parameters of the parents are split between the two child agents created, and some hidden units are assigned
def crossover(networks):
    for _ in range(int((population - len(networks)) / 2)):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        # No identical parents - will duplicate
        while parent1 == parent2:
            parent1 = random.choice(networks)
        child1 = Network()
        child2 = Network()
        # Crossing over parent hyper-params
        child1.epochs = int(parent1.epochs / 4) + int(parent2.epochs / 2)
        child2.epochs = int(parent1.epochs / 2) + int(parent2.epochs / 4)
        child1.units1 = int(parent1.units1 / 4) + int(parent2.units1 / 2)
        child2.units1 = int(parent1.units1 / 2) + int(parent2.units1 / 4)
        child1.units2 = int(parent1.units2 / 4) + int(parent2.units2 / 2)
        child2.units2 = int(parent1.units2 / 2) + int(parent2.units2 / 4)
        child1.act1 = random.choice([parent1.act1, parent1.act2, parent1.act3])
        child2.act1 = random.choice([parent2.act1, parent2.act2, parent2.act3])
        child1.act2 = random.choice([parent2.act1, parent2.act2, parent2.act3])
        child2.act2 = random.choice([parent1.act1, parent1.act2, parent1.act3])
        child1.act3 = random.choice([parent1.act1, parent1.act2, parent1.act3])
        child2.act3 = random.choice([parent2.act1, parent2.act2, parent2.act3])
        networks.append(child1)
        networks.append(child2)
    return networks
# Less than 10% chance for mutation
def mutate(networks):
    for network in networks:
        if np.random.uniform(0, 1) <= 0.1:
            network.epochs += np.random.randint(-5, 5)
            network.units1 += np.random.randint(-100, 100)
            network.units2 += np.random.randint(-100, 100)
    return networks
# Spawns agents into the environment and performs functions on them
networks = init_networks(population)
print(population)
for i in range(generations):
    print('Generation' + (i + 1))
    networks = fitness(networks)
    networks = selection(networks)
    networks = crossover(networks)
    networks = mutate(networks)
    # Look for more than 99.5% accuracy to stop
    for network in networks:
        if network.accuracy > threshold:
            print('Threshold met')
            print(network.init_hyperparams())
            print(model.summary())
            print('Best accuracy: {}'.format(network.accuracy))
            exit(0)
