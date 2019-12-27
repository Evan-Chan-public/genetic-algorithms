#Import packages
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import to_categorical
#Define variables
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')[:1000] / 255
x_test = x_test.reshape(10000, 784).astype('float32')[:1000] / 255
y_train = to_categorical(y_train, 10)[:1000]
y_test = to_categorical(y_test, 10)[:1000]
classes = 10
batch_size = 64
population = 100
generations = 100
threshold = 0.995
#Create the model and serve it variables
def serve_model(epochs, units1, act1, units2, act2, classes, act3, loss, opt, xtrain, ytrain, summary=False):
    model = Sequential()
    model.add(Dense(units1, input_shape = [784, ]))
    model.add(Activation(act1))
    model.add(Dense(units2))
    model.add(Activation(act2))
    model.add(Dense(classes))
    model.add(Activation(act3))
    model.compile(loss = loss, optimizer = opt, metrics = ['acc'])
    if summary:
        model.summary()
    model.fit(xtrain, ytrain, batch_size = batch_size, epochs = epochs, verbose = 0)
    return model
#Randomly generate a set of hyper-parameters
class Network():
    def __init__(self):
        self._epochs = np.random.randint(1, 15)
        self._units1 = np.random.randint(1, 500)
        self._units2 = np.random.randint(1, 500)
        self._act1 = random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'])
        self._act2 = random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'])
        self._act3 = random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'])
        self._loss = random.choice([
            'categorical_crossentropy',
            'binary_crossentropy',
            'mean_squared_error',
            'mean_absolute_error',
            'sparse_categorical_crossentropy'
        ])
        self._opt = random.choice(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])
        self._accuracy = 0
#Initialize hyperparams
    def init_hyperparams(self):
        hyperparams = {
            'epochs': self._epochs,
            'units1': self._units1,
            'act1': self._act1,
            'units2': self._units2,
            'act2': self._act2,
            'act3': self._act3,
            'loss': self._loss,
            'optimizer': self._opt
        }
        return hyperparams
#Initialize the population of networks
def init_networks(population):
    return [Network() for _ in range(population)]
#Calculates the accuracy of the model with the random hyper-parameters
def fitness(networks):
    for network in networks:
        hyperparams = network.init_hyperparams()
        epochs = hyperparams['epochs']
        units1 = hyperparams['units1']
        act1 = hyperparams['act1']
        units2 = hyperparams['units2']
        act2 = hyperparams['act2']
        act3 = hyperparams['act3']
        loss = hyperparams['loss']
        opt = hyperparams['optimizer']
        #Some model architectures break the code due to dimensionality issues or other errors
        try:
            model = serve_model(epochs, units1, act1, units2, act2, classes, act3, loss, opt, x_train, y_train)
            accuracy = model.evaluate(x_test, y_test, verbose = 0)[1]
            network._accuracy = accuracy
            print ('Accuracy: {}'.format(network._accuracy))
        except:
            network._accuracy = 0
            print ('Build failed.')
    return networks
#Select only the best 20% of networks for next gen
def selection(networks):
    networks = sorted(networks, key = lambda network: network._accuracy, reverse = True)
    networks = networks[:int(0.2 * len(networks))]
    return networks
#hyper-parameters of the parents are split between the two child agents created, and some hidden units are assigned
def crossover(networks):
    offspring = []
    for _ in range(int((population - len(networks)) / 2)):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        child1 = Network()
        child2 = Network()
        # Crossing over parent hyper-params
        child1._epochs = int(parent1._epochs / 4) + int(parent2._epochs / 2)
        child2._epochs = int(parent1._epochs / 2) + int(parent2._epochs / 4)
        child1._units1 = int(parent1._units1 / 4) + int(parent2._units1 / 2)
        child2._units1 = int(parent1._units1 / 2) + int(parent2._units1 / 4)
        child1._units2 = int(parent1._units2 / 4) + int(parent2._units2 / 2)
        child2._units2 = int(parent1._units2 / 2) + int(parent2._units2 / 4)
        child1._act1 = parent2._act2
        child2._act1 = parent1._act2
        child1._act2 = parent2._act1
        child2._act2 = parent1._act1
        child1._act3 = parent2._act2
        child2._act3 = parent1._act2
        offspring.append(child1)
        offspring.append(child2)
    networks.extend(offspring)
    return networks
#Less than 10% chance for mutation
def mutate(networks):
    for network in networks:
        if np.random.uniform(0, 1) <= 0.1:
            network._epochs += np.random.randint(0, 100)
            network._units1 += np.random.randint(0, 100)
            network._units2 += np.random.randint(0, 100)
    return networks
#Spawns agents into the environment and performs functions on them
def main():
    networks = init_networks(population)
    print(population)
    for gen in range(generations):
        print('Generation {}'.format(gen + 1))
        networks = fitness(networks)
        networks = selection(networks)
        networks = crossover(networks)
        networks = mutate(networks)
        #Look for more than 99.5% accuracy to stop
        for network in networks:
            if network._accuracy > threshold:
                print ('Threshold met')
                print (network.init_hyperparams())
                print ('Best accuracy: {}'.format(network._accuracy))
                exit(0)
#Execute
if __name__ == '__main__':
    main()
