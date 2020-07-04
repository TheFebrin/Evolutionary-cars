import numpy as np
import matplotlib.pyplot as plt
import torch
from neural_network import nn
import random
from tqdm import tqdm


class GA:
    def __init__(
        self,
        n_sensors,
        population_size,
        chromosome_len,
        K=0.9, tau0=None, tau1=None,
        n_samples=1000, n_epochs=200, l_rate=0.01,
        evolve=True
    ):
        '''
        Args:
            n_sensors : int
                Number of sensors in a car

            population_size : int
                Number of cars in the population

            chromosome_len : int
                Length of the list, which is our answer for the problem

            K : int
                Kind of learning rate parameter

            n_samples : int
                The number of samples we use while training a child neural network during NN crossover

            n_epochs : int
                The number of epochs while training a child neural network during NN crossover

            l_rate : float
                Learning rate while training a child neural network during NN crossover

            evolve : bool
                if False population doesn't change during the generations
        '''

        self.n_sensors = n_sensors
        self.population_size = population_size
        self.d = chromosome_len
        self.K = K
        self.tau0 = tau0
        self.tau1 = tau1
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.l_rate = l_rate
        self.evolve = evolve

        if self.tau0 is None:
            self.tau0 = K / np.sqrt(2 * np.sqrt(self.d))

        if self.tau1 is None:
            self.tau1 = K / np.sqrt(2 * self.d)

        self.population = np.random.uniform(
            low=0,
            high=1,
            size=(self.population_size, self.d)
        )
        self.sigmas = np.random.uniform(
            low=0,
            high=0.05,
            size=(self.population_size, self.d)
        )
        self.training_data = np.random.uniform(
            low=0,
            high=1,
            size=(self.population_size, self.d)
        )

        self.cost = np.ones(self.population_size)
        self.cost_history = []
        self.population_history = []
        self.sigmas_history = []
        self.best_sigmas_history = []

    def parents_selection(self):
        fitness_values = self.cost
        fitness_values = fitness_values - fitness_values.min()
        if fitness_values.sum() > 0:
            fitness_values = fitness_values / fitness_values.sum()
        else:
            fitness_values = np.ones(len(self.population)) / len(self.population)

        ids = np.random.choice(
            np.arange(self.population_size),
            size=self.population_size,
            replace=True,
            p=fitness_values
        )
        return ids

    def crossover1(self, parent1_id, parent2_id, parents, parent_sigmas):
        parent1 = parents[parent1_id],
        parent2 = parents[parent2_id],

        '''
        # Creating two parent neural networks

        net1 = nn.Network(
            in_dim=self.n_sensors + 2,  # n_sensors + act_velocity + act angle
            h1=4,
            h2=3,
            out_dim=2
        )
        net2 = nn.Network(
            in_dim=self.n_sensors + 2,  # n_sensors + act_velocity + act angle
            h1=4,
            h2=3,
            out_dim=2
        )
        # init net1
        idx = 0
        with torch.no_grad():
            for name, p in net1.named_parameters():
                if 'weight' in name:
                    w_size = p.shape[0] * p.shape[1]
                    w = parent1[idx: idx + w_size]
                    p.copy_(torch.from_numpy(w).view(p.shape))
                elif 'bias' in name:
                    w_size = p.shape[0]
                    w = parent1[idx: idx + w_size]
                    p.copy_(torch.from_numpy(w))
                else:
                    raise ValueError('Unknown parameter name "%s"' % name)
                idx += w_size

        # init net2
        idx = 0
        with torch.no_grad():
            for name, p in net2.named_parameters():
                if 'weight' in name:
                    w_size = p.shape[0] * p.shape[1]
                    w = parent2[idx: idx + w_size]
                    p.copy_(torch.from_numpy(w).view(p.shape))
                elif 'bias' in name:
                    w_size = p.shape[0]
                    w = parent2[idx: idx + w_size]
                    p.copy_(torch.from_numpy(w))
                else:
                    raise ValueError('Unknown parameter name "%s"' % name)
                idx += w_size

        '''
        child_net = nn.Network(
            in_dim=self.n_sensors + 2,  # n_sensors + act_velocity + act angle
            h1=4,
            h2=3,
            out_dim=2
        )

        self.n_samples = len(self.training_data)

        if len(self.training_data[parent1_id]) == 0 or len(self.training_data[parent1_id]) == 0:
            return parent1

        indices1 = np.random.choice(
            np.arange(len(self.training_data[parent1_id])),
            self.n_samples
        )
        indices2 = np.random.choice(
            np.arange(len(self.training_data[parent2_id])),
            self.n_samples
        )

        X_train1 = np.array(self.training_data[parent1_id][indices1])[:, 0]
        X_train2 = np.array(self.training_data[parent2_id][indices2])[:, 0]
        X_train2 = np.concatenate(X_train2, axis=0).reshape(self.n_samples, -1)
        X_train1 = np.concatenate(X_train1, axis=0).reshape(self.n_samples, -1)

        X_train1, X_train2 = torch.Tensor(X_train1), torch.Tensor(X_train2)

        # In case of using parent neural nets
        # y_train1 = net1(X_train1.float()).detach().numpy()
        # y_train2 = net1(X_train2.float()).detach().numpy()

        y_train1 = np.array(self.training_data[parent1_id][indices1])[:, 1]
        y_train1 = np.concatenate(y_train1, axis=0).reshape(self.n_samples, -1)

        y_train2 = np.array(self.training_data[parent2_id][indices2])[:, 1]
        y_train2 = np.concatenate(y_train2, axis=0).reshape(self.n_samples, -1)

        X_train = torch.Tensor(np.vstack((X_train1, X_train2)))
        y_train = torch.Tensor(np.vstack((y_train1, y_train2)))

        assert(X_train.shape[0] == y_train.shape[0])
        optimizer = torch.optim.Adam(child_net.parameters(), lr=self.l_rate)

        def loss_fun(x, x_hat):
            loss = torch.sum((x - x_hat) ** 2)
            return loss / x.size(0)

        child_net.train()
        for i in range(self.n_epochs):
            x_hat = child_net(X_train)
            loss = loss_fun(y_train, x_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f'Step: {i + 1} / {self.n_epochs} | Loss: {loss}')
        child_net.eval()

        # child_preds = child_net(X_train.float()).detach().numpy()
        # error = np.sum((child_preds - y_train.numpy()) ** 2) / len(y_train)
        # print(f'\nError: {error}\n')

        all_params = []
        with torch.no_grad():
            for name, p in child_net.named_parameters():
                all_params += list(p.detach().numpy().ravel())

        return np.array(all_params)

    def crossover2(self, parent1_id, parent2_id, parents, parent_sigmas):
        parent1 = parents[parent1_id]
        parent2 = parents[parent2_id]
        parent1_sigma = parent_sigmas[parent1_id]
        parent2_sigma = parent_sigmas[parent2_id]

        child1, child2 = [], []
        child1_sigma, child2_sigma = [], []
        for i in range(len(parent1)):
            coin_toss = np.random.randint(0, 2)
            if coin_toss % 2 == 0:
                child1.append(parent1[i])
                child2.append(parent2[i])
                child1_sigma.append(parent1_sigma[i])
                child2_sigma.append(parent2_sigma[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
                child1_sigma.append(parent2_sigma[i])
                child2_sigma.append(parent1_sigma[i])
        return child1, child2, child1_sigma, child2_sigma

    def mutation1(self):
        ''' ES algorithm based mutation '''
        X = self.population
        Sigmas = self.sigmas

        E = np.random.normal(0, self.tau1, size=Sigmas.shape)
        eps_o = np.random.normal(0, self.tau0)
        Sigmas *= np.exp(E + eps_o)

        self.population = X + np.random.normal(0, 1, size=Sigmas.shape) * Sigmas
        self.sigmas = Sigmas

    def mutation2(self):
        ''' Adding Gaussian Noise '''
        self.population += np.random.normal(0, 0.1, size=self.population.shape)

    def select_new_population(self, n_gen, crossover=1, mutation=2):
        '''
        Args:
            crossover : int
                1: crossover1, (learning third neural network)
                2: crossover2, children get random genes from parent
        '''
        if not self.evolve or n_gen == 1:
            return self.population

        ids = self.parents_selection()
        parents = self.population[ids]
        self.training_data = self.training_data[ids]
        parent_sigmas = self.sigmas[ids]

        assert(len(self.population) == len(parents) == self.population_size)

        children, children_sigmas = [], []
        desc = 'Creating offspring...'
        r = self.population_size if crossover == 1 \
            else self.population_size // 2

        for i in tqdm(range(r), position=0, leave=True, desc=desc):
            parents_ids = random.sample(range(len(parents)), 2)
            # Neural networks crossover
            if crossover == 1:
                child = self.crossover1(
                    parents_ids[0],
                    parents_ids[1],
                    parents,
                    parent_sigmas
                )
                children.append(child)
                child_sigmas = (
                    parent_sigmas[parents_ids[0]] + parent_sigmas[parents_ids[1]]
                ) / 2
                children_sigmas.append(child_sigmas)

            # Simple toin coss over genotypes
            elif crossover == 2:
                siblings = self.crossover2(
                    parents_ids[0],
                    parents_ids[1],
                    parents,
                    parent_sigmas
                )
                children.append(siblings[0])
                children.append(siblings[1])
                children_sigmas.append(siblings[2])
                children_sigmas.append(siblings[3])

            else:
                raise ValueError('Wrong crossover type!')

        children = np.array(children)
        self.population = children
        self.sigmas = np.array(children_sigmas)

        if mutation == 1:
            self.mutation1() # ES mutation
        elif mutation == 2:
            self.mutation2()  # adding noise
        else:
            raise ValueError('Wrong mutation type!')

        # self.population_history.append(self.population)
        self.cost_history.append(
            (self.cost.min(), self.cost.mean(), self.cost.max())
        )
        self.sigmas_history.append(
            self.sigmas.mean(axis=0)  # mean of sigmas in population
        )
        best_indi = self.cost.argmax()
        self.best_sigmas_history.append(
            self.population[best_indi]  # sigmas of best individual
        )
        return self.population

    def plot_cost(self):
        self.cost_history = np.array(self.cost_history)
        plt.figure(figsize=(15, 5))
        plt.plot(self.cost_history)
        maxi_id = self.cost_history[:, 0].argmax()
        maxi_val = self.cost_history[:, 0][maxi_id]
        plt.title(f'POPULATION SIZE: {self.population_size}  |  CHROMOSOME LEN: {self.d}  |  BEST_ITER: {maxi_id}  |  MAX: {maxi_val :.3f}')
        plt.legend(['Min', 'Mean', 'Max'], loc='upper right')
        plt.savefig('plots/cost.png')

    def plot_sigmas(self, sigmas, mode=''):
        plt.figure(figsize=(15, 5))
        plt.title('Sigmas')
        plt.plot(sigmas)
        plt.savefig(f'plots/sigma_{mode}.png')
