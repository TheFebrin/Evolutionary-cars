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
        n_samples=10000, n_epochs=100, l_rate=0.1
    ):
        '''
        Args:
            n_sensors : int
                Desc
            population_size : int
                ...

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

        if self.tau0 is None:
            self.tau0 = K / np.sqrt(2 * np.sqrt(self.d))

        if self.tau1 is None:
            self.tau1 = K / np.sqrt(2 * self.d)

        self.population = np.random.uniform(0, 1, size=(self.population_size, self.d))
        self.sigmas = np.random.uniform(0, 0.05, size=(self.population_size, self.d))

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

    def crossover(self, parent1, parent2):
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

        # generate random data and learn a child net
        child_net = nn.Network(
            in_dim=self.n_sensors + 2,  # n_sensors + act_velocity + act angle
            h1=4,
            h2=3,
            out_dim=2
        )

        X_train1 = np.hstack((
            np.random.uniform(0, 800, size=(self.n_samples, self.n_sensors)),
            np.random.uniform(0, 10, size=(self.n_samples, 1)),    # velocity
            np.random.uniform(-90, 90, size=(self.n_samples, 1)),  # angle
        ))
        X_train2 = np.hstack((
            np.random.uniform(0, 800, size=(self.n_samples, self.n_sensors)),
            np.random.uniform(0, 10, size=(self.n_samples, 1)),    # velocity
            np.random.uniform(-90, 90, size=(self.n_samples, 1)),  # angle
        ))
        X_train1, X_train2 = torch.Tensor(X_train1), torch.Tensor(X_train2)

        y_train1 = net1(X_train1).detach().numpy()
        y_train2 = net1(X_train2).detach().numpy()

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

        # print(f'Step: {i + 1} / {self.n_epochs} | Loss: {loss}')

        all_params = []
        with torch.no_grad():
            for name, p in child_net.named_parameters():
                all_params += list(p.detach().numpy().ravel())

        return np.array(all_params)

    def crossover2(self, parent1, parent2):
        child1, child2 = [], []
        for i in range(len(parent1)):
            coin_toss = np.random.randint(0, 2)
            if coin_toss % 2 == 0:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2

    def mutation(self):
        X = self.population
        Sigmas = self.sigmas

        E = np.random.normal(0, self.tau1, size=Sigmas.shape)
        eps_o = np.random.normal(0, self.tau0)
        Sigmas *= np.exp(E + eps_o)

        self.population = X + np.random.normal(0, 1, size=Sigmas.shape) * Sigmas
        self.sigmas = Sigmas

    def mutation2(self):
        self.population += np.random.normal(0, 0.1, size=self.population.shape)

    def select_new_population(self):
        ids = self.parents_selection()
        parents = self.population[ids]
        # parent_sigmas = self.sigmas[ids]

        assert(len(self.population) == len(parents) == self.population_size)

        children, children_sigmas = [], []
        desc = 'Creating offspring...'
        for i in tqdm(range(self.population_size // 2), position=0, desc=desc):
            parents_ids = random.sample(range(len(parents)), 2)

            # Neural networks crossover
            # child = self.crossover(
            #     parents[parents_ids[0]],
            #     parents[parents_ids[1]]
            # )
            # children.append(child)
            # child_sigmas = (
            #     parent_sigmas[parents_ids[0]] + parent_sigmas[parents_ids[1]]
            # ) / 2
            # children_sigmas.append(child_sigmas)

            # Simple toin coss over genotypes
            siblings = self.crossover2(
                parents[parents_ids[0]],
                parents[parents_ids[1]]
            )
            children.append(siblings[0])
            children.append(siblings[1])

        children = np.array(children)
        self.population = children
        self.sigmas = np.array(children_sigmas)

        # self.mutation()
        self.mutation2()

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

        return self.population, set(ids)

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