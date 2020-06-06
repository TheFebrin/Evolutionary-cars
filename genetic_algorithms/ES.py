import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class ES:
    def __init__(
        self,
        mu, lambda_,
        chromosome_len,
        K=0.6, tau0=None, tau1=None
    ):
        self.mu = mu
        self.lambda_ = lambda_
        self.d = chromosome_len
        self.K = K
        self.tau0 = tau0
        self.tau1 = tau1

        if self.tau0 is None:
            self.tau0 = K / np.sqrt(2 * np.sqrt(self.d))

        if self.tau1 is None:
            self.tau1 = K / np.sqrt(2 * self.d)

        self.P = np.hstack((
            np.random.uniform(0, 1, size=(self.mu, self.d)),
            np.random.uniform(0, 0.1, size=(self.mu, self.d)),
        ))

        self.cost = np.ones(self.mu)
        self.cost_history = []
        self.population_history = []
        self.sigmas_history = []
        self.best_sigmas_history = []

    def parents_selection(self):
        ids = np.argsort(self.cost)[::-1]
        best_ids = ids[:self.lambda_]
        return self.P[best_ids], best_ids

    def mutation(self, parents):
        X = parents[:, :self.d]
        Sigmas = parents[:, self.d:]

        E = np.random.normal(0, self.tau1, size=Sigmas.shape)
        eps_o = np.random.normal(0, self.tau0)
        Sigmas *= np.exp(E + eps_o)

        return np.hstack((
            X + np.random.normal(0, 1, size=Sigmas.shape) * Sigmas,
            Sigmas
        ))

    def select_new_population(self):
        children, ids = self.parents_selection()

        remaining_ids = np.random.choice(
            np.arange(len(ids)),
            size=self.mu,
            replace=True
        )
        children = children[remaining_ids]
        children = self.mutation(children)
        self.P = children

        self.cost_history.append(
            (self.cost.min(), self.cost.mean(), self.cost.max()))
        self.population_history.append(self.P[:, :self.d])
        self.sigmas_history.append(self.P[:, self.d:].mean(
            axis=0))  # mean of sigmas in population

        best_indi = self.cost.argmax()
        self.best_sigmas_history.append(
            self.P[:, self.d:][best_indi])  # sigmas of best indi

        return self.P[:, :self.d], ids

    def plot_cost(self):
        self.cost_history = np.array(self.cost_history)
        plt.figure(figsize=(15, 5))
        plt.plot(self.cost_history)
        mini_id = self.cost_history[:, 0].argmax()
        mini_val = self.cost_history[:, 0][mini_id]
        desc = 'ES(µ + λ)'
        plt.title(f'{desc} --> POPULATION SIZE: {self.mu}  |  CHROMOSOME LEN: {self.d}  |  BEST_ITER: {mini_id}  |  MIN: {mini_val :.3f}')
        plt.legend(['Min', 'Mean', 'Max'], loc='upper right')
        plt.savefig('plots/cost.png')

    def plot_sigmas(self, sigmas, mode=''):
        plt.figure(figsize=(15, 5))
        plt.title('Sigmas')
        plt.plot(sigmas)
        plt.savefig(f'plots/sigma_{mode}.png')
