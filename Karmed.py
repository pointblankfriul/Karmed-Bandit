import os
import numpy as np
import Leverage
from abc import abstractmethod
import copy
import math

class Karmed(object):

    def __init__(self, arms, mus, sigmas, initial_reward_values):
        '''
        Define the number of Leverages you want, their mus (numpy array), their sigmas (same)
        and the initial values of the rewards
        '''
        self.estimated_rewards = np.zeros(arms)
        self.no_selections = np.zeros(arms)
        self.leverages = []
        for i in range(arms):
            self.leverages.append(Leverage.Leverage(mus[i], sigmas[i]))
            self.estimated_rewards[i] = initial_reward_values

        self.iter_per_run = 0
        self.epsilon = 0
        self.iter_values = []
        

    def setTrainParams(self, iter_per_run, epsilon):
        '''
        Define number of training runs (epochs) and number of iterations for run
        '''
        self.iter_per_run = iter_per_run
        self.epsilon = epsilon
        self.iter_values = np.zeros(iter_per_run)


    @abstractmethod
    def train(self):
        raise NotImplementedError("Please Implement this method")
        

class KarmedStationary(Karmed):

    def __init__(self, arms, mus, sigmas, initial_reward_values, ucb_params):
        '''
        Define the number of Leverages you want, their mus (numpy array), their sigmas (same)
        and the initial values of the rewards
        
        ucb_params = (ucb_activation, c)
                    - ucb_activation: bool --  Upper-Confidence Method. True to use
                    - c: float > 0 -- high values increase exploration
        '''
        self.ucb_as = ucb_params[0]
        self.c = ucb_params[1]
        Karmed.__init__(self, arms, mus, sigmas, initial_reward_values)


    def ucbSelection(self, estimates, t, act_selections):
        ucb = []
        for i in range(estimates):
            ucb.append(estimates[i] + self.c*math.sqrt(math.log(t)/act_selections[i]))
        return np.argmax(ucb)
    
    
    def train(self):
        for iteration in range(self.iter_per_run):
            if (np.random.uniform(0, 1) <= self.epsilon):
                # Exploring
                if self.ucb_as == True:
                    index = self.ucbSelection(self.estimated_rewards, iteration, self.no_selections)
                else:
                    index = np.random.randint(0, self.estimated_rewards.shape[0])
            else:
                # Exploiting
                temp = max(self.estimated_rewards)
                index = [i for i, j in enumerate(self.estimated_rewards) if j == temp]
                if len(index) > 1:
                    index = index[np.random.randint(0, len(index))]
                else:
                    index = index[0]
            
            self.no_selections[index] += 1
            self.iter_values[iteration] = self.leverages[index].pull()
            self.estimated_rewards[index] += 1/self.no_selections[index]*(self.iter_values[iteration] - self.estimated_rewards[index])
        
        return self.iter_values, self.estimated_rewards

    
class KarmedNonStationary(Karmed):

    def __init__(self, arms, mus, sigmas, initial_reward_values, alpha):
        '''
        Define the number of Leverages you want, their mus (numpy array), their sigmas (same),
        the initial values of the rewards and the alpha decay parameter
        '''
        self.alpha = alpha
        self.rewardsMat = [[] for i in range(arms)]
        Karmed.__init__(self, arms, mus, sigmas, initial_reward_values)


    def setRandomNoise(self, mu, sigma):
        self.noise = Leverage.Leverage(mu, sigma)


    def train(self):
        init_estimates = copy.deepcopy(self.estimated_rewards)
        for iteration in range(self.iter_per_run):
            if (np.random.uniform(0, 1) <= self.epsilon):
                # Exploring
                index = np.random.randint(0, self.estimated_rewards.shape[0])
            else:
                # Exploiting
                temp = max(self.estimated_rewards)
                index = [i for i, j in enumerate(self.estimated_rewards) if j == temp]
                if len(index) > 1:
                    index = index[np.random.randint(0, len(index))]
                else:
                    index = index[0]

            # update number of leverage selection
            self.no_selections[index] += 1
            # get reward
            self.iter_values[iteration] = self.leverages[index].pull()
            # append reward for estimates update and update formula
            self.rewardsMat[index].append(self.iter_values[iteration])
            exp_decay_sum = 0
            i = 1
            for reward in self.rewardsMat[index]:
                exp_decay_sum += self.alpha*((1 - self.alpha)**(self.no_selections[index] - i))*reward
                i += 1
                
            self.estimated_rewards[index] = ((1-self.alpha)**self.no_selections[index])*init_estimates[index] + exp_decay_sum

            # Add gaussian noise to all leverages
            for leverage in self.leverages:
                orig_mu = leverage.getMu() 
                noise = self.noise.pull()
                leverage.setMu(orig_mu+noise)
        
        return self.iter_values, self.estimated_rewards, self.leverages


                        




