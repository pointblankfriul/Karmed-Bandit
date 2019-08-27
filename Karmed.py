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
        self.act_selections = np.zeros(arms)
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


    def setRandomNoise(self, mu, sigma):
        self.noise = Leverage.Leverage(mu, sigma)

    
    def _stationaryEstimates(self, iteration, action_index):
        return 1/self.act_selections[action_index]*(self.iter_values[iteration] - self.estimated_rewards[action_index])

    
    def _nonStationaryEstimates(self, action_index, init_estimates):
        exp_decay_sum = 0
        i = 1
        for reward in self.rewardsMat[action_index]:
            exp_decay_sum += self.alpha*((1 - self.alpha)**(self.act_selections[action_index] - i))*reward
            i += 1
        return ((1-self.alpha)**self.act_selections[action_index])*init_estimates[action_index] + exp_decay_sum


    @abstractmethod
    def run(self):
        raise NotImplementedError("Please Implement this method")
        

class KarmedStationary(Karmed):

    def __init__(self, arms, mus, sigmas, initial_reward_values, ucb_params):
        '''
        Class to manage stationary problems with a K-Armed Bandit

        arms : int > 0 -- number of possible actions
        mus : float numpy -- arms mean values
        sigmas : float numpy -- arms sigma values
        initial_reward_values : float -- first iteration estimates
        ucb_params : tuple(ucb_activation, c)
                    - ucb_activation: bool --  Upper-Confidence Method. True to use
                    - c: float > 0 -- high values increase exploration
        '''
        self.ucb_as = ucb_params[0]
        self.c = ucb_params[1]
        Karmed.__init__(self, arms, mus, sigmas, initial_reward_values)


    def __ucbSelection(self, estimates, t, act_selections):
        ucb = []
        i = 0
        for estimate in estimates:
            if act_selections[i] == 0:
                return i
            ucb.append(estimate + self.c*math.sqrt(math.log(t)/act_selections[i]))
            i += 1
        return np.argmax(ucb)
    
    
    def run(self):
        for iteration in range(self.iter_per_run):
            if (np.random.uniform(0, 1) <= self.epsilon):
                # Exploring
                if self.ucb_as == True:
                    index = self.__ucbSelection(self.estimated_rewards, iteration, self.act_selections)
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
            
            self.act_selections[index] += 1
            self.iter_values[iteration] = self.leverages[index].pull()
            self.estimated_rewards[index] += self._stationaryEstimates(iteration, index)
        
        return self.iter_values, self.estimated_rewards

    
class KarmedNonStationary(Karmed):

    def __init__(self, arms, mus, sigmas, initial_reward_values, alpha):
        '''
        Class to manage non-stationary problems with a K-Armed Bandit

        arms : int > 0 -- number of possible actions
        mus : float numpy -- arms mean values
        sigmas : float numpy -- arms sigma values
        initial_reward_values : float -- first iteration estimates
        alpha : float > 0 -- step-size decay parameter

        Always set gaussian noise with setRandomNoise!
        '''
        self.alpha = alpha
        self.rewardsMat = [[] for i in range(arms)]
        Karmed.__init__(self, arms, mus, sigmas, initial_reward_values)


    def run(self):
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
            self.act_selections[index] += 1
            # get reward
            self.iter_values[iteration] = self.leverages[index].pull()
            # append reward for estimates update and update formula
            self.rewardsMat[index].append(self.iter_values[iteration])               
            self.estimated_rewards[index] = self._nonStationaryEstimates(index, init_estimates)

            # Add gaussian noise to all leverages
            for leverage in self.leverages:
                orig_mu = leverage.getMu()
                noise = self.noise.pull()
                leverage.setMu(orig_mu+noise)
        
        return self.iter_values, self.estimated_rewards, self.leverages


class KarmedGradient(Karmed):

    def __init__(self, arms, mus, sigmas, initial_reward_values, alpha, stationarity):
        '''
        Class to manage gradient-ascent based K-Armed Bandit

        arms : int > 0 -- number of possible actions
        mus : float numpy -- arms mean values
        sigmas : float numpy -- arms sigma values
        initial_reward_values : float -- first iteration estimates
        alpha : float > 0 -- step-size parameter
        stationarity : string ('stat', 'non_stat') -- determine problem type

        Call setRandomNoise if problem is non-stationary to set gaussian noise
        '''

        self.alpha = alpha
        self.rewardsMat = [[] for i in range(arms)]
        Karmed.__init__(self, arms, mus, sigmas, initial_reward_values)

                        




