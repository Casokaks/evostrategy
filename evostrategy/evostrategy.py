"""
evostrategy
==================================
Library implementing evolution strategy

Author: 
Andrea Casati, andrea1.casati@gmail.com
Casokaks (https://github.com/Casokaks/)

Created on: Nov 26th 2018

"""


import sys
import numpy as np
import multiprocessing as mp
import pprint
from copy import deepcopy
from datetime import datetime
from .utils import dedup_list


def worker_process(arg):
    get_reward_func, weights, minimization = arg
    return get_reward_func(weights, minimization)


class EvolutionStrategy(object):
    '''Class implementing evolution strategy'''
    
    def __init__(self, init_solution, get_reward_func, solution_bounds=None, minimization=False,
                 population_size=50, keep_top=5, no_iterations=1000, 
                 early_stop=50, round_digs=4, init_std=1, learning_rate=0.01, decay=0.995, 
                 num_threads=1, seed=None, verbose=False):
        '''Initialization method'''
        
        if isinstance(seed, type(None)) == False:
            np.random.seed(seed)
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads
        self.minimization = minimization
        self.init_solution = init_solution
        self.solutions = []
        self.top_solutions = []
        self.solution_bounds = solution_bounds
        self.get_reward = get_reward_func
        self.population_size = population_size
        self.no_iterations = no_iterations
        self.learning_rate = learning_rate
        self.decay = decay    
        self.early_stop = early_stop    
        self.keep_top = keep_top   
        self.round_digs = round_digs
        self.init_std = init_std 
        self.verbose = verbose
                    
    def get_solutions(self):
        '''Returns the list of solutions'''
        return self.solutions

    def get_init_solution(self):
        '''Returns the initial solution'''
        return self.solutions[0]

    def get_last_solution(self):
        '''Returns the last solution'''
        return self.solutions[-1]

    def get_top_solutions(self, top_n=1):
        '''Returns the best solution(s)'''
        return self.top_solutions[:top_n]
    
    def print_top_solutions(self):
        '''Prints the top solutions'''
        for tops in self.top_solutions(top_n=self.keep_top):
            print(pprint.pformat(tops))

    def make_solution_feasible(self, solution):
        '''Update the solution so that it satisfy min/max constraints'''
        for i in range(len(solution)):
            minval = self.solution_bounds[i][0]
            maxval = self.solution_bounds[i][1]
            x = min(maxval,max(solution[i],minval))
            solution[i] = x
        return solution
    
    def _get_rewards(self, population, pool=None):
        '''Compute rewards on the given population using the provided get_reward function'''
        if pool is not None:
            worker_args = ((self.get_reward, solution, self.minimization) 
                            for solution in population)
            rewards = pool.map(worker_process, worker_args)
        else:
            rewards = [self.get_reward(solution, self.minimization) 
                       for solution in population]
        rewards = [round(x,self.round_digs) for x in rewards]
        return rewards

    def _get_population(self, solutions):
        '''Generate new population based on given solutions'''
        
        # get distributions from given solutions (top solutions included)
        solutions_transp = np.array(solutions).T
        mean = np.mean(solutions_transp, axis=1)
        if len(solutions) > 1:
            std = np.std(solutions_transp, axis=1)    
        else:
            std = [self.init_std]*len(mean)

        # init new population with current top solutions 
        # top solutions is built with no dups
        new_population = [tops['solution'] for tops in self.top_solutions]
        
        # create and add new solutions to the new population
        solution_size = len(solutions[0])
        for i in range(self.population_size-len(new_population)):
            new_population.append(self._get_new_solution(solution_size=solution_size, 
                                                         mean=mean, std=std))
        return new_population
    
    
    def _get_new_solution(self, solution_size, mean=0, std=1):
        '''Create new solution'''

        lr_none = isinstance(self.learning_rate, type(None))
        decay_none = isinstance(self.decay, type(None))

        # learning rate version:
        # generate random variation from normal distribution (0,1)
        # select one of the current top_solutions to be the base for the update
        # mutate the base solution with (variation * learning rate) 
        if (lr_none==False and decay_none==False):
            variation = np.random.normal(loc=0, scale=1, size=solution_size)
            base = np.random.randint(low=0,high=len(self.top_solutions))
            base = self.top_solutions[base]['solution']
            new_solution = base + variation * self.learning_rate

        # no learning rate version:
        # draw new sequence of numbers from normal distribution with
        # mean and std of the previous population 
        else:
            new_solution = np.random.normal(loc=mean, scale=std, size=solution_size)

        # make new solution feasible
        if isinstance(self.solution_bounds,type(None)) == False:
            new_solution = self.make_solution_feasible(new_solution)   

        # round the solution
        new_solution = [round(x,self.round_digs) for x in new_solution]
        return new_solution
    
    def _update_solutions(self, population, rewards, iteration):
        '''Append new solutions to the solutions list'''
        for i in range(len(population)):
            self.solutions.append({'id_iteration':iteration,
                                   'id_solution':i+1,
                                   'solution':population[i],
                                   'reward':rewards[i]})

    def _init_top_solutions(self):
        '''Initialize the top solutions list with initial solution'''
        self.top_solutions = [{'id_iteration':1,
                               'id_solution':1,
                               'solution':self.init_solution, 
                               'reward':self.get_reward(self.init_solution)}]

    def _sort_top_solutions(self):
        '''Sort top solutions from high reward to low'''  
        
        # sort top solutions rewards (low > high)
        temp_solutions = deepcopy(self.top_solutions)
        self.top_solutions = []
        rewards = list(sorted([s['reward'] for s in temp_solutions]))
        
        # revert order for maximization problems (high > low)
        if self.minimization==False: 
            rewards = list(reversed(rewards))
        
        # create new sorted top_solutions
        for r in rewards:
            for s in temp_solutions:
                if s['reward'] == r:
                    self.top_solutions.append(s)
                    
        # consistency check >>> first try dedup        
        if len(self.top_solutions) != len(temp_solutions):
            self.top_solutions = dedup_list(self.top_solutions)
            
        # consistency check >>> if still persist raise error        
        if len(self.top_solutions) != len(temp_solutions):
            print('\nERROR: Sorting of tops_solutions produced '
                  '{} solutions instead of {}'.format(
                          len(temp_solutions),len(self.top_solutions)))
            sys.exit(-1)
        
    def _update_top_solutions_if_top(self, solution, reward, solution_id, iteration_id):
        '''Update top solutions with new solution if better then any already inside'''
        
        # if:
        # - solution better than worse top solution 
        # (NOTE: top_solutions always ordered by reward)
        # - and solution not already in top_solutions
        # then add new solution to top_solutions

        # init useful object             
        top_solutions = [tops['solution'] for tops in self.top_solutions]
        top_rewards = [tops['reward'] for tops in self.top_solutions]
        
        # check if solution should be inserted into top solutions
        update_tops = False
        
        # maximization case
        if self.minimization == False:
            if reward > top_rewards[-1]:
                if solution not in top_solutions:
                    update_tops = True
                else:
                    pass
            else:
                pass
                    
        # minimization case (self.minimization == True)
        else:
            if reward < top_rewards[-1]:
                if solution not in top_solutions:
                    update_tops = True
                else:
                    pass
            else:
                pass
            
        # update top_solutions if needed
        if update_tops==True:     
            
            # create new top to add
            new_top = {'id_iteration':iteration_id,
                       'id_solution':solution_id,
                       'solution':solution, 
                       'reward':reward}
            
            if self.debug==True:
                print('\nReward {} > self.top_solutions[-1][reward] {}'.format(
                        reward, top_rewards[-1]))
                print('\nTop solutions:')
                print(pprint.pformat(self.top_solutions))
                print('len = {} | keep_top = {}'.format(
                        len(self.top_solutions), self.keep_top))

            # if top solutions already max number then replace last item
            if len(self.top_solutions) == self.keep_top:
                self.top_solutions[-1] = new_top
                if self.debug==True:
                    print('\nLast item replaced')

            # else it means that top solutions is not full, so just add 
            elif len(self.top_solutions) < self.keep_top:
                self.top_solutions.append(new_top)
                if self.debug==True:
                    print('\nItem added')
            
            # else error
            else:
                print('\nERROR: top solution length unexpectedly {}, '
                      'while keep_top is {}'.format(len(self.top_solutions), self.keep_top))
                print('\nTop solutions:')
                print(pprint.pformat(self.top_solutions))
                sys.exit(-1)
                
            if self.debug==True:
                print('\nNew top solutions:')
                print(pprint.pformat(self.top_solutions))
                
            # sort new top solutions 
            self._sort_top_solutions()

    def _update_top_solutions(self, population, rewards, iteration):
        '''Update top solutions with given population and rewards'''
        
        # if empty then fill with initial solution (repeated n times)
        if len(self.top_solutions) == 0:
            self._init_top_solutions()
            if self.debug==True:
                print('\nTop solution initialized')
                print(pprint.pformat(self.top_solutions))

        # update top solutions with new tops
        for i in range(len(rewards)):            
            self._update_top_solutions_if_top(solution=population[i], 
                                              reward=rewards[i],
                                              solution_id=i+1,
                                              iteration_id=iteration)

    def _update_learning_rate(self):
        '''Decrease learning rate by decay'''
        lr_none = isinstance(self.learning_rate, type(None))
        decay_none = isinstance(self.decay, type(None))
        if (lr_none==False and decay_none==False):
            self.learning_rate *= self.decay
    
    def run(self, print_step=1, debug=False):
        '''Runs the evolution strategy optimization'''
        
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None        
        self.debug = debug
        
        print('\n--- Evolution Strategy search started at {} ---'.format(datetime.now()))

        lr_none = isinstance(self.learning_rate, type(None))
        decay_none = isinstance(self.decay, type(None))

        if self.verbose==True:
            print('\nMethod:')
            if (lr_none==False and decay_none==False):
                print('Mutation of top solutions based on decreasing learning rate')
            else:
                print('Mutation of populations base on their own normal distribution')
            print('Objective function minimization = {}'.format(self.minimization))
            print('Learning rate = {}'.format(self.learning_rate))
            print('Decay = {}'.format(self.decay))
            print('\nInitial solution:')
            print(pprint.pformat(self.init_solution))
            print('')
        
        iter_no_improve = 0
        population = [self.init_solution]
        self._init_top_solutions()
        for iteration in range(1, self.no_iterations+1):      
            
            # setup new population
            population = self._get_population(solutions=population)
            
            # get rewards for the population            
            rewards = self._get_rewards(population=population, pool=pool)
            
            # update solutions 
            self._update_solutions(population, rewards, iteration)
            prev_tops = self.top_solutions
            self._update_top_solutions(population, rewards, iteration)
            
            # update learning rate
            self._update_learning_rate()
                
            # print update
            if self.verbose == True and iteration % print_step == 0:
                print('Iteration {})\t top rewards = [{} - {}]'.format(iteration, 
                     self.top_solutions[0]['reward'], self.top_solutions[-1]['reward']))
        
            if self.debug==True:
                print('\nIterations with no improvement {}'.format(iter_no_improve))
                print('\nPopulation:')
                print(pprint.pformat(population))
                print('\nRewards:')
                print(pprint.pformat(rewards))
                print('\nTop solutions:')
                print(pprint.pformat(self.top_solutions))

            # early stop
            if self.top_solutions == prev_tops: iter_no_improve += 1
            else: iter_no_improve = 0               
            if iter_no_improve >= self.early_stop:
                if self.verbose==True:
                    print('\n>>> Early stop after {} iterations with no improvement'.format(iter_no_improve))
                break
        
        # closing up
        print('\n--- Evolution Strategy search completed at {} ---'.format(datetime.now()))
        if self.debug==True:
            print('\nInit solutions (#1):') 
            print(pprint.pformat(self.get_init_solution()))
            print('\nBest solutions (#{}):'.format(len(self.top_solutions)))
            print(pprint.pformat(self.get_top_solutions(top_n=self.keep_top)))
            print('\n')
        else:
            if self.verbose == True:
                print('\nBest solution:')
                print(pprint.pformat(self.get_top_solutions(top_n=1)))
                print('\n')
            
        if pool is not None:
            pool.close()
            pool.join()
            
            