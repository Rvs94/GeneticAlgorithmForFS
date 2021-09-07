# -*- coding: utf-8 -*-
"""
@author: rschulte
"""

import os
import numpy as np
import h5py
import datetime
import time

import warnings
warnings.filterwarnings("ignore")

def h5_to_dict(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            aux = item[()]
            if type(aux[0]) == np.bytes_:
                ans[key] = np.char.decode(aux,encoding='utf8')
            else:
                ans[key] = aux
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = h5_to_dict(h5file, path + key + '/')
    return ans

def dict_to_h5(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + str(key)] = item
        elif isinstance(item, list):
            if type(item[0]) == str: # list of strings
                h5file[path + str(key)] = np.chararray.encode(np.asarray(item), encoding='utf8') 
            else:
                h5file[path + str(key)] = np.asarray(item)
        elif isinstance(item, dict):
            dict_to_h5(h5file, path + str(key) + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))
            
class GeneticAlgorithm():
    def __init__(self,evalFunction,n_iter=100,
                 pop_size=100,n_parent=8,mutation_rate=0.1, hf_filename = 'results.hdf5',
                 n_iter_local=8,n_random=8):
        # to store results
        self.hf_filename = hf_filename
        
        # GA specifics
        self.EF = evalFunction
        self.n_iter = n_iter
        self.n_iter_local = n_iter_local
        self.n_random = n_random
        self.n_type = self.EF.n_type
        self.n_feat = self.EF.n_feat
        self.list_feat = self.EF.list_feat
        self.pop_size = pop_size
        self.n_parent = n_parent
        self.mutation_rate = mutation_rate
        
        # Fitness variables
        self.a = [0.25, 0.1, 0.6]
        self.c = 0.05
    
    def initialize(self):
        # Initialize population
        print('GA v1 - Initialize')
        population = self.first_generation()
        fitness = self.determine_fitness(population)
        return population, fitness
    
    def first_generation(self):
        # Creates initial population
        population = []
        
        while True:
            # Chromosome: shape is [n_feat x n_type]
            chromosome = [list(np.random.choice(2, size=(self.n_feat), replace=True, p=[0.8,0.2])) for _ in range(self.n_type)]
            # Check the validity of the child
            if np.sum(chromosome) > 0:
                population.append(chromosome)
            if len(population) == self.pop_size:
                break        
        return population
    
    def random_pop(self,num_pop,):
        population = []
        while True:
            # Chromosome: shape is [n_feat x n_type]
            chromosome = [list(np.random.choice(2, size=(self.n_feat), replace=True, p=[0.8,0.2])) for _ in range(self.n_type)]
            # Check the validity of the child
            if np.sum(chromosome) > 0:
                population.append(chromosome)
            if len(population) == num_pop:
                break
        return population
    
    def determine_fitness(self,population):
        # Determine fitness of population
        fitness = []
        self.score = []
        self.test_score = []
        Tstart = time.time()
        for i,pop in enumerate(population):
            print('\rProgress: {}/{} - Elapsed time: {}'.format(i+1,self.pop_size,datetime.timedelta(seconds=round(time.time()-Tstart))),end="")
            score = self.EF.CV_score(pop) # Should return ov, ss and tr score
            n_size = sum([sum(l) for l in pop])
            
            sub_fitness = 0
            for i_key,key in enumerate(score):
                sub_fitness += self.a[i_key]*np.mean(score[key])
            if np.isnan(sub_fitness):
                sub_fitness = 0

            fitness.append(sub_fitness + self.c/n_size)
            self.score.append(score)
        fitness = np.asarray(fitness)
        print('\rProgress: {}/{} - Elapsed time: {}'.format(i+1,self.pop_size,datetime.timedelta(seconds=round(time.time()-Tstart))),end="")
        return fitness
    
    def parent_selection(self,population,fitness):
        # select ideal parents
        # Based on score
        sortID1 = np.argsort(fitness)[-int(self.n_parent/2):]
        # Based on randomness
        w = fitness.copy()

        sortID2 = np.random.choice(len(population), size=int(self.n_parent/2), replace=False, p=w/w.sum())
        parents = [population[i] for i in np.concatenate([sortID1,sortID2])]
        best_parent = population[np.argmax(fitness)]
        return parents,best_parent
    
    def breed(self,parent1,parent2):
        # Breeding based on random selection
        # Initialize random ints
        r_f = np.random.randint(self.n_feat,size=self.n_type)
                
        # Feature genes
        child = []
        for t,f in enumerate(r_f):
            child.append(np.concatenate([parent1[t][:f],parent2[t][f:]]))
        return child
    
    def mutate(self,individual,mutationRate):
        # Mutation
        while np.random.random() < mutationRate:
            # Mutate feature (binary flip)
            r_tp = np.random.randint(self.n_type)
            r_f = np.random.randint(self.n_feat)
            individual[r_tp][r_f] = not individual[r_tp][r_f]
        return individual
    
    def breed_population(self,parents, num_random=0):
        # breed & mutate the children
        children = []
        while True:
            # Select two random parents
            p = np.random.choice(self.n_parent, 2, replace=False)
            # Create the child
            child = self.breed(parents[p[0]],parents[p[1]])
            child = self.mutate(child,self.mutation_rate)
            # Check the validity of the child
            if np.sum(child) > 0:
                children.append(child)
            if len(children) == self.pop_size - num_random:
                break
        return children
    
    def next_generation(self,population,fitness):
        # Next generation
        # Determine whether to add randoms
        q1,q3 = np.percentile(fitness,[25,75],interpolation='midpoint')
        if q3-q1 < 0.002:
            random_population = self.random_pop(self.n_random)
            num_random = self.n_random
        else:
            random_population = []
            num_random = 0
        # Select top parents
        parents,best_parent = self.parent_selection(population,fitness)
        
        # Breed next generation
        next_population = self.breed_population(parents, num_random=num_random+1) # +2 as we add the best & opt_parent as well
        next_population += random_population
        next_population.append(best_parent)
        
        # Determine its fitness
        next_fitness = self.determine_fitness(next_population)
        return next_population,next_fitness
    
    def iterate(self):
        # Ensure the h5py file is always closed, even on exceptions
        with h5py.File(self.hf_filename, 'w') as hf_res:
            # Initialize
            self.elite = []
            self.elite_fitness = []
            best = 0
            
            # Determine gen 0 population & fitness
            self.population,self.fitness = self.initialize()
            self.elite.append(self.population[np.argmax(self.fitness)])
            self.elite_fitness.append(max(self.fitness))
    
            # Get scores
            cv_score = self.score[np.argmax(self.fitness)]
            
            # Print results
            print(' - Gen {:3d} - Fitness (avg): {:.3f} ({:.3f})- Opt Performance: {:.2f} +/- {:.2f}% - ss {:.2f} +/- {:.2f}% - tr {:.2f} +/- {:.2f}%'.format(
                    0,max(self.fitness),np.mean(self.fitness),
                    *np.ravel([[np.mean(cv_score[key])*100,np.std(cv_score[key])*100] for key in cv_score]),
                    ))
        
            # Save results
            dict_to_h5(hf_res,'/Gen {}/'.format(0),{'cv_score':cv_score,
                                                      'elite_fitness':self.elite_fitness[-1],
                                                      'chromosome':self.elite[-1],
                                                      'fitness':self.fitness}) 
        
            # Iteration
            for i in range(self.n_iter-1):
                # Update previous population & fitness
                prev_population = self.population
                prev_fitness = self.fitness
                
                # Determine new generation
                self.population,self.fitness = self.next_generation(prev_population,prev_fitness)
                
                # Append fittest 
                self.elite.append(self.population[np.argmax(self.fitness)])
                self.elite_fitness.append(max(self.fitness))
                
                # Get scores
                cv_score = self.score[np.argmax(self.fitness)]
                
                # Print
                print(' - Gen {:3d} - Fitness (avg)): {:.3f} ({:.3f})- Opt Performance: {:.2f} +/- {:.2f}% - ss {:.2f} +/- {:.2f}% - tr {:.2f} +/- {:.2f}%'.format(
                    i+1,max(self.fitness),np.mean(self.fitness),
                    *np.ravel([[np.mean(cv_score[key])*100,np.std(cv_score[key])*100] for key in cv_score]),
                    ))
                
                # Save all
                dict_to_h5(hf_res,'/Gen {}/'.format(i+1),{'cv_score':cv_score,
                                                            'elite_fitness':self.elite_fitness[-1],
                                                            'chromosome':self.elite[-1],
                                                            'fitness':self.fitness}) 
                # Check for improvement
                if self.elite_fitness[-1] > best:
                    best = self.elite_fitness[-1]
                    count = 0
                    if self.mutation_rate >= 0.1:
                        self.mutation_rate -= 0.05
                # If fitness does not increase for 10 iterations, increase mutation until stop
                elif self.elite_fitness[-1] == best:
                    count += 1
                    if count == 10 and self.mutation_rate <= 0.2:
                        self.mutation_rate += 0.05
                        count = 0
                    elif self.mutation_rate > 0.2:
                        break
        return self.elite,self.elite_fitness
