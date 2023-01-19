"""
evo.py: An evolutionary computing framework
"""

import random as rnd
import copy
from functools import reduce
import pickle
import time
import json

class Environment:

    def __init__(self):
        self.pop = {} # evaluation tuple ((name1, obj1), (name2, obj2)...) --> sol

        self.fitness = {} # name --> function
        self.agents = {} # name --> (operator, num_solution)

    def size(self):
        """ The number of solutions in the population """
        return len(self.pop)

    def add_fitness_criteria(self, name, f):
        """ Every new solution is evaluated wrt each of the fitness criteria """
        self.fitness[name] = f

    def add_agent(self, name, op, k=1):
        """ Register an agent with the framework """
        self.agents[name] = (op, k)

    def add_solution(self, sol):
        evaluation = tuple([(name, f(sol)) for name, f in self.fitness.items()])
        self.pop[evaluation] = sol

    def get_random_solutions(self, k=1):
        """ Pick k random solutions from the population """
        if self.size() == 0:
            return []
        else:
            popvals = tuple(self.pop.values())
            return [copy.deepcopy(rnd.choice(popvals)) for _ in range(k)]

    def run_agent(self, name):
        """ Invoke an agent against the population """
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)

    @staticmethod
    def _dominates(p, q):
        """ p = evaluation of solution: ((obj1, score1), (obj2, score2), ...) """
        pscores = [score for _, score in p]
        qscores = [score for _, score in q]
        score_diffs = list(map(lambda x,y: y-x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0

    @staticmethod
    def _reduce_nds(S, p):
        return S - {q for q in S if Environment._dominates(p, q)}


    @staticmethod
    def _fits_constraints(k):
        f = open('constraints.json')
        data = json.load(f)
        f.close()
        for obj in k:
            if obj[1] > data[obj[0]]:
                return False
        return True


    def remove_dominated(self):
        """ Remove dominated solutions from the populations """
        nds = reduce(self._reduce_nds, self.pop.keys(), self.pop.keys())
        new = {k for k in nds if Environment._fits_constraints(k)}
        self.pop = {k: self.pop[k] for k in new}

    def evolve(self, n=1, dom = 100, status = 10000, sync = 1000):
        """ Run n random agents (default = 1) """

        start = time.time()
        agent_names = list(self.agents.keys())
        for i in range(n):
            pick = rnd.choice(agent_names)
            self.run_agent(pick)

            if i % dom == 0:
                self.remove_dominated()

            if i % status == 0:
                self.remove_dominated()
                print('Iteration: ', i)
                print('Population size: ', self.size())
                rslt = ''
                for eval, sol in self.pop.items():
                    rslt += str(dict(eval)) + '\n'
                print(rslt)

            if i % sync == 0:

                # load saved solutions and merge them into our population
                try:
                    with open('solutions.dat', 'rb') as file:
                        loaded = pickle.load(file)
                        for eval, sol in loaded.items():
                            self.pop[eval] = sol
                except Exception as e:
                    print(e)

                # remove dominated solutions before saving to disk
                self.remove_dominated()

                # save the solutions
                with open('solutions.dat', 'wb') as file:
                    pickle.dump(self.pop, file)

            if time.time() - start > 600:
                print('Reached 10 minutes...')
                break

        self.remove_dominated()
        print(self)

    def __str__(self):
        """ Outputs the solutions in the population """
        rslt = ''
        for eval, sol in self.pop.items():
            rslt += str(dict(eval)) + ':\t' + str(sol) + '\n'
        return rslt








