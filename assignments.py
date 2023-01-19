"""
assignments.py

Assigns TAs to labs using an evolutionary approach from the evo.py framework.

List of Objective functions:
overallocation
conflicts
undersupport
unwilling
unpreferred

List of Agent functions:
swap_columns
swap_rows
flip_val
overlay
remove_unwilling
remove_unpreferred

"""
import pandas as pd
import numpy as np
import os
import random as rnd

import evo

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '.'))
tas = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'tas.csv'))
sections = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'sections.csv'))


def overallocation(sol):
    """ Determines the total overallocation penalty across all TAs.
        This score is based on the 'max_assigned' column in tas.csv.
        E.g. If a TA requests at most 2 labs and they are assigned to 5 the penalty is 3. """
    arr = np.subtract(np.sum(sol, axis=1), np.array(tas.max_assigned))
    return np.sum(np.where(arr<0, 0, arr))


def conflicts(sol):
    """ Determines the amount of time conflicts across all TAs.
        A time conflict happens when a TA is assigned to two labs meeting at the same time.
        If a TA has multiple time conflicts it is only counted as one overall conflict for that TA"""
    arr = np.array(sections.daytime)
    return len(sol[np.array([len(arr[row>0]) != len(set(arr[row>0])) for row in sol])])


def undersupport(sol):
    """ Determines the total amount of underallocation penalties across all TAs.
        This score is based on the 'min_ta' column in sections.csv.
        E.g. If a section needs 3 TAs and is only assigned 1 the penalty is 2. """
    arr = np.subtract(np.array(sections.min_ta), np.sum(sol, axis=0))
    return np.sum(np.where(arr<0, 0, arr))


def unwilling(sol):
    """ Determines the amount of times a TA is assigned to a section they are unwilling to support.
        This score counts all instances of 'unavailable' assignments using the tas.csv file. """
    us = np.array(tas.iloc[:, 3:])
    bools = np.array(us == 'U')
    return np.logical_and(sol.astype(bool), bools).sum()


def unpreferred(sol):
    """ Determines the amount of times a TA is assigned to a section that is unpreferred to them.
        This score counts all the instances of 'willing' assignments using the tas.csv file. """
    us = np.array(tas.iloc[:, 3:])
    bools = np.array(us == 'W')
    return np.logical_and(sol.astype(bool), bools).sum()


def swap_columns(sols):
    """ Agent: swaps two random columns """
    L = sols[0]
    c1 = rnd.randrange(0, L.shape[1])
    c2 = rnd.randrange(0, L.shape[1])
    L[:, [c1, c2]] = L[:, [c2, c1]]
    return L


def swap_rows(sols):
    """ Agent: swaps two random rows """
    L = sols[0]
    r1 = rnd.randrange(0, L.shape[0])
    r2 = rnd.randrange(0, L.shape[0])
    L[[r1, r2]] = L[[r2, r1]]
    return L


def flip_val(sols):
    """ Agent: flip a single binary value """
    L = sols[0]
    r = rnd.randrange(0, L.shape[0])
    c = rnd.randrange(0, L.shape[1])

    if L[r][c] == 0:
        L[r][c] = 1
    else:
        L[r][c] = 0

    return L


def overlay(sols):
    """ Agent: performs an and operation on two solutions """
    L = np.logical_and(sols[0], sols[1])
    return L.astype(int)


def remove_unwilling(sols):
    """ Agent: removes any unwilling assignments """
    L = sols[0]
    us = np.array(tas.iloc[:, 3:])
    bools = np.array(us != 'U')
    return np.logical_and(L.astype(bool), bools).astype(int)


def remove_unpreferred(sols):
    """ Agent: removes any unpreferred assignments """
    L = sols[0]
    ws = np.array(tas.iloc[:, 3:])
    bools = np.array(ws != 'W')
    return np.logical_and(L.astype(bool), bools).astype(int)



def main():

    # create population
    E = evo.Environment()

    # register the fitness criteria
    E.add_fitness_criteria('overallocation', overallocation)
    E.add_fitness_criteria('conflicts', conflicts)
    E.add_fitness_criteria('undersupport', undersupport)
    E.add_fitness_criteria('unwilling', unwilling)
    E.add_fitness_criteria('unpreferred', unpreferred)

    # register the agents
    E.add_agent('swap_columns', swap_columns)
    E.add_agent('swap_rows', swap_rows)
    E.add_agent('flip_val', flip_val)
    E.add_agent('overlay', overlay, k=2)
    E.add_agent('remove_unwilling', remove_unwilling)
    E.add_agent('remove_unpreferred', remove_unpreferred)

    # seed the population with initial solutions
    s1 = np.random.randint(0, 2, (43, 17))
    s2 = np.random.randint(0, 2, (43, 17))
    s3 = np.random.randint(0, 2, (43, 17))
    s4 = np.random.randint(0, 2, (43, 17))
    s5 = np.random.randint(0, 2, (43, 17))
    s6 = np.random.randint(0, 2, (43, 17))
    s7 = np.random.randint(0, 2, (43, 17))
    s8 = np.random.randint(0, 2, (43, 17))
    s9 = np.random.randint(0, 2, (43, 17))
    s10 = np.random.randint(0, 2, (43, 17))
    s11 = np.zeros((43, 17))
    s12 = np.ones((43, 17))
    ps = np.array(tas.iloc[:, 3:])
    s13 = np.array(ps == 'P').astype(int)
    s14 = np.array(ps != 'U').astype(int)
    s15 = np.array(ps != 'W').astype(int)
    solutions = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15]
    for s in solutions:
        E.add_solution(s)

    # run the evolver
    E.evolve(5000000)

    # print result
    print(E)


if __name__ == '__main__':
    main()