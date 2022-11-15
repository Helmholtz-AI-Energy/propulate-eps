#!/usr/bin/env python3
import numpy as np
import optuna
import sys
from pathlib import Path
import os
from propulate.utils import get_default_propagator
from propulate.propagators import SelectBest, SelectWorst
from propulate import Islands
import random
from mpi4py import MPI
import json
import heapq
import pandas as pd

#################################
# PROPULATE FUNCTION BENCHMARKS #
#################################
# Digalakis, J. G., & Margaritis, K. G. (2001). 
# On benchmarking functions for genetic algorithms. 
# International journal of computer mathematics, 77(4), 481-506.
#
# F1: SPHERE
# Use N_dim = 2.
# Limits: -5.12 <= x_i <= 5.12
def sphere(params):
    """Sphere benchmark function."""
    vec = np.array(list(params.values()))
    return np.sum(vec**2)

# F2: ROSENBROCK
# N_dim = 2
# Limits: -2.048 <= x_i <= 2.048
def rosenbrock(params):
    """Rosenbrock benchmark function."""
    vec = np.array(list(params.values()))
    return 100 * (vec[0]**2 - vec[1])**2 + (1 - vec[0])**2

# F3: STEP
# Use N_dim = 5.
# Limits: -5.12 <= x_i <= 5.12
def step(params):
    """Step benchmark function."""
    vec = np.array(list(params.values()))
    return np.sum(vec.astype(int))

# F4: QUARTIC
# Use N_dim = 30.
# Limits: -1.28 <= x_i <= 1.28
def quartic(params):
    """Quartic benchmark function."""
    vec = np.array(list(params.values()))
    idx = np.arange(1, len(vec)+1)
    gauss = np.random.normal(size = len(vec))
    ret = np.sum(idx * vec**4 + gauss)
    return abs(ret)

# F5: RASTRIGIN
# Use N_dim = 20.
# Limits: -5.12 <= x_i <= 5.12
def rastrigin(params):
    """Rastrigin benchmark function."""
    A = 10.
    vec = np.array(list(params.values()))
    return A * len(vec) + np.sum(vec**2 - A * np.cos(2 * np.pi * vec))

# F6: GRIEWANK
# Use N_dim = 10.
# Limits: -600 <= x_i <= 600
def griewank(params):
    """Griewank benchmark function."""
    vec = np.array(list(params.values()))
    idx = np.arange(1, len(vec)+1)
    return 1 + 1.0 / 4000 * np.sum(vec**2) - np.prod(np.cos(vec / np.sqrt(idx)))

# F7: SCHWEFEL
# Use N_dim = 10.
# Limits: -500 <= x_i <= 500
def schwefel(params):
    """Schwefel benchmark function."""
    V = 418.982887
    vec = np.array(list(params.values()))
    return V * len(vec) - np.sum(vec * np.sin(np.sqrt(np.abs(vec))))

# Lunacek, M., Whitley, D., & Sutton, A. (2008, September). 
# The impact of global structure on search. 
# In International Conference on Parallel Problem Solving from Nature 
# (pp. 498-507). Springer, Berlin, Heidelberg.
#
# F8: DOUBLE-SPHERE
# Use N_dim = 30.
# Limits: -5.12 <= x_i <= 5.12
def bisphere(params):
    """Lunacek's double-sphere benchmark function."""
    vec = np.array(list(params.values()))
    N = len(vec)
    d = 1
    s = 1 - np.sqrt(1 / (2 * np.sqrt(N + 20) - 8.2))
    mu1 = 2.5
    mu2 = - np.sqrt( (mu1**2 - d) / s)
    return min(np.sum((vec - mu1)**2), d * N + s * np.sum((vec - mu2)**2) )

# F9: DOUBLE-RASTRIGIN
# Use N_dim = 30.
# Limits: -5.12 <= x <= 5.12 
def birastrigin(params):
    """Lunacek's double-Rastrigin benchmark function."""
    vec = np.array(list(params.values()))
    N = len(vec)
    d = 1
    s = 1 - np.sqrt(1 / (2 * np.sqrt(N + 20) - 8.2))
    mu1 = 2.5
    mu2 = - np.sqrt( (mu1**2 - d) / s)
    return min(np.sum((vec - mu1)**2), d * N + s * np.sum((vec - mu2)**2) ) + 10 * np.sum(1 - np.cos(2 * np.pi * (vec - mu1)))


def get_limits(fname):
    """Determine search-space limits of input benchmark function."""
    if fname == "sphere":
        function = sphere
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
        }
    
    elif fname == "rosenbrock":
        function = rosenbrock
        limits = {
            "a": (-2.048, 2.048),
            "b": (-2.048, 2.048),
        }
    
    elif fname == "step":
        function = step
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12)
        }
    
    elif fname == "quartic":
        function = quartic
        limits = {
            "a": (-1.28, 1.28),
            "b": (-1.28, 1.28),
            "c": (-1.28, 1.28),
            "d": (-1.28, 1.28),
            "e": (-1.28, 1.28),
            "f": (-1.28, 1.28),
            "g": (-1.28, 1.28),
            "h": (-1.28, 1.28),
            "i": (-1.28, 1.28),
            "j": (-1.28, 1.28),
            "k": (-1.28, 1.28),
            "l": (-1.28, 1.28),
            "m": (-1.28, 1.28),
            "n": (-1.28, 1.28),
            "o": (-1.28, 1.28),
            "p": (-1.28, 1.28),
            "q": (-1.28, 1.28),
            "r": (-1.28, 1.28),
            "s": (-1.28, 1.28),
            "t": (-1.28, 1.28),
            "u": (-1.28, 1.28),
            "v": (-1.28, 1.28),
            "w": (-1.28, 1.28),
            "x": (-1.28, 1.28),
            "y": (-1.28, 1.28),
            "z": (-1.28, 1.28),
            "A1": (-1.28, 1.28),
            "B1": (-1.28, 1.28),
            "C1": (-1.28, 1.28),
            "D1": (-1.28, 1.28)
        }
    
    elif fname == "bisphere":
        function = bisphere
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12),
            "f": (-5.12, 5.12),
            "g": (-5.12, 5.12),
            "h": (-5.12, 5.12),
            "i": (-5.12, 5.12),
            "j": (-5.12, 5.12),
            "k": (-5.12, 5.12),
            "l": (-5.12, 5.12),
            "m": (-5.12, 5.12),
            "n": (-5.12, 5.12),
            "o": (-5.12, 5.12),
            "p": (-5.12, 5.12),
            "q": (-5.12, 5.12),
            "r": (-5.12, 5.12),
            "s": (-5.12, 5.12),
            "t": (-5.12, 5.12),
            "u": (-5.12, 5.12),
            "v": (-5.12, 5.12),
            "w": (-5.12, 5.12),
            "x": (-5.12, 5.12),
            "y": (-5.12, 5.12),
            "z": (-5.12, 5.12),
            "A1": (-5.12, 5.12),
            "B1": (-5.12, 5.12),
            "C1": (-5.12, 5.12),
            "D1": (-5.12, 5.12)
        }
    
    elif fname == "birastrigin":
        function = birastrigin
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12),
            "f": (-5.12, 5.12),
            "g": (-5.12, 5.12),
            "h": (-5.12, 5.12),
            "i": (-5.12, 5.12),
            "j": (-5.12, 5.12),
            "k": (-5.12, 5.12),
            "l": (-5.12, 5.12),
            "m": (-5.12, 5.12),
            "n": (-5.12, 5.12),
            "o": (-5.12, 5.12),
            "p": (-5.12, 5.12),
            "q": (-5.12, 5.12),
            "r": (-5.12, 5.12),
            "s": (-5.12, 5.12),
            "t": (-5.12, 5.12),
            "u": (-5.12, 5.12),
            "v": (-5.12, 5.12),
            "w": (-5.12, 5.12),
            "x": (-5.12, 5.12),
            "y": (-5.12, 5.12),
            "z": (-5.12, 5.12),
            "A1": (-5.12, 5.12),
            "B1": (-5.12, 5.12),
            "C1": (-5.12, 5.12),
            "D1": (-5.12, 5.12)
        }
    
    elif fname == "rastrigin":
        function = rastrigin
        limits = {
            "a": (-5.12, 5.12),
            "b": (-5.12, 5.12),
            "c": (-5.12, 5.12),
            "d": (-5.12, 5.12),
            "e": (-5.12, 5.12),
            "f": (-5.12, 5.12),
            "g": (-5.12, 5.12),
            "h": (-5.12, 5.12),
            "i": (-5.12, 5.12),
            "j": (-5.12, 5.12),
            "k": (-5.12, 5.12),
            "l": (-5.12, 5.12),
            "m": (-5.12, 5.12),
            "n": (-5.12, 5.12),
            "o": (-5.12, 5.12),
            "p": (-5.12, 5.12),
            "q": (-5.12, 5.12),
            "r": (-5.12, 5.12),
            "s": (-5.12, 5.12),
            "t": (-5.12, 5.12)
        }
    
    elif fname == "griewank":
        function = griewank
        limits = {
            "a": (-600., 600.),
            "b": (-600., 600.),
            "c": (-600., 600.),
            "d": (-600., 600.),
            "e": (-600., 600.),
            "f": (-600., 600.),
            "g": (-600., 600.),
            "h": (-600., 600.),
            "i": (-600., 600.),
            "j": (-600., 600.)
        }
    
    elif fname == "schwefel":
        function = schwefel
        limits = {
            "a": (-500., 500.),
            "b": (-500., 500.),
            "c": (-500., 500.),
            "d": (-500., 500.),
            "e": (-500., 500.),
            "f": (-500., 500.),
            "g": (-500., 500.),
            "h": (-500., 500.),
            "i": (-500., 500.),
            "j": (-500., 500.)
            }
    else:
        sys.exit("ERROR: Function undefined...exiting")
    return function, limits


def optuna_objective(trial, fname):
    func, limits = get_limits(fname)
    
    opt_dict = {}
    for k in limits:
        #print(k)
        opt_dict[k] = trial.suggest_float(k, *limits[k])
    
    return func(opt_dict)


def propulate_objective(
    fname,
    num_islands,
    migration_prob,
    pollination=True,
    mate_prob=0.7,
    mut_prob=0.4,
    random_prob=0.1
):
    func, limits = get_limits(fname)
    pop_size = MPI.COMM_WORLD.size // num_islands
    # checkpoint file is only for saving
    checkpoint = Path(
        f"/hkfs/work/workspace/scratch/qv2382-propulate/exps/function_benchmark/logs/"
        f"search/{fname}/checkpoints/"
        f"{os.environ['SLURM_JOBID']}-pop{pop_size}-islands{num_islands}-migprob{migration_prob}"
        f"-mate{mate_prob}-mut{mut_prob}-rand{random_prob}"
    )
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    if rank == 0:
        print("checkpoint path:", checkpoint)
    checkpoint.mkdir(exist_ok=True, parents=True)
    checkpoint = str(checkpoint / "pop_cpt.p")

    num_gens = 256

    rng = random.Random(int(os.environ["SLURM_JOBID"]) + MPI.COMM_WORLD.rank)

    propagator = get_default_propagator(
        pop_size=pop_size,
        limits=limits,
        mate_prob=mate_prob,
        mut_prob=mut_prob,
        random_prob=random_prob,
        rng=rng
    )
    islands = Islands(
        func,
        propagator,
        generations=num_gens,
        num_isles=num_islands,
        rng=rng,
        # isle_sizes=[4, 4, 4, 4],  # migration_topology=migration_topology,
        load_checkpoint="nothing",  # pop_cpt.p",
        save_checkpoint=checkpoint,
        migration_probability=migration_prob,
        emigration_propagator=SelectBest,
        immigration_propagator=SelectWorst,
        pollination=pollination,
    )

    full_dict_loc = Path(
        f"/hkfs/work/workspace/scratch/qv2382-propulate/exps/function_benchmark/logs/"
        f"search/{fname}/results/overall/"
    )
    full_dict_loc.mkdir(exist_ok=True, parents=True)
    full_dict = full_dict_loc / "search_results-3.txt"

    out_loc = Path(
        f"/hkfs/work/workspace/scratch/qv2382-propulate/exps/function_benchmark/logs/"
        f"search/{fname}/results/"
        f"{os.environ['SLURM_JOBID']}-pop{pop_size}-poll{pollination}-islands{num_islands}"
        f"-migprob{migration_prob}-mate{mate_prob}-mut{mut_prob}-rand{random_prob}"
    )
    out_loc.mkdir(exist_ok=True, parents=True)
    if rank == 0:
        print("out logs:", out_loc)

    best = islands.evolve(top_n=3, logging_interval=100, DEBUG=0, out_file=out_loc / "summary.png")
    # TODO: need to record when the peak is reached!
    gens = [x.generation for x in islands.propulator.population]
    ls = [x.loss for x in islands.propulator.population]
    # rn = [x.rank for x in islands.propagator.population]

    island_size = size // num_islands
    island_rank = rank % island_size
    island_id = rank // island_size
    if island_rank == 0:
        # get the top 1 best from the island and their indices
        bottomk_ls = heapq.nsmallest(3, ls)  # best 3
        argbotk = [ls.index(i) for i in bottomk_ls]  # indexes of the best 3
        botk_gens = [gens[i] for i in argbotk]  # generation of best 3
        pnt_str = [{"ls": i, "gen": g} for i, g in zip([ls[i] for i in argbotk], botk_gens)]
        print(f"Island: {island_id} - Best 3 results: {pnt_str}")

        # send results to rank 0
        #comm = MPI.COMM_WORLD
        #if rank == 0:
        #    island_out_dict = {}
        #    island_out_dict[0] = pnt_str
        #    for rcv in range(1, num_islands):
        #        rcv_rank = rcv * island_size
        #        lp_res = comm.recv(source=rcv_rank)
        #        island_out_dict[rcv] = lp_res
        #    island_out_dict["best"] = best
        #if rank != 0:
        #    comm.send(pnt_str, dest=0)

    # best = islands.propulator.summarize(top_n=3, out_file=out_loc / "summary.png", DEBUG=1)
    #if rank == 0:
        #old_data = {}
        #try:
        #    old_data = json.loads(full_dict.read_text())
        #except FileNotFoundError:
        #    print("No file was found!, creating one")

    #    old_data[f"pop-{pop_size}-poll{pollination}-islands-{num_islands}" \
    #             f"-migprob-{migration_prob}-mate-{mate_prob}-mut-{mut_prob}-rand-{random_prob}"] \
    #        = island_out_dict
    #    full_dict.write_text(json.dumps(old_data, indent=4))
    return best
