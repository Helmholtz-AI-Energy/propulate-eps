import optuna
import time
import random
import sys
import os
from pathlib import Path

import itertools

import functions
from propulate.utils import get_default_propagator
from propulate.propagators import SelectBest, SelectWorst

from propulate import Islands
from mpi4py import MPI

#def objective(trial):
#    """Sphere benchmark function."""
#    lower_sleep = 0.5
#    upper_sleep = 30
#    a = trial.suggest_uniform('a', -5.12, 5.12)
#    b = trial.suggest_uniform('b', -5.12, 5.12)
#    time.sleep(random.uniform(lower_sleep, upper_sleep))
#    return a**2 + b**2


def variable_propulate():
    fname = os.environ["FNAME"]
    
    # assuming 72 procs on 2 nodes!!! (144 CPUs)
    # this is divisible by 4 (for the NN stuff later)
    islands = [  36]  # [  2,  4,  8, 16, 36]
    # equals  [ 72, 36, 18,  9,  4]
    migrations_prob = [0.01, 0.10, 0.30, 0.50, 0.70, 0.90]  # , 0.99]
    pollination = [True, False]
    mate_prob = [0.1, 0.325, 0.55, 0.775]
    mut_prob = [0.1, 0.325, 0.55, 0.775]
    rand_prob = [0.1, 0.325, 0.55, 0.775]
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size  # should be 144

    lst = list(itertools.product(
        islands, migrations_prob, pollination, mate_prob, mut_prob, rand_prob
    ))

    st = 0  # lst.index((36, 0.99, True, 0.1, 0.1, 0.1))

    #for isl, mig, pol, mate, mut, rand in itertools.product(
    #        islands, migrations_prob, pollination, mate_prob, mut_prob, rand_prob
    #):
    for isl, mig, pol, mate, mut, rand in lst[st:]:
        MPI.COMM_WORLD.Barrier()
        if rank == 0:
            print(f"starting islands: {isl}\tmigration: {mig}\tpollination: {pol}\tmate_prob: "
                  f"{mate}\tmut_prob: {mut}\trand_prob: {rand}")
        functions.propulate_objective(
            fname=fname,
            num_islands=isl,
            migration_prob=mig,
            pollination=pol,
            mate_prob=mate,
            mut_prob=mut,
            random_prob=0.1,
        )
        MPI.COMM_WORLD.Barrier()

        if rank == 0:
            # print(best)
            print(f"Finished islands: {isl}\tmigration: {mig}\tpollination: {pol}\tmate_prob: "
                  f"{mate}\tmut_prob: {mut}\trand_prob: {rand}")


def optimize_propulate():
    job_id = int(os.environ["SLURM_JOBID"])
    seed = int(os.environ["SEED"])
    #int(job_id) + os.getpid()
    fname = os.environ["FNAME"]
    n_trials = 1000
    func, limits = functions.get_limits(fname)
    rng = random.Random(seed + MPI.COMM_WORLD.rank)  # int(os.environ["SLURM_JOBID"]) + MPI.COMM_WORLD.rank)
    print(seed)
    # TODO: pop size, num_generations
    pop_size = MPI.COMM_WORLD.size
    num_gens = int(os.environ["EVALS_PER_WORKER"])
    islands = os.environ["SLURM_NNODES"]

    checkpoint = Path(f"/hkfs/work/workspace/scratch/qv2382-propulate/exps/function_benchmark/logs/{fname}/checkpoints/")  / os.environ["SLURM_JOBID"]
    checkpoint.mkdir(exist_ok=True, parents=True)
    checkpoint = str(checkpoint / "pop_cpt.p")

    propagator = get_default_propagator(pop_size, limits, 0.7, 0.4, 0.1, rng=rng)
    islands = Islands(
        func,
        propagator,
        generations=num_gens,
        num_isles=int(islands),
        rng=rng,
        # isle_sizes=[4, 4, 4, 4],  # migration_topology=migration_topology,
        load_checkpoint="nothing",  # pop_cpt.p",
        save_checkpoint=checkpoint,
        migration_probability=0.25,
        emigration_propagator=SelectBest,
        immigration_propagator=SelectWorst,
        pollination=True,
    )
    islands.evolve(top_n=1, logging_interval=1, DEBUG=1)


def main():
    rank = MPI.COMM_WORLD.rank

    job_id = int(os.environ["SLURM_JOBID"])
    seed = job_id  # int(os.environ["SEED"])
    #int(job_id) + os.getpid()
    fname = os.environ["FNAME"]
    n_trials = int(os.environ["EVALS_PER_WORKER"])
    framework=os.environ["FRAMEWORK"]

    print(f"framework: {framework} function: {fname}\ttrials: {n_trials}\tseed: {seed}")
    if framework == "propulate":
        return optimize_propulate()
    elif framework == "propulate-scan":
        return variable_propulate()
    # below is only optuna ==================================
    if n_trials == -1:
        n_trials = None

    # fname is the function name. the storage is the same name as the function
    storage = fname
    # study_name = os.environ["STUDY_NAME"]
    host = os.environ["SLURM_SRUN_COMM_HOST"]  #SLURM_LAUNCH_NODE_IPADDR"]  # "localhost"

    #sampler = optuna.samplers.CmaEsSampler(seed=seed)
    sampler = optuna.samplers.TPESampler(seed=seed)

    if rank == 0:
        optuna.create_study(
            study_name=f"{fname}-{seed}",  # "sphere", 
            storage=f"mysql://root:1234@{host}/{storage}", 
            sampler=sampler,
        )

    MPI.COMM_WORLD.Barrier()

    start_time = time.perf_counter()
    
    study = optuna.load_study(
        study_name=f"{fname}-{seed}",  # "sphere",
        storage=f"mysql://root:1234@{host}/{storage}",
        sampler=sampler,
    )

    study.optimize(
        lambda trial: functions.optuna_objective(trial=trial, fname=fname),
        n_trials=n_trials,
    )
    
    total_time = time.perf_counter() - start_time
    # Access the best result.
    res_str = f"Minimum objective value: {study.best_value}\n"
    res_str += f"Best parameter: {study.best_params}"
    print(res_str)

    print(f"{rank}: total_time -> {total_time}")

# Optuna calls `objective` `n_trials`  times changing the value of `x`,
# and `y` where the range of `x` and `y` is specified as [-5.12, 5.12) in 
# `trial.suggest_uniform("x", -5.12, 5.12)`.
# A `trial` is an object passed by Optuna, corresponds to a
# single call of `objective`, and provides interfaces to get
# next HPs to be tried.

# Note that `objective` is a blackbox function for Optuna.
# The library only observes the input `x` and the output
# of the function. The library gradually improves `x` with
# a smart algorithm (Bayesian optimization).

# In summary, you need these steps to set up the optimization:
# - Define OF that calculates minimization/maximization target.
# - Inside OF, set HPs to be optimized with `suggest` methods.
# - Instantiate `study` object.
# - Start optimization with `study.optimize`, specifying number
#   of trials with `n_trials`.


if __name__ == "__main__":
    main()
