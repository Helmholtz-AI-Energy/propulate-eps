import optuna
import time
import random
import sys
import os
import numpy

def objective(trial):
    A = 10.
    lower_sleep = 0.5
    upper_sleep = 30
   
    a = trial.suggest_float('a', -5.12, 5.12)
    b = trial.suggest_float('b', -5.12, 5.12)
    c = trial.suggest_float('c', -5.12, 5.12)
    d = trial.suggest_float('d', -5.12, 5.12)
    e = trial.suggest_float('e', -5.12, 5.12)
    f = trial.suggest_float('f', -5.12, 5.12)
    g = trial.suggest_float('g', -5.12, 5.12)
    h = trial.suggest_float('h', -5.12, 5.12)
    i = trial.suggest_float('i', -5.12, 5.12)
    j = trial.suggest_float('j', -5.12, 5.12)
    k = trial.suggest_float('k', -5.12, 5.12)
    l = trial.suggest_float('l', -5.12, 5.12)
    m = trial.suggest_float('m', -5.12, 5.12)
    n = trial.suggest_float('n', -5.12, 5.12)
    o = trial.suggest_float('o', -5.12, 5.12)
    p = trial.suggest_float('p', -5.12, 5.12)
    q = trial.suggest_float('q', -5.12, 5.12)
    r = trial.suggest_float('r', -5.12, 5.12)
    s = trial.suggest_float('s', -5.12, 5.12)
    t = trial.suggest_float('t', -5.12, 5.12)

    vec = numpy.array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t])

    rsleep = random.uniform(lower_sleep, upper_sleep)
    time.sleep(rsleep)
    #return A * len(vec) + numpy.sum(vec**2 - A * numpy.cos(2 * numpy.pi * vec))
    return numpy.sum(vec**2)

if __name__ == "__main__":
    
    fname = str(sys.argv[1])
    job_id = sys.argv[2]
    seed = int(job_id) + os.getpid()
    n_trials = 2
    study = optuna.load_study(
            study_name=fname, 
            storage="sqlite:///"+fname+"_"+str(job_id)+".db",
            sampler=optuna.samplers.CmaEsSampler(seed=seed))
    study.optimize(
            objective, 
            n_trials=n_trials)

    # Access the best result.
    res_str = f"Minimum objective value: {study.best_value}\n"
    res_str += f"Best parameter: {study.best_params}"
    print(res_str)

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
