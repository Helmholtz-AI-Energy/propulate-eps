import optuna
import time
import random
import sys
import os
import numpy

def objective(trial):
    a = trial.suggest_uniform('a', -600., 600.)
    b = trial.suggest_uniform('b', -600., 600.)
    c = trial.suggest_uniform('c', -600., 600.)
    d = trial.suggest_uniform('d', -600., 600.)
    e = trial.suggest_uniform('e', -600., 600.)
    f = trial.suggest_uniform('f', -600., 600.)
    g = trial.suggest_uniform('g', -600., 600.)
    h = trial.suggest_uniform('h', -600., 600.)
    i = trial.suggest_uniform('i', -600., 600.)
    j = trial.suggest_uniform('j', -600., 600.)
    lower_sleep = 0.5
    upper_sleep = 30
    vec = numpy.array([a, b, c, d, e, f, g, h, i, j])
    idx = numpy.arange(1, len(vec) + 1)
    time.sleep(random.uniform(lower_sleep, upper_sleep))
    return 1 + 1.0 / 4000 * numpy.sum(vec**2) - numpy.prod(numpy.cos(vec / numpy.sqrt(idx)))

if __name__ == "__main__":
    
    fname = str(sys.argv[1])
    job_id = sys.argv[2]
    seed = int(job_id) + os.getpid()
    n_trials = 1000
    study = optuna.load_study(
            study_name=fname, 
            storage="sqlite:///"+fname+"_"+str(job_id)+".db",
            sampler=optuna.samplers.TPESampler(seed=seed))
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
