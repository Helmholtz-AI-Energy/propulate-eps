import optuna
import time
import random
import sys
import os

def objective(trial):
    x = trial.suggest_uniform('x', -2.048, 2.048)
    y = trial.suggest_uniform('y', -2.048, 2.048)
    lower_sleep = 0.5
    upper_sleep = 30
    r = random.uniform(lower_sleep, upper_sleep)
    time.sleep(r)
    return 100 * (x**2 - y)**2 + (1 - x)**2

if __name__ == "__main__":
    
    seed = int(sys.argv[1]) + os.getpid()
    n_trials = 1000
    study = optuna.load_study(
            study_name="rosenbrock", 
            storage="sqlite:///db_rosenbrock.db",
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
