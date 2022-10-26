import optuna
import time
import random
import sys
import os

def objective(trial):
    """Step benchmark function."""
    a = trial.suggest_uniform('a', -5.12, 5.12)
    b = trial.suggest_uniform('b', -5.12, 5.12)
    c = trial.suggest_uniform('c', -5.12, 5.12)
    d = trial.suggest_uniform('d', -5.12, 5.12)
    e = trial.suggest_uniform('e', -5.12, 5.12)
    lower_sleep = 0.5
    upper_sleep = 30
    time.sleep(random.uniform(lower_sleep, upper_sleep))
    return float(int(a) + int(b) + int(c) + int(d) + int(e))

if __name__ == "__main__":

    fname = str(sys.argv[1])
    job_id = sys.argv[2]
    seed = int(job_id) + os.getpid()
    n_trials = 1000
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
