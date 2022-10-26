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
    time.sleep(random.uniform(lower_sleep, upper_sleep))
    return 100 * (x**2 - y)**2 + (1 - x)**2

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
