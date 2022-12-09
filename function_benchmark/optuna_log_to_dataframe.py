import re
import pandas as pd
import optuna
from pathlib import Path
import time

def save_trial_to_csv(fname, jid, out_dir):
    try:
        study = optuna.load_study(
            study_name=f"{fname}-{jid}", storage=f"mysql://root@localhost/{fname}_{jid}"
        )
        trials_df = study.trials_dataframe()
    except:  # try again...done know why
        print("filed to open trial....sleeping for 5 and trying again")
        time.sleep(5)
        study = optuna.load_study(
            study_name=f"{fname}-{jid}", storage=f"mysql://root@localhost/{fname}_{jid}"
        )
        trials_df = study.trials_dataframe()

    trials_df.to_csv(str(out_dir / f"{fname}-{jid}.csv"))


if __name__ == "__main__":
    jobs = pd.read_csv("optuna_runs.csv", sep="\t")
    functions = jobs["function"].unique()
    out = Path("/hkfs/work/workspace/scratch/qv2382-propulate/exps/function_benchmark/results/")
    out.mkdir(exist_ok=True)
    for fname in functions:
        fout = out / fname
        fout.mkdir(exist_ok=True)
        for jid in jobs[jobs["function"] == fname]["jobid"]:
            # print(f"logs/paper/{fname}/optuna-{jid}.out")
            save_trial_to_csv(f"logs/paper/{fname}/optuna-{jid}.out")
            # break
        # break
