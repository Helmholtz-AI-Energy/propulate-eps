import re
import pandas as pd
import optuna
from pathlib import Path
import time

def save_trial_to_csv(fname, jid, out_dir):
    if (out_dir / f'{fname}-{jid}.csv').is_file():
        print(out_dir / f'{fname}-{jid}.csv', 'exists, exiting')
        return
    try:
        study = optuna.load_study(
            study_name=f"{fname}-{jid}", storage=f"mysql://root@localhost/{fname}_{jid}"
        )
        trials_df = study.trials_dataframe()
    except:  # try again...done know why
        print("failed to open trial....sleeping for 5 and trying again")
        time.sleep(5)
        try:
            study = optuna.load_study(
                study_name=f"{fname}-{jid}", storage=f"mysql://root@localhost/{fname}_{jid}"
            )
            trials_df = study.trials_dataframe()
        except Exception as e:
            print("failed again with:", e)
            return

    trials_df.to_csv(str(out_dir / f"{fname}-{jid}.csv"))
    print(f"saved {out_dir / f'{fname}-{jid}.csv'}")


if __name__ == "__main__":
    jobs = pd.read_csv("optuna_runs.csv", sep="\t")
    functions = jobs["function"].unique()
    out = Path("/hkfs/work/workspace/scratch/qv2382-propulate/exps/function_benchmark/results/")
    out.mkdir(exist_ok=True)
    for fname in ["birastrigin"]:  #functions:
        fout = out / fname
        print("creating", fout)
        fout.mkdir(exist_ok=True)
        #print(jobs[jobs["function"] == fname])
        for jid in jobs[jobs["function"] == fname]["jobid"]:
            print(fname, jid)
            # print(f"logs/paper/{fname}/optuna-{jid}.out")
            save_trial_to_csv(fname=fname, jid=jid, out_dir=fout)
            # break
        # break
