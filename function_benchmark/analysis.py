from pathlib import Path
import pandas as pd
import numpy as np
import re
import os


def load_data(exp_name):
    islands = [  2,  4,  8, 16, 36]
    # equals  [ 72, 36, 18,  9,  4]
    migrations_prob = [0.01, 0.10, 0.30, 0.50, 0.70, 0.90, 0.99]
    pollination = [True, False]
    mate_prob = [0.1, 0.325, 0.55, 0.775]
    mut_prob = [0.1, 0.325, 0.55, 0.775]
    rand_prob = [0.1, 0.325, 0.55, 0.775]

    # root =


def append_pandas_dict_from_log(logfile, outfile=None):
    with open(logfile, 'r') as log:
        loglines = log.readlines()

    df_dict = {
        "islands": [],
        "migration": [],
        "pollination": [],
        "mate_prob": [],
        "mut_prob": [],
        "rand_prob": [],
        "loss1": [],
        "gen1": [],
        "loss2": [],
        "gen2": [],
        "loss3": [],
        "gen3": [],
    }

    c = 0
    islands_to_read = 0
    for line in loglines:
        if line.startswith("  0: starting"):
            parsed_key = re.split("\s|:|\t|\n", line)
            islands = int(parsed_key[7])
            df_dict["islands"].append(islands)
            df_dict["migration"].append(float(parsed_key[10]))
            df_dict["pollination"].append(bool(parsed_key[13]))
            df_dict["mate_prob"].append(float(parsed_key[16]))
            df_dict["mut_prob"].append(float(parsed_key[19]))
            df_dict["rand_prob"].append(float(parsed_key[22]))
            islands_to_read = islands

            bestlosses = []
            bestgens = []

            c += 1
        if islands_to_read and line[5:].startswith("Island: "):
            # print(line)
            parsed_line = re.split("\s|,|}", line[5:])
            # losses: 7 14 21   gens: 10 17 24
            bestlosses.extend(
                [float(parsed_line[7]), float(parsed_line[14]), float(parsed_line[21])]
            )
            bestgens.extend([int(parsed_line[10]), int(parsed_line[17]), int(parsed_line[24])])
            islands_to_read -= 1
            if islands_to_read == 0:
                # sort list, get best 3 values + inds
                sorted_losses = bestlosses.copy()
                sorted_losses.sort()
                df_dict["loss1"].append(float(sorted_losses[0]))
                df_dict["loss2"].append(float(sorted_losses[1]))
                df_dict["loss3"].append(float(sorted_losses[2]))
                df_dict["gen1"].append(int(bestgens[bestlosses.index(sorted_losses[0])]))
                df_dict["gen2"].append(int(bestgens[bestlosses.index(sorted_losses[1])]))
                df_dict["gen3"].append(int(bestgens[bestlosses.index(sorted_losses[2])]))
    if islands_to_read > 0:  # remove last key values
        df_dict["islands"].pop()
        df_dict["migration"].pop()
        df_dict["pollination"].pop()
        df_dict["mate_prob"].pop()
        df_dict["mut_prob"].pop()
        df_dict["rand_prob"].pop()

    out_df = pd.DataFrame(df_dict)
    if os.path.exists(outfile):
        other_df = pd.read_csv(outfile)
        out_df = pd.concat([other_df, out_df])
    # TODO: remove duplicates?
    # out_df.drop_duplicates(subset=[""])
    out_df.to_csv(outfile, index=False)
    print(out_df)


if __name__ == "__main__":
    # append_pandas_dict_from_log(
    #     logfile="logs/rastrigin/propulate-1868467.out", outfile="results/rastrigin_grid.csv"
    # )
    append_pandas_dict_from_log(
        logfile="logs/rastrigin/propulate-1869494.out", outfile="results/rastrigin_grid.csv"
    )
    append_pandas_dict_from_log(
        logfile="logs/rastrigin/propulate-1870668.out", outfile="results/rastrigin_grid.csv"
    )
