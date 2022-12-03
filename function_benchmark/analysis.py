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


def mangled_log_to_pandas(logfile, outfile=None, overwrite=False):
    print(f"Active file: {logfile}")
    with open(logfile, 'r') as log:
        loglines = log.readlines()
    df_dict = {
        "islands": [],
        "migration": [],
        "pollination": [],
        "mate_prob": [],
        "mut_prob": [],
        "rand_prob": [],
        "avg_loss": [],
        "avg_gen": [],
        "std_loss": [],
        "std_gen": [],
    }

    islands_dict = {
        "loaded_islands": [],
        # "island_number": {
        #     "loaded_losses": [],
        #     "loaded_gens": [],
        # },
    }

    c = 0
    times = {2: [], 4: [], 8: [], 16: [], 32: [], 36: [], }
    # collect all island results and all parameters
    for en, line in enumerate(loglines):
        if line.startswith("  0: starting"):
            # try to get the time of the last run
            parsed_key = re.split("\s|:|\t|\n", line)
            islands = int(parsed_key[7])
            df_dict["islands"].append(islands)  # number of islands
            df_dict["migration"].append(float(parsed_key[10]))
            df_dict["pollination"].append(True if parsed_key[13] == "True" else False)
            df_dict["mate_prob"].append(float(parsed_key[16]))
            df_dict["mut_prob"].append(float(parsed_key[19]))
            df_dict["rand_prob"].append(float(parsed_key[22]))
            islands_dict["loaded_islands"].append(islands)

            # if c == 544 or c == 608:
            #     print(parsed_key[13], bool(parsed_key[13]))

            last_islands = islands
            c += 1
        # todo: fix me to l ook for the specific islands
        if line[5:].startswith("Island: "):
            parsed_line = re.split("\s|,|}", line[5:])
            island_number = int(parsed_line[1])
            # losses: 7 14 21   gens: 10 17 24
            rets = [
                np.array(
                    [float(parsed_line[7]), float(parsed_line[14]), float(parsed_line[21])],
                    # np.float
                ),
                np.array([int(parsed_line[10]), int(parsed_line[17]), int(parsed_line[24])])
            ]
            # islands_dict["loaded_islands"].append(rets)
            if island_number not in islands_dict.keys():
                islands_dict[island_number] = {
                    "losses": [],
                    "gens": [],
                }
            islands_dict[island_number]["losses"].append(rets[0])
            islands_dict[island_number]["gens"].append(rets[1])
        parsed_key = re.split("\s|:|\t|\n", line)
        if "time" in parsed_key and parsed_key[-5] == "time":
            # possible error here: last_islands may not be defined. but the logs SHOULD have
            # '  0: starting' before this, so it should be okay...
            times[last_islands].append(float(parsed_key[-2]))

    st = 0
    for r in range(len(islands_dict["loaded_islands"])):
        # have the island sizes, just need to load the island results
        to_load = islands_dict["loaded_islands"].pop(0)
        try:
            losses = np.concatenate(
                [islands_dict[n]["losses"].pop(0) for n in range(to_load)],

            )
            gens = np.concatenate(
                [islands_dict[n]["gens"].pop(0) for n in range(to_load)],

            )
        except IndexError:
            # if a run which didn't finish, remove the params from the other lists
            df_dict["islands"].pop()
            df_dict["migration"].pop()
            df_dict["pollination"].pop()
            df_dict["mate_prob"].pop()
            df_dict["mut_prob"].pop()
            df_dict["rand_prob"].pop()
            break

        # probably best to get the average loss of the top 3 here...
        best_losses_inds = np.argpartition(losses, 5)
        avg_losses = np.mean(losses[best_losses_inds])
        avg_gens = np.mean(gens[best_losses_inds])
        std_losses = np.std(losses[best_losses_inds], ddof=0)
        std_gens = np.std(gens[best_losses_inds], ddof=0)
        df_dict["avg_loss"].append(avg_losses)
        df_dict["std_loss"].append(std_losses)
        df_dict["avg_gen"].append(avg_gens)
        df_dict["std_gen"].append(std_gens)

        # print(losses)
        # return

    for k in times:
        # print(times[k])
        if len(times[k]) > 0:
            print(f"islands: {k} times:\taverage: {np.mean(times[k]):.5f}\tstd: "
                  f"{np.std(times[k]):.5f}\tmax: {np.max(times[k]):.5f}\tmin: "
                  f"{np.min(times[k]):.5f}")

    min_length = 100000000
    for k in df_dict:
        if len(df_dict[k]) < min_length:
            min_length = len(df_dict[k])
        print(f"{k}\t{len(df_dict[k])}")

    for k in df_dict:
        df_dict[k] = df_dict[k][:min_length]

    out_df = pd.DataFrame(df_dict)
    if os.path.exists(outfile) and not overwrite:
        other_df = pd.read_csv(outfile)
        out_df = pd.concat([other_df, out_df])
    # TODO: remove duplicates?
    # out_df.drop_duplicates(subset=[""])
    out_df.to_csv(outfile, index=False)
    # print(out_df)
    return out_df


def append_pandas_dict_from_log(logfile, outfile=None):
    print(f"Active file: {logfile}")
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
    next_islands_to_read = 0
    times = {2: [], 4: [], 8: [], 16: [], 32: [], 36: [], }
    # collect all island results and all parameters
    for en, line in enumerate(loglines):
        if line.startswith("  0: starting"):
            # try to get the time of the last run
            if c > 0:
                parsed_key = re.split("\s|:|\t|\n", loglines[en - 2])
                if parsed_key[-5] == "time":
                    times[last_islands].append(float(parsed_key[-2]))

            parsed_key = re.split("\s|:|\t|\n", line)
            islands = int(parsed_key[7])
            df_dict["islands"].append(islands)
            df_dict["migration"].append(float(parsed_key[10]))
            df_dict["pollination"].append(bool(parsed_key[13]))
            df_dict["mate_prob"].append(float(parsed_key[16]))
            df_dict["mut_prob"].append(float(parsed_key[19]))
            df_dict["rand_prob"].append(float(parsed_key[22]))
            if islands_to_read > 0:
                next_islands_to_read = islands
            else:
                islands_to_read = islands

            last_islands = islands

            bestlosses = []
            bestgens = []

            c += 1
        # todo: fix me to l ook for the specific islands
        if islands_to_read and line[5:].startswith("Island: "):
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
                if next_islands_to_read > 0:
                    islands_to_read = next_islands_to_read
                    next_islands_to_read = 0

    if islands_to_read > 0:  # remove last key values
        df_dict["islands"].pop()
        df_dict["migration"].pop()
        df_dict["pollination"].pop()
        df_dict["mate_prob"].pop()
        df_dict["mut_prob"].pop()
        df_dict["rand_prob"].pop()

    for k in times:
        # print(times[k])
        if len(times[k]) > 0:
            print(k, np.mean(times[k]), np.std(times[k]), np.max(times[k]), np.min(times[k]))

    # for k in df_dict:
    #     print(f"{k}\t{len(df_dict[k])}")

    out_df = pd.DataFrame(df_dict)
    # if os.path.exists(outfile):
    #     other_df = pd.read_csv(outfile)
    #     out_df = pd.concat([other_df, out_df])
    # # TODO: remove duplicates?
    # # out_df.drop_duplicates(subset=[""])
    # out_df.to_csv(outfile, index=False)
    # print(out_df)


def get_filenames(exp, trial):
    propualte_exps = {
        "birastrigin": {
            1: ["propulate-1868350.out", "propulate-1869493.out", "propulate-1870669.out"],
            2: ["propulate-2-islands36-1877310.out",
                         "propulate-2-islands36-1877313.out",
                "propulate-2-islands36-1877316.out", "propulate-2-islands36-1877319.out",
                "propulate-2-islands36-1879603.out"],
            3: ["propulate-2-424242-1879664.out","propulate-2-424242-1879667.out",
                "propulate-2-424242-1879670.out", "propulate-2-424242-1879673.out",
                "propulate-2-424242-1879676.out"],
            4: ["propulate-3-1881118.out", "propulate-3-1881119.out",
                "propulate-3-1881120.out", "propulate-3-1881121.out",
                "propulate-3-1881122.out"
                ],
            5: [
                "propulate-3-1881123.out",
                "propulate-3-1881124.out",
                "propulate-3-1881125.out",
                "propulate-3-1881126.out",
                "propulate-3-1881127.out"
                ],
        },
        "quartic": {
            # 1: ["propulate-1867833.out", "propulate-1869492.out", "propulate-1870667.out"],
            2: ["propulate-2-1877311.out", "propulate-2-1877314.out",
                "propulate-2-1877317.out", "propulate-2-1877320.out",
                "propulate-2-1877323.out"],
            3: ["propulate-2-424242-1879665.out", "propulate-2-424242-1879668.out",
                "propulate-2-424242-1879671.out", "propulate-2-424242-1879674.out",
                "propulate-2-424242-1879677.out"],
            4: ["propulate-3-1880644.out", "propulate-3-1880647.out",
                "propulate-3-1880650.out", "propulate-3-1880653.out",
                "propulate-3-1880656.out"
                ],
            5: ["propulate-3-1880659.out", "propulate-3-1880662.out",
                "propulate-3-1880665.out", "propulate-3-1880668.out",
                "propulate-3-1880671.out"
                ],
        },
        "rastrigin": {
            1: ["propulate-1868467.out", "propulate-1869494.out", "propulate-1870668.out"],
            2: ["propulate-2-islands36-1877312.out",
                "propulate-2-islands36-1877315.out",
                "propulate-2-islands36-1877318.out",
                "propulate-2-islands36-1877321.out",
                "propulate-2-islands36-1877324.out"],
            3: ["propulate-2-424242-1879666.out",
                "propulate-2-424242-1879669.out",
                "propulate-2-424242-1879672.out",
                "propulate-2-424242-1879675.out",
                "propulate-2-424242-1879678.out"],
            4: ["propulate-3-1880645.out", "propulate-3-1880648.out",
                "propulate-3-1880651.out", "propulate-3-1880654.out",
                "propulate-3-1880657.out"
                ],
            5: ["propulate-3-1880660.out", "propulate-3-1880663.out",
                "propulate-3-1880666.out", "propulate-3-1880669.out",
                "propulate-3-1880672.out"
                ],
        },
    }
    return propualte_exps[exp][trial]


if __name__ == "__main__":
    exp = "quartic"
    # trial = 2
    for trial in range(2, 6):
        outfile = f"results/{exp}_grid_{trial}.csv"
        files = get_filenames(exp, trial)

        print(files)
        overwrite = True
        for f in files:
            logfile = f"logs/{exp}/{f}"
            # append_pandas_dict_from_log(logfile=logfile, outfile=outfile)
            full_df = mangled_log_to_pandas(logfile=logfile, outfile=outfile, overwrite=overwrite)
            overwrite = False
            # print(full_df[544:609])
            # print(
            #     full_df[
            #         (full_df["islands"] == 8) & (full_df["migration"] == 0.7) & (
            #                     full_df["pollination"] == True) &
            #         (full_df["mate_prob"] == 0.55) & (full_df["mut_prob"] == 0.1) & (
            #                     full_df["rand_prob"] == 0.1)
            #         ]
            # )

    # append_pandas_dict_from_log(
    #     logfile="logs/rastrigin/propulate-1868467.out", outfile="results/rastrigin_grid.csv"
    # )
    # append_pandas_dict_from_log(
    #     logfile="logs/rastrigin/propulate-1869494.out", outfile="results/rastrigin_grid.csv"
    # )
    # append_pandas_dict_from_log(
    #     logfile="logs/rastrigin/propulate-1870668.out", outfile="results/rastrigin_grid.csv"
    # )
