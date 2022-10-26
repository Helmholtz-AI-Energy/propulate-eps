import optuna
import sys

if __name__ == "__main__":
    function = str(sys.argv[1])
    job_id = str(sys.argv[2])
    storage = "sqlite:///"+function+"_"+job_id+".db"
    study = optuna.create_study(
            direction="minimize", 
            study_name=function, 
            storage=storage)
