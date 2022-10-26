import optuna
import sys

if __name__ == "__main__":
    function = str(sys.argv[1])
    storage = "sqlite:///db_"+function+".db"
    study = optuna.create_study(
            direction="minimize", 
            study_name=function, 
            storage=storage)
