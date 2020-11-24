import subprocess
import json
from utils.util import read_config, get_logger
import numpy as np

#list_files = subprocess.run(["ls", "-l"])
#print("The exit code was: %d" % list_files.returncode)

mylogger = get_logger("Iterator")

config = read_config("config.json")

pvals = np.linspace(0,1,10+1)
mylogger.info(f"Starting experiment with p={pvals}")

child_processes = []

for n, p in enumerate(pvals):

    config["p"] = p
    config["gpu"] = n%8
    config["filename"] = f"results_{p}.csv"

    command = ["python", "main_fed.py"]

    for k,v  in config.items():
        command.extend([f"--{k}", str(v)])

    mylogger.debug(command)

    p = subprocess.Popen(command)
    child_processes.append(p)

for cp in child_processes:
    cp.wait()
