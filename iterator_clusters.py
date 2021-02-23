import subprocess
import json
from utils.util import read_config, get_logger
import numpy as np
import argparse
import torch.cuda as cutorch
from utils.gpuutils import get_available_gpus
import time
import os
import shutil
import pprint

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--filename', default=[], help='configuration filename',
        action="append")
    parser.add_argument('--dry-run', action='store_true', help='do not fire')
    parser.add_argument('--min_clusters', type=int, default=1, help="Minimum number of clusters to try")
    parser.add_argument('--max_clusters', type=int, default=5, help="Minimum number of clusters to try")
    return parser.parse_args()

def get_fields(d):
    fields = []
    for key, value in d.items():
        if isinstance(value, dict):
            fields.extend(get_fields(value))
        else:
            fields.extend([f"--{key}", str(value)])
    return fields

if __name__ == "__main__":

    args = args_parser()
    mylogger = get_logger("Iterator")

    mylogger.debug(args)

    # Loop over multiple files
    gpus = get_available_gpus()
    number_of_gpus = len(gpus)
    mylogger.debug(f"gpus: {gpus}")

    for filename in args.filename:

        # Read config
        config = read_config(filename)
        experiment = config.pop("experiment")
        experiment_name = experiment.get("name", "default")

        # Set up output paths
        log_path = f"save/{experiment_name}"
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        # Copy experiment parameters for later reference
        shutil.copy2(filename, os.path.join(log_path,"experiment.json"))

        flags = experiment.pop("flags")

        config["runs"] = experiment.get("runs", 1)
        config["experiment"] = experiment_name

        if config["data"]["dataset"] == "femnist":
            pvals = [0]
        elif config["data"]["dataset"] == "cifar10rot":
            pvals = [0]
        else:
            pvals = np.linspace(.8, 1, 3)

        frac = config["federated"]["frac"]
        mylogger.info(f"Starting {experiment_name} from {filename} with p={pvals}")

        dataset = config["data"]["dataset"]
        model = config["model"]

        cluster_list = range(args.min_clusters, args.max_clusters + 1)

        # Make variable replacable
        child_processes = []

        for clusters in cluster_list:

            mylogger.info(f"Cluster k={clusters}")

            for n, p in enumerate(pvals):

                config["federated"]["clusters"] = clusters
                config["data"]["p"] = np.round(p / .1) * .1

                available_gpus = get_available_gpus()

                while not available_gpus:
                    time.sleep(60)

                # Add back GPU
                config["gpu"] = np.random.choice(available_gpus, 1)[0]

                mylogger.debug(f"Assigning p={p} to GPU {gpus[n % number_of_gpus]}")

                config["filename"] = "results_clusters"

                command = ["python", "main_fed.py"]

                command.extend(get_fields(config))

                for k, v in flags.items():
                    if v:
                        command.append(f"--{k}")

                mylogger.debug(" ".join(command))

                # Allow dry-runs
                if not args.dry_run:
                    p = subprocess.Popen(command, shell=False)
                    child_processes.append(p)

                time.sleep(20)

        for cp in child_processes:
            cp.wait()
