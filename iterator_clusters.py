import subprocess
import json
from utils.util import read_config, get_logger
import numpy as np
import argparse
import torch.cuda as cutorch
from utils.gpuutils import get_available_gpus
import time

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--filename', default=[], help='configuration filename',
        action="append")
    parser.add_argument('--dry-run', action='store_true', help='do not fire')
    parser.add_argument('--experiment',
                        type=str, default='result', help='output path')
    return parser.parse_args()


if __name__ == "__main__":

    args = args_parser()
    mylogger = get_logger("Iterator")

    mylogger.debug(args)
    # Loop over multiple files

    gpus = get_available_gpus()
    number_of_gpus = len(gpus)
    mylogger.debug(f"gpus: {gpus}")

    for filename in args.filename:

        config = read_config(filename)
        flags = config.pop("flags")
        # for clusters in range(1, config["clusters"] + 1  )

        if config["dataset"] == "femnist":
            pvals = [0]
        else:
            pvals = np.linspace(.2, 1, 9)

        frac = config["frac"]
        mylogger.info(f"Starting experiment from {filename} with p={pvals}")

        dataset = config["dataset"]
        model = config["model"]

        cluster_list = range(2, 4 + 1)

        # Make variable replacable
        for clusters in cluster_list:
            #config["frac"] = clusters * frac
            mylogger.info(f"Cluster k={clusters}")
            child_processes = []

            for n, p in enumerate(pvals):

                config["clusters"] = clusters
                config["p"] = np.round(p / .1) * .1

                available_gpus = get_available_gpus()

                if not available_gpus:
                    time.sleep(60)

                config["gpu"] = np.random.choice(available_gpus, 1)[0]

                mylogger.debug(f"Assigning p={p} to GPU {gpus[n % number_of_gpus]}")

                config["filename"] = "results_clusters"

                command = ["python", "main_fed.py"]

                for k, v in config.items():
                    command.extend([f"--{k}", str(v)])

                for k, v in flags.items():
                    if v:
                        command.append(f"--{k}")

                command.extend(["--experiment", args.experiment])

                mylogger.debug(" ".join(command))

                # Allow dry-runs
                if not args.dry_run:
                    p = subprocess.Popen(command)
                    child_processes.append(p)

                time.sleep(20)

            for cp in child_processes:
                cp.wait()
