import os
import sys
import multiprocessing as mp
from configparser import ConfigParser
import time

import logging

log = logging.getLogger(__name__)

from Agent import Agent


def main():
    """Main execution script ot run the drydown analysis"""
    start = time.perf_counter()

    print("--- Initializing the model ---")

    # _______________________________________________________________________________________________
    # Read config
    cfg = ConfigParser()
    cfg.read("config.ini")

    # Initiate agent
    agent = Agent(cfg=cfg)
    agent.initialize()

    # _______________________________________________________________________________________________
    # Define serial/parallel mode
    run_mode = cfg["MODEL"]["run_mode"]
    print(f"--- Analysis started with {run_mode} mode ---")

    # Run the model
    if run_mode == "serial":
        results = agent.run(agent.target_EASE_idx[500])
    if run_mode == "parallel":
        nprocess = int(cfg["MULTIPROCESSING"]["nprocess"])
        with mp.Pool(nprocess) as pool:
            results = list(pool.imap(agent.run, agent.target_EASE_idx))
        pool.close()
        pool.join()

    # _______________________________________________________________________________________________
    # Finalize the model
    print(f"--- Finished analysis ---")

    try:
        agent.finalize(results)
    except NameError:
        print("No results are returned")

    end = time.perf_counter()
    log.debug(f"Run took : {(end - start):.6f} seconds")


if __name__ == "__main__":
    main()
