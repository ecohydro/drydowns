import multiprocessing as mp
from configparser import ConfigParser
import time

from Agent import Agent
from MyLogger import getLogger

# Create a logger
log = getLogger(__name__)


def main():
    """Main execution script ot run the drydown analysis"""
    start = time.perf_counter()

    log.info("--- Initializing the model ---")

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
    log.info(f"--- Analysis started with {run_mode} mode ---")

    # Run the model
    if run_mode == "serial":
        results = agent.run(agent.target_EASE_idx[500])
    elif run_mode == "parallel":
        nprocess = int(cfg["MULTIPROCESSING"]["nprocess"])
        with mp.Pool(nprocess) as pool:
            results = list(pool.imap(agent.run, agent.target_EASE_idx))
        pool.close()
        pool.join()
    else:
        log.info(
            "run_mode in config is invalid: should be either 'serial' or 'parallel'"
        )

    # _______________________________________________________________________________________________
    # Finalize the model
    log.info(f"--- Finished analysis ---")

    if not results:
        log.info("No results are returned")
    else:
        try:
            agent.finalize(results)
        except NameError:
            log.info("No results are returned")

    end = time.perf_counter()
    log.debug(f"Run took : {(end - start):.6f} seconds")


if __name__ == "__main__":
    main()
