import multiprocessing as mp
from configparser import ConfigParser
from .config import Config
import time

from .agent import Agent
from .toweragent import TowerAgent

from .mylogger import getLogger
from .utils import is_true

# Create a logger
log = getLogger(__name__)


def main():
    """Main execution script ot run the drydown analysis"""
    start = time.perf_counter()
    log.info("--- Initializing the model ---")

    # _______________________________________________________________________________________________
    # Read config
    # cfg = ConfigParser()
    # cfg.read("config.ini")
    config = Config().config
    cfg = config[config.get('RUN','type')]

    # cfg_model = cfg["MODEL"]

    # Initiate agent
    # if cfg['DATA']['data_type'] != 'SMAP':
    # TODO: Get object type from config
    if cfg.name != 'SMAP':
        agent = TowerAgent(cfg=cfg)
    else:
        agent = Agent(cfg=cfg)
    agent.initialize()

    # _______________________________________________________________________________________________
    # Define serial/parallel mode
    # run_mode = cfg["MODEL"]["run_mode"]
    run_mode = cfg["run_mode"]
    log.info(f"--- Analysis started with {run_mode} mode ---")

    # _______________________________________________________________________________________________
    # Verbose models to run
    log.info(f"Running the following models:")
    # if is_true(cfg["MODEL"]["exponential_model"]):
    #     log.info(f"Exponential model")
    # if is_true(cfg["MODEL"]["q_model"]):
    #     log.info(f"q model")
    # if is_true(cfg["MODEL"]["sigmoid_model"]):
    #     log.info(f"Sigmoid model")
    for mod_name in ['exponential_model', 'q_model', 'sigmoid_model']:
        if cfg.getboolean(mod_name):
            log.info(f"{mod_name}")
    # [m for m in ['exponential_model', 'q_model', 'sigmoid_model'] if cfg_model.getboolean(m)]

    # Run the model
    if run_mode == "serial":
        results = agent.run(agent.data_ids[500])
    
    elif run_mode == "parallel":
        # nprocess = int(cfg["MULTIPROCESSING"]["nprocess"])
        # nprocess = cfg.getint("MULTIPROCESSING", "nprocess")
        nprocess = cfg.getint("nprocess")
        with mp.Pool(nprocess) as pool:
            results = list(pool.imap(agent.run, agent.data_ids))
        pool.close()
        pool.join()
    else:
        log.info(
            "run_mode in config is invalid: should be either 'serial' or 'parallel'"
        )

    # _______________________________________________________________________________________________
    # Finalize the model
    log.info(f"--- Finished analysis ---")

    if run_mode == "serial":
        if results.empty:
            log.info("No results are returned")
        else:
            try:
                agent.finalize(results)
            except NameError:
                log.info("No results are returned")

    elif run_mode == "parallel":
        if not results:
            log.info("No results are returned")
        else:
            try:
                agent.finalize(results)
            except NameError:
                log.info("No results are returned")

    end = time.perf_counter()
    log.info(f"Run took : {(end - start):.6f} seconds")


if __name__ == "__main__":
    main()
