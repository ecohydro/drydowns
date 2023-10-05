import os
import sys
import multiprocessing as mp
from configparser import ConfigParser
import time

import logging
log = logging.getLogger(__name__)

from Agent import Agent

def main():

    start = time.perf_counter()
    
    print("--- Initializing the model ---")

    cfg = ConfigParser()
    cfg.read("config.ini")
    
    agent = Agent(cfg=cfg)
    agent.initialize()
    
    run_mode = cfg['MODEL']['run_mode']
    print(f"--- Analysis started with {cfg['MODEL']['model_type']} model, {run_mode} mode ---")

    if run_mode == "serial":
        results = agent.run(agent.target_EASE_idx[0])
    if run_mode == "parallel":
        nprocess = int(cfg["MULTIPROCESSING"]["nprocess"])
        with mp.Pool(nprocess) as pool:
             results = list(pool.imap(agent.run, agent.target_EASE_idx))
        pool.close()
        pool.join()

    print(f"--- Finished analysis ---")
    
    try:
        agent.finalize(results)
    except NameError:
        print("No results are returned")
    
    end = time.perf_counter()
    log.debug(f"Run took : {(end - start):.6f} seconds")
    
    
if __name__ == "__main__":
    main()

        
        
        