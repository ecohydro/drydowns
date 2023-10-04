import os
import sys
import multiprocessing as mp
from configparser import ConfigParser
import time
from tqdm import tqdm

import logging
log = logging.getLogger(__name__)

from Agent import Agent

def map_with_progress(func, iterable, num_processes):
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(func, iterable), total=len(iterable)))
    pool.close()
    pool.join()
    return results

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
        results = map_with_progress(agent.run, agent.target_EASE_idx, nprocess)

    print(f"--- Finished analysis ---")
    
    try:
        agent.finalize(results)
    except NameError:
        print("No results are returned")
    
    end = time.perf_counter()
    log.debug(f"Run took : {(end - start):.6f} seconds")
    
    
if __name__ == "__main__":
    main()

        
        
        