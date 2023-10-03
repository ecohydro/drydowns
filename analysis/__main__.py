import os
import sys
import multiprocessing as mp
import configparser
import time
from tqdm import tqdm

import Model

import configParser

def map_with_progress(func, iterable, num_processes):
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(func, iterable), total=len(iterable)))
    pool.close()
    pool.join()
    return results

def main():

    start = time.perf_counter()
    
    config = configparser.ConfigParser()
    config.read("config.ini")
    
    model = Model(config=config)
    model.initialize()
    
    print("--- Analysis started ---")
    results = pool.map(model.run, smapgrid.EASEindex)
    print(f"--- Finished analysis ---")
       
    model.finalize(results)
    
    end = time.perf_counter()
    print(f"Run took : {(end - start):.6f} seconds")
    
    
if __name__ == "__main__":
    main()

        
        
        