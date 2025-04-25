"""
Script for running the experiments.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="../../conf", config_name="base")
def main(cfg: DictConfig):
    logging.info("---------------------------------------------------------------")
    envs = {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "PYTHONOPTIMIZE"]}
    logging.info("Env:\n%s", yaml.dump(envs))

    # Log overrides
    hydra_config = HydraConfig.get()
    logging.info("Command line args:\n%s", "\n".join(hydra_config.overrides.task))

    # Setup dir
    # OmegaConf.set_struct(cfg, False)
    # out_dir = Path(hydra_config.runtime.output_dir).absolute()
    # logging.info("Hydra output path: %s", out_dir)
    # if not cfg.get("out_dir"):
    #     cfg.out_dir = str(out_dir)
    # logging.info("Solver output path: %s", cfg.out_dir)
   
    OmegaConf.set_struct(cfg, True)
    
    # Log config and overrides
    logging.info("---------------------------------------------------------------")
    logging.info("Run config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logging.info("---------------------------------------------------------------")

    try:
        # take any task from cfg and run it
        device = cfg.get("device")
        # set working device
        if device != None :
            os.environ["CUDA_VISIBLE_DEVICES"]= str(device)
        task = instantiate(cfg.task, cfg)
        task()

    except Exception as e:
        logging.critical(e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
