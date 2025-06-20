import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import os
# import netket_pro as nkp
from jax.tree import structure as tree_structure
from omegaconf import DictConfig, OmegaConf

from hydra.utils import call, instantiate, get_class, get_method
from netket.callbacks import InvalidLossStopping
import json
import matplotlib.pyplot as plt

from grad_sample.tasks import Base
from grad_sample.utils import smart_instantiate
from functools import partial
import advanced_drivers as advd
from copy import copy, deepcopy

class InfidelityMC(Base):
    """
    Task class for MC infidelity minimization
    """
    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)

        self.sample_size = self.cfg.get("sample_size")
        self.is_distrib = instantiate(self.cfg.is_distrib)
        self.auto_is = self.cfg.get("auto_is", False)
        self.use_symmetries = self.cfg.get("use_symmetries", False)
        self.momentum = self.cfg.get('momentum')
        self.collect_gradient_statistics = self.cfg.get('collect_gradient_statistics', False)
        
        self.Nsample = 2**self.sample_size
        self.chunk_size = self.chunk_size_jac

        # Choose between SR and SRt automatically
        self.use_ntk = self.max_nparams > self.Nsample

        # args for sampler
        kwargs_hydra = {'hilbert': self.model.hilbert_space,
                        'graph': self.model.graph,
                        'sweep_size': self.model.hilbert_space.size,
                        'n_chains_per_rank': self.Nsample//2,
                        }
       
        self.sampler = smart_instantiate(self.cfg.sampler, kwargs_hydra)
        
        self.vs_fs = nk.vqs.FullSumState.load(cfg.initial_state_path)

        self.vstate = nk.vqs.MCState(sampler= self.sampler, 
                                        model= self.ansatz, 
                                        chunk_size= self.chunk_size, 
                                        n_samples= self.Nsample,
                                        # n_discard_per_chain = 2**6,
                                        variables=self.vs_fs.variables
                                    #  seed=0
                                    )
        
        if "LogStateVector" in self.cfg.ansatz._target_:
            self.vstate.init_parameters()
        #     seed=cfg.seed_vs,
        #     sampler_seed=cfg.seed_sampler_vs,

        self.ts_fs = nk.vqs.FullSumState.load(cfg.final_state_path)
        self.ts = copy(self.vstate)
        self.ts.variables = deepcopy(self.ts_fs.variables)

        sampling_state_distribution=self.is_distrib
        sampling_target_distribution=self.is_distrib
       
        # Save the current config to the custom path
        self.output_dir = self.base_path + f"/Infidelity_opt/L{self.model.graph.n_nodes}/{self.ansatz_name}/{self.alpha}/MC_{self.sample_size}_{self.is_distrib.q_variables['alpha'].item()}/{self.lr}_{self.diag_exp}"
        # create dir if it doesn't already exist, if not in analysis mode
        self.run_index = self.cfg.get("run_index")
        if self.run_index == None:
            run_index = 0
            while True:
                run_dir = os.path.join(self.output_dir, f"run_{run_index}")
                if not os.path.exists(run_dir):
                    os.makedirs(run_dir)
                    self.output_dir = run_dir  # Update the output_dir to point to the newly created run_N folder
                    break
                run_index += 1
        else :
            self.output_dir = self.output_dir + '/run_%d'%self.run_index
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
        
        # Create the logger with the new filename
        self.json_log = nk.logging.JsonLog(output_prefix=self.output_dir)
        
        self.mc_driver = advd.driver.InfidelityOptimizerNG(
            target_state=self.ts,
            optimizer=self.opt,
            diag_shift=self.diag_shift,
            linear_solver_fn=self.solver_fn,
            variational_state=self.vstate,
            use_ntk=self.use_ntk,
            chunk_size_bwd=self.chunk_size_jac,
            sampling_state_distribution=sampling_state_distribution,
            sampling_target_distribution=sampling_target_distribution,
            U=None,
            V=None,
        )

        self.callbacks = (InvalidLossStopping(),) +  tuple(smart_instantiate(cb, kwargs_hydra, mode='call') for cb in cfg.callback_list)

    def __call__(self):
        self.mc_driver.run(
            n_iter=self.n_iter, 
            out=self.json_log, 
            callback=self.callbacks
        )
        log_opt = self.output_dir + ".log"
        data = json.load(open(log_opt))
        #Plot
        fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
        ax.plot(data["Infidelity"]["Mean"], "o-")
        ax.plot(data['inf_fs']['iters'], data['inf_fs']['value'], label='FS infid')

        ax.set_ylim(1e-12, 1)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("Infidelity")
        ax.legend()
        plt.savefig(self.output_dir + '/training.png')