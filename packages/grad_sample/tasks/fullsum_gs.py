from grad_sample.tasks import Base
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import os
# import netket_pro as nkp
from jax.tree import structure as tree_structure
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate

from netket.vqs import FullSumState
from netket.callbacks import InvalidLossStopping
import json
import matplotlib.pyplot as plt
import optax

from grad_sample.tasks import Base
from grad_sample.utils import smart_instantiate
from functools import partial
import advanced_drivers as advd

class FullSumGS(Base):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.graph.n_nodes}/{self.ansatz_name}/{self.alpha}/schedule_schedule"
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
        print(self.output_dir)
        # Save the current config to the custom path
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
        if self.chunk_size_jac is not None and self.chunk_size_jac < self.model.hilbert_space.n_states:
            self.chunk_size = self.model.hilbert_space.n_states // (self.model.hilbert_space.n_states//self.chunk_size_jac)
        else:
            self.chunk_size = self.model.hilbert_space.n_states
            self.chunk_size = None
            # print(self.model.hi.n_states // self.chunk_size)
            # print(self.chunk_size)
            # print(self.model.hi.n_states )
            self.vstate = nk.vqs.FullSumState(hilbert=self.model.hilbert_space, model=self.ansatz, chunk_size=self.chunk_size, seed=0)
        
        if "LogStateVector" in self.cfg.ansatz._target_:
            self.vstate.init_parameters()

        self.sr = nk.optimizer.SR(qgt=nk.optimizer.qgt.QGTJacobianDense, 
                                    solver=self.solver_fn, 
                                    diag_shift=self.diag_shift, 
                                    holomorphic= self.mode == "holomorphic")
        
        self.gs = nk.VMC(hamiltonian=self.model.hamiltonian.to_jax_operator(), 
                    optimizer=self.opt, 
                    variational_state=self.vstate, 
                    preconditioner=self.sr)
        
        self.json_log = nk.logging.JsonLog(output_prefix=self.output_dir)
        self.out_log = (self.json_log,)
        self.kwargs_hydra['fs_state'] = self.vstate
        self.kwargs_hydra['output_dir'] = self.output_dir
        self.callbacks = (InvalidLossStopping(),)
        
        self.callback_list = self.cfg.get('callback_list', None)
        if self.callback_list is not None:
            self.callbacks = (InvalidLossStopping(),) +  tuple(smart_instantiate(cb, self.kwargs_hydra, mode='call') for cb in cfg.callback_list)


    def __call__(self):
        print('calling run')
        self.gs.run(n_iter=self.n_iter, out=self.out_log, callback=self.callbacks)
        
        log_opt = self.output_dir + ".log"
        data = json.load(open(log_opt))

        E = jnp.array(data["Energy"]["Mean"]["real"])
        
        plt.plot(jnp.abs(E-self.E_gs)/jnp.abs(self.E_gs), label= "FS")
        plt.title(f"Relative error w.r.t. exact GS during training")
        plt.xlabel("iteration")
        plt.ylabel("Relative error")
        plt.yscale("log")
        plt.savefig('out.png')