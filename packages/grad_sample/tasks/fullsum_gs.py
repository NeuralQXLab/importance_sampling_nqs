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
from functools import partial
import advanced_drivers as advd

class FullSumGS(Base):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.graph.n_nodes}/{self.ansatz_name}/{self.alpha}/{self.lr}_{self.diag_exp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(self.output_dir)
        # Save the current config to the custom path
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
        if self.chunk_size_jac < self.model.hilbert_space.n_states:
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
        

        def __call__(self):
            print('calling run')
            self.gs.run(n_iter=self.n_iter, out=self.out_log, callback=self.callbacks)
            
            log_opt = self.output_dir + ".log"
            data = json.load(open(log_opt))

            E = jnp.array(data["Energy"]["Mean"]["real"])
            
            if self.plot_training_curve and (self.E_gs != None):
                plt.plot(jnp.abs(E-self.E_gs)/jnp.abs(self.E_gs), label= "FS")
                plt.title(f"Relative error w.r.t. exact GS during training")
                plt.xlabel("iteration")
                plt.ylabel("Relative error")
                plt.yscale("log")