from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate
import os
import json
import netket as nk
import optax
import flax
import jax.numpy as jnp
import numpy as np
import jax

from grad_sample.utils import e_diag
from typing import Sequence

def to_sequence(arg):
    # tranforms arguments into sequences if they're just single values
    if not isinstance(arg, Sequence):
        return (arg,)
    else:
        return arg
    
class Base:
    def __init__(self, cfg : DictConfig):
        self.cfg = deepcopy(cfg)
        nk.config.netket_enable_x64 = True

        # Instantiate model class (Ising, Heisenberg...)
        self.model = instantiate(self.cfg.model)

        # Instantiate ansatz
        if "LogStateVector" in self.cfg.ansatz._target_:
            self.mode = "holomorphic"
            self.ansatz = instantiate(self.cfg.ansatz, hilbert = self.model.hilbert_space)
            self.alpha = 0

        elif "CNN" in self.cfg.ansatz._target_:
            self.mode = "complex" 
            self.ansatz = instantiate(self.cfg.ansatz, lattice=self.model.graph)
            self.alpha = len(self.cfg.ansatz.channels)

        elif "ViT" in self.cfg.ansatz._target_:
            # only works with rajah's model in models/system !
            self.ansatz = call(self.cfg.ansatz, system = self.model).network
            self.alpha = f"{self.cfg.ansatz.depth}_{self.cfg.ansatz.d_model}_{self.cfg.ansatz.heads}" #unique identifier for vit
            self.mode = "complex"

        elif 'LogNeuralBackflow' in self.cfg.ansatz._target_:
            self.ansatz = instantiate(self.cfg.ansatz, hilbert = self.model.hilbert_space)
            self.alpha = self.cfg.ansatz.hidden_units
            self.mode = 'complex'

        elif 'RBM' in self.cfg.ansatz._target_:
            self.ansatz = instantiate(self.cfg.ansatz)
            self.alpha = self.ansatz.alpha
            if self.cfg.ansatz.param_dtype == 'complex':
                self.mode = 'holomorphic'
            else :
                self.mode = 'real'
            

        dict_name = {"netket.models.RBM": "RBM",       
        "netket.models.LogStateVector": "log_state",
         "netket.experimental.models.LSTMNet": "RNN",
         "grad_sample.ansatz.cnn.CNN": "CNN",
         'deepnets.net.ptvmc.CNN': 'CNN',
         "deepnets.net.ViT2D": "ViT2D",
         "deepnets.net.ViT1D": "ViT1D",
         'netket.models.MLP': 'MLP',
         'grad_sample.ansatz.nnbf.LogNeuralBackflow' : 'NNBF'}
         
        self.ansatz_name = dict_name[self.cfg.ansatz._target_]

        # set hparams and relevant variables
        self.solver_fn = call(self.cfg.solver_fn)
        
        self.n_iter = self.cfg.get('n_iter')
        
        self.chunk_size_jac = self.cfg.get("chunk_size_jac")
        self.chunk_size_vmap = self.cfg.get("chunk_size_vmap")
        self.chunk_size_vstate = self.cfg.get('chunk_size_vstate', None)

        self.save_every = self.cfg.get("save_every")
        self.ckpt = self.cfg.get('ckpt')
        
        self.base_path = '././out_log/'

        n_iter = self.n_iter
        try:
            self.diag_shift = call(self.cfg.diag_shift)
            self.diag_exp = self.diag_shift
        except:
            self.diag_shift = self.cfg.get('diag_shift')
            self.diag_exp = self.diag_shift

        try:
            self.lr = call(self.cfg.lr)
        except:
            self.lr = self.cfg.get('lr')

        if self.diag_shift == 'schedule':
            start_diag_shift, end_diag_shift = 1e-2, 1e-4

            # Define a linear schedule for diag_shift using optax
            self.diag_shift = optax.cosine_decay_schedule(
                init_value=start_diag_shift,
                decay_steps=400,
                alpha=0.001
            ) #moves the diagonal shift from 1e-2 to 1e-4

            self.diag_shift = optax.linear_schedule(
                init_value = 5e-3,
                end_value = 1e-4,
                transition_steps = 500
            )
            self.diag_shift = optax.linear_schedule(
                init_value = 5e-3,
                end_value = 1e-5,
                transition_steps = 500
            )
        if self.lr == "schedule":
            lr_schedule = optax.cosine_decay_schedule(
                                                    init_value=3e-3,
                                                    decay_steps=3000,
                                                    alpha=0.1
                                                ) #moves the lr from 1e-3 to 1e-4
            # for qchem, good one used first
            
            lr_schedule = optax.linear_schedule(
                                        init_value=0.1,
                                        transition_steps=self.n_iter//4,
                                        end_value=0.01
                                        )
            self.opt = optax.sgd(learning_rate=lr_schedule)
        
        # define optimizer
        self.opt = optax.sgd(learning_rate=self.lr)
        # self.opt = optax.inject_hyperparams(optax.sgd)(learning_rate=self.lr) #used with autodiagshift

        rng_key_pars = jax.random.PRNGKey(np.random.randint(10000))
        # rng_key_pars = jax.random.PRNGKey(5)
        params = self.ansatz.init(
            rng_key_pars, jnp.zeros((1, self.model.graph.n_nodes))
        )
        self.max_nparams = nk.jax.tree_size(params)
        print(f"Nparams = {self.max_nparams}")
    
        # code only support default and overdispersed distribution for naming right now
        if hasattr(self.model, 'E_fci') and self.model.E_fci is not None:
            self.E_gs = self.model.E_fci
        else:
            try :
                try:
                    self.e_dict = self.ref_energies[self.model.name][str(self.model.h)][str(int(self.model.graph.n_nodes**(1/self.model.graph.ndim)))]
                    if 'exact' in self.e_dict.keys():
                        self.E_ref = self.e_dict['exact']
                    
                except:
                    self.E_gs = e_diag(self.model.hamiltonian.to_sparse())
                    print("The ground state energy is:", self.E_gs)
            except : 
                self.E_gs = None
                print('Hilbert space too large for exact diag, loading reference energy from litterature')
                self.ref_energies = json.load(open("../../energy_ref_litt.json"))
            
                self.e_dict = self.ref_energies[self.model.name][str(self.model.h)][str(int(self.model.graph.n_nodes**(1/self.model.graph.ndim)))]
                if 'exact' in self.e_dict.keys():
                    self.E_ref = self.e_dict['exact']
                elif 'qmc' in self.e_dict.keys():
                    self.E_ref = self.e_dict['qmc']
                elif 'rbm+pp' in self.e_dict.keys():
                    self.E_ref = self.e_dict['rbm+pp']
                else :
                    self.E_ref = self.e_dict['aochen']
                print('Ref energy %.4f'%(self.E_ref*self.model.graph.n_nodes*4))
                self.E_gs = self.E_ref*self.model.graph.n_nodes*4

        self.E_gs_per_site = self.E_gs/self.model.graph.n_nodes/4
        

        
    
