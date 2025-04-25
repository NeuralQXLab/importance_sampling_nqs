import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import os
# import netket_pro as nkp
from jax.tree import structure as tree_structure
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate, get_class
import inspect

from netket.vqs import FullSumState
from netket.callbacks import InvalidLossStopping
import json
import matplotlib.pyplot as plt
import optax

from grad_sample.utils import save_cb
from grad_sample.tasks import Base
from grad_sample.utils import save_rel_err_fs, save_rel_err_large, compute_snr_callback, save_sampler_state
from functools import partial
import advanced_drivers as advd

def add_module(old_params: dict, new_params: dict, max_attempts: int = 10):
    """
    Modify old_params to match new_params by recursively adding the key {"module": ...} until the dictionaries match.
    If all keys of the dictionary already match at the beginning we do not attempt this.
    Returns the values of old_params with the new key structure.
    If the structures do not match after max_attempts iterations raise an error
    """
    for i in range(max_attempts):
        if tree_structure(old_params) != tree_structure(new_params):
            old_params = {"module": old_params}
        else:
            return old_params

    raise RuntimeError(
        f"Exceed maximum number of attempts to match params structures ({max_attempts})"
    )

def smart_instantiate(cfg, available_vars: dict, mode='instantiate'):
    cls = get_class(cfg._target_)
    sig = inspect.signature(cls.__init__)
    # Skip 'self'
    expected_args = set(sig.parameters) - {'self'}

    # Find matching args
    injected_args = {k: v for k, v in available_vars.items() if k in expected_args}
    if mode == 'instantiate':
        return instantiate(cfg, **injected_args)
    elif mode == 'call':
        return call(cfg, **injected_args)
    else:
        raise ValueError('invalid mode specified')

class VMC_GS(Base):
    def __init__(self, cfg: DictConfig):
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
                        'hamiltonian': self.model.hamiltonian,
                        'output_dir': self.output_dir,
                        'e_gs': self.E_gs,
                        'H_sp': self.model.hamiltonian.to_sparse()}
       
        self.sampler = smart_instantiate(self.cfg.sampler, kwargs_hydra)
        
            
        self.vstate = nk.vqs.MCState(sampler= self.sampler, 
                                        model=self.ansatz, 
                                        chunk_size= self.chunk_size, 
                                        n_samples= self.Nsample,
                                        n_discard_per_chain = 2**6
                                    #  seed=0
                                    )
        if "LogStateVector" in self.cfg.ansatz._target_:
            self.vstate.init_parameters()

        print("MC state loaded, num samples %d"%self.Nsample)
        if self.ckpt is not None:
            # with open(self.ckpt, 'rb') as f:
            #     print(f.read())  
            #     self.vstate = flax.serialization.from_bytes(self.vstate, f.read())
            try:
                with open(self.ckpt, 'rb') as f:
                    vars = nk.experimental.vqs.variables_from_file(self.ckpt,
                                                                    self.vstate.variables)
                    # update the variables of vstate with the loaded data.
                    self.vstate.variables = vars
                
                # self.vstate._sample(chain_length=2**15)
            except:
                print('bypassing checkpoint, invalid vars')

        # Save the current config to the custom path
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
        if self.is_distrib.name == 'overdispersed':
            if self.auto_is: 
                self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.graph.n_nodes}/{self.ansatz_name}/{self.alpha}/MC_{self.sample_size}_isauto/{self.lr}_{self.diag_exp}"
            else:
                self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.graph.n_nodes}/{self.ansatz_name}/{self.alpha}/MC_{self.sample_size}_{self.is_distrib.q_variables['alpha'].item()}/{self.lr}_{self.diag_exp}"
        elif self.is_distrib.name == 'default':
            self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.graph.n_nodes}/{self.ansatz_name}/{self.alpha}/MC_{self.sample_size}/{self.lr}_{self.diag_exp}"
        else:
            raise NotImplementedError()
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(self.output_dir)
        self.json_log = nk.logging.JsonLog(output_prefix=self.output_dir)
            
        self.gs = advd.driver.VMC_NG(hamiltonian=self.model.hamiltonian.to_jax_operator(), 
                                        optimizer=self.opt, 
                                        sampling_distribution=self.is_distrib,
                                        variational_state=self.vstate, 
                                        diag_shift=self.diag_shift, 
                                        auto_is=self.auto_is,
                                        use_ntk=self.use_ntk,
                                        momentum=self.momentum,
                                        collect_gradient_statistics=self.collect_gradient_statistics,
                                        on_the_fly=False)
        
        kwargs_hydra['fs_state'] = FullSumState(hilbert = self.gs.state.hilbert, 
                                        model = self.gs.state.model, 
                                        chunk_size=self.chunk_size_vstate, 
                                        seed=0)

        # self.autodiagshift = advd.callbacks.PI_controller_diagshift(diag_shift_max=0.01, diag_shift_min=1e-6, safety_fac=1.0, clip_min=0.99, clip_max=1.01)
        
        if self.save_every != None: 
            self.state_dir = self.output_dir + "/state"
            self.state_log = nk.logging.StateLog(output_prefix=self.state_dir, save_every=self.save_every)
            self.out_log = (self.json_log, self.state_log)
        else :
            self.out_log = (self.json_log,)

        self.callbacks = (InvalidLossStopping(),) +  tuple(smart_instantiate(cb, kwargs_hydra, mode='call') for cb in cfg.callbacks)

    def __call__(self):
        print('calling run')
        self.gs.run(n_iter=self.n_iter, out=self.out_log, callback=self.callbacks)
        
        log_opt = self.output_dir + ".log"
        data = json.load(open(log_opt))

        E = jnp.array(data["Energy"]["Mean"]["real"])
        
        if self.plot_training_curve and (self.E_gs != None):
            
            plt.plot(jnp.abs(E-self.E_gs)/jnp.abs(self.E_gs), label= "MC")
            
            try :
                plt.title(f"Relative error w.r.t. exact GS during training, {self.Nsample} samples")
                e_r_fs = data["rel_err"]
                plt.plot(e_r_fs["iters"], e_r_fs["value"], label= "FullSum")
            except: 
                plt.title(f"Relative error w.r.t. exact GS during training")
            plt.xlabel("iteration")
            plt.ylabel("Relative error")
            plt.yscale("log")
        
        elif self.plot_training_curve:
            
            plt.plot(E, label= "MC Energy")
            try :
                plt.title(f"Energy during training, {self.Nsample} samples")
            except: 
                plt.title(f"Energy during training")
            plt.xlabel("iteration")
            plt.ylabel("Energy")
            
        plt.legend()
        plt.savefig(self.output_dir + '/training.png')