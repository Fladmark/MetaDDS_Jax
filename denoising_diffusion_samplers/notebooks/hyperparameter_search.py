from dds.configs.config import set_task, get_config
from dds.train_dds import train_dds
from utility_func import *
import numpy as onp

"""
HYPERPARAMETERS:

sigma:
alpha:
div:

epochs:
samples:

"""



funnel_config = get_config()

# Time and step settings (Need to be done before calling set_task)
funnel_config.model.tfinal = 6.4
funnel_config.model.dt = 0.05  # 0.05

if funnel_config.model.reference_process_key == "oudstl":
    funnel_config.model.step_scheme_key = "cos_sq"

from dds.targets.toy_targets import get_attr

### SET TASK
task = "xor"
div, e, other_dim = get_attr()
div = str(div).replace(".", "")
e = str(e).replace(".", "")

save_name = f"{task}_s{div}_plus{e}_od{other_dim}"

funnel_config = set_task(funnel_config, task)
funnel_config.model.reference_process_key = "oudstl"

# funnel_config.model.reference_process_key = "pisstl"
# funnel_config.model.step_scheme_key = "uniform"

# exp_dec
# cos_sq
# uniform
# last_small
# linear_dds
# linear
# uniform_dds

if funnel_config.model.reference_process_key == "oudstl":
    funnel_config.model.step_scheme_key = "cos_sq"

    # Opt setting for funnel
    funnel_config.model.sigma = 1.075
    funnel_config.model.alpha = 0.6875
    funnel_config.model.m = 1.0

    # Path opt settings
    funnel_config.model.exp_dds = False

# funnel_config.model.stl = False
# funnel_config.model.detach_stl_drift = False

funnel_config.model.stl = True
funnel_config.model.detach_stl_drift = True

funnel_config.trainer.notebook = True
funnel_config.trainer.epochs = 11000
# Opt settings we use
# funnel_config.trainer.learning_rate = 0.0001
funnel_config.trainer.learning_rate = 5 * 10 ** (-3)
funnel_config.trainer.lr_sch_base_dec = 0.95  # For funnel
# %%
funnel_config.model.reference_process_key
# %%
input_dim = funnel_config.model.input_dim
# %%
funnel_config.model.step_scheme_key
# %%
funnel_config.model.ts.shape
# %%
funnel_config.trainer.epochs = 3000
out_dict = train_dds(funnel_config)