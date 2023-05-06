from dds.configs.config import set_task, get_config
#from dds.train_dds import train_dds

import numpy as onp
from dds.targets.toy_targets import *
from dds.train_dds_anneal import train_dds_anneal

# %%
funnel_config = get_config()


# Time and step settings (Need to be done before calling set_task)
funnel_config.model.tfinal = 6.4
funnel_config.model.dt = 0.05  # 0.05

if funnel_config.model.reference_process_key == "oudstl":
    funnel_config.model.step_scheme_key = "linear"

from dds.targets.toy_targets import get_attr

### SET TASK
task = "breastcancer"
div = 1000
c = 100000

# div, e, other_dim = get_attr()
# div = str(div).replace(".", "")
# e = str(e).replace(".", "")
#
# save_name = f"{task}_s{div}_plus{e}_od{other_dim}"

funnel_config = set_task(funnel_config, task, div, c)
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
    funnel_config.model.step_scheme_key = "last_small"
    # Opt setting for funnel
    funnel_config.model.sigma = 1.075
    funnel_config.model.alpha = 0.6875
    funnel_config.model.m = 1.0

    # Path opt settings    
    funnel_config.model.exp_dds = False

funnel_config.model.stl = False
funnel_config.model.detach_stl_drift = False

# funnel_config.model.stl = True
# funnel_config.model.detach_stl_drift = True

funnel_config.trainer.notebook = True
funnel_config.trainer.epochs = 11000
# Opt settings we use
# funnel_config.trainer.learning_rate = 0.0001
funnel_config.trainer.learning_rate = 1 * 10 ** (-3)
funnel_config.trainer.lr_sch_base_dec = 0.90  # For funnel
# %%
funnel_config.model.reference_process_key
# %%
input_dim = funnel_config.model.input_dim
# %%
funnel_config.model.step_scheme_key
# %%
funnel_config.model.ts.shape
# %%


funnel_config.model.input_dim = 171
funnel_config.trainer.epochs = 51

# divs = [1000, 100, 50, 25, 10]
# epochs = [51, 151, 301, 301, 301]
# lrs = [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]

epochs = [51, 301, 301]
divs = [1, 100, 10]
lrs = [0.001, 0.001, 0.001]

eval = False
pretrained = None
iterations = 20
for i in range(iterations):
    if iterations == i + 1:
        eval = True
    funnel_config.trainer.epochs = epochs[i]
    funnel_config.model.target_class.div = divs[i]
    funnel_config.trainer.learning_rate = lrs[i]
    params, model_state, forward_fn_wrap, rng_key, results_dict, [trainable_params, non_trainable_params] = train_dds_anneal(funnel_config, pretrained, eval)
    pretrained = [trainable_params, non_trainable_params]

    print("\n\n")


print(trainable_params)
print("\n\n\n\n")
print(non_trainable_params)