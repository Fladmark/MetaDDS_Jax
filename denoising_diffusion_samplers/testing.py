# %%

from dds.configs.config import set_task, get_config
from dds.train_dds import train_dds

import numpy as onp
import torch

# %%

funnel_config = get_config()

# Time and step settings (Need to be done before calling set_task)
funnel_config.model.tfinal = 6.4
funnel_config.model.dt = 0.05

if funnel_config.model.reference_process_key == "oudstl":
    funnel_config.model.step_scheme_key = "cos_sq"

funnel_config = set_task(funnel_config, "layeb01")
funnel_config.model.reference_process_key = "oudstl"

if funnel_config.model.reference_process_key == "oudstl":
    funnel_config.model.step_scheme_key = "cos_sq"

    # Opt setting for funnel
    funnel_config.model.sigma = 1.075
    funnel_config.model.alpha = 0.6875
    funnel_config.model.m = 1.0

    # Path opt settings
    funnel_config.model.exp_dds = False

funnel_config.model.stl = False
funnel_config.model.detach_stl_drift = False

funnel_config.trainer.notebook = True
funnel_config.trainer.epochs = 11000
# Opt settings we use
# funnel_config.trainer.learning_rate = 0.0001
funnel_config.trainer.learning_rate = 5 * 10 ** (-3)
funnel_config.trainer.lr_sch_base_dec = 0.95  # For funnel

# %%

funnel_config.model.reference_process_key

# %%

funnel_config.model.input_dim

# %%

funnel_config.model.step_scheme_key

# %%

funnel_config.model.ts.shape

# %%

funnel_config.trainer.epochs = 50  # 2000
out_dict = train_dds(funnel_config)

# %%

out_dict[-1].keys()

# %%

onp.mean(out_dict[-1]["is_eval"])

# %%

onp.mean(out_dict[-1]["pf_eval"])

# %%

out_dict[-1]["pf_eval"]

# %%

funnel_config.model.reference_process_key

# %%

import matplotlib.pyplot as plt

ode_targ = out_dict[-1]["aug_ode"][:, -1, :2]
sde_targ = out_dict[-1]["aug"][:, -1, :2]

plt.plot(ode_targ[:, 0], ode_targ[:, 1], ".", alpha=0.4)
plt.plot(sde_targ[:, 0], sde_targ[:, 1], ".", alpha=0.4)

# %%

import matplotlib.pyplot as plt

timestep = 40

ode_targ = out_dict[-1]["aug_ode"][:, timestep, :2]
sde_targ = out_dict[-1]["aug"][:, timestep, :2]

plt.plot(ode_targ[:, 0], ode_targ[:, 1], ".", alpha=0.4)
plt.plot(sde_targ[:, 0], sde_targ[:, 1], ".", alpha=0.4)

# %%

cake = funnel_config.trainer.lnz_pf_estimator(
    out_dict[-1]["aug_ode"], funnel_config.model.source, funnel_config.model.target, debug=False)

# %%

cake

# %%

funnel_config.model.target(out_dict[-1]["aug_ode"][:, -1, :10]).mean()

# %%

funnel_config.model.source(out_dict[-1]["aug_ode"][:, 0, :10]).mean()

# %%

import distrax
import numpy as np

equi_normal2 = distrax.MultivariateNormalDiag(np.zeros(10), funnel_config.model.sigma * np.ones(10))

equi_normal2.log_prob(out_dict[-1]["aug_ode"][:, 0, :10]).mean()

# %%

funnel_config

# %%


