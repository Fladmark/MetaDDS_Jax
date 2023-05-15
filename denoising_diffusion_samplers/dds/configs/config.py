"""Experiment config file for SDE samplers.

This config acts as a super config which is properly innitailised by the sub
configs in the config directory.
"""
import distrax
import jax

from jax import numpy as np

from ml_collections import config_dict as configdict

from dds.configs import brownian_config
from dds.configs import lgcp_config
from dds.configs import log_reg_config
from dds.configs import pretrained_nice_config
from dds.configs import vae_config

from dds.discretisation_schemes import cos_sq_fn_step_scheme
from dds.discretisation_schemes import exp_fn_step_scheme
from dds.discretisation_schemes import linear_step_scheme
from dds.discretisation_schemes import linear_step_scheme_dds
from dds.discretisation_schemes import small_lst_step_scheme
from dds.discretisation_schemes import uniform_step_scheme
from dds.discretisation_schemes import uniform_step_scheme_dds

from dds.drift_nets import gelu
from dds.drift_nets import PISGRADNet
from dds.drift_nets import SimpleDriftNet

from dds.drift_nets_udp import UDPPISGRADNet
from dds.drift_nets_udp import UDPSimpleDriftNet

from dds.objectives import importance_weighted_partition_estimate
from dds.objectives import ou_terminal_loss
from dds.objectives import prob_flow_lnz
from dds.objectives import relative_kl_objective

from dds.stl_samplers import AugmentedBrownianFollmerSDESTL
from dds.stl_samplers import AugmentedOUDFollmerSDESTL
from dds.stl_samplers import AugmentedOUFollmerSDESTL

from dds.targets import toy_targets

from dds.udp_samplers import AugmentedOUDFollmerSDEUDP

from experimental.breastcancer_t import breast_task


def get_config():
  """Setup base config see /configs for more details."""

  config = configdict.ConfigDict()
  config.task = "no_name"

  config.model = configdict.ConfigDict()
  config.dataset = configdict.ConfigDict()
  config.trainer = configdict.ConfigDict()

  config = configdict.ConfigDict()
  config.model = configdict.ConfigDict()
  config.dataset = configdict.ConfigDict()
  config.trainer = configdict.ConfigDict()

  config.model.fully_connected_units = [64, 64]
  # config.model.fully_connected_units = [512, 512, 512]

  config.model.learn_betas = False
  config.trainer.timer = False

  config.model.batch_size = 300  # 128
  config.model.elbo_batch_size = 2000
  config.model.terminal_cost = ou_terminal_loss
  config.model.tfinal = 5.0
  config.model.dt = 0.078

  config.model.stl = True

  config.model.tpu = True

  config.model.network_key = "pis"
  config.model.network_dict = configdict.ConfigDict()
  config.model.network_dict.pis = PISGRADNet
  config.model.network_dict.pisudp = UDPPISGRADNet
  config.model.network_dict.vanilla = SimpleDriftNet
  config.model.network_dict.vanilla_udp = UDPSimpleDriftNet

  config.model.activation_key = "gelu"
  config.model.activation_dict = configdict.ConfigDict()
  config.model.activation_dict.gelu = gelu
  config.model.activation_dict.swish = jax.nn.swish
  config.model.activation_dict.relu = jax.nn.relu

  config.model.activation = config.model.activation_dict[
      config.model.activation_key]

  config.model.step_scheme_key = "uniform"
  config.model.step_scheme_dict = configdict.ConfigDict()
  config.model.step_scheme_dict.exp_dec = exp_fn_step_scheme
  config.model.step_scheme_dict.cos_sq = cos_sq_fn_step_scheme
  config.model.step_scheme_dict.uniform = uniform_step_scheme
  config.model.step_scheme_dict.last_small = small_lst_step_scheme
  config.model.step_scheme_dict.linear_dds = linear_step_scheme_dds
  config.model.step_scheme_dict.linear = linear_step_scheme
  config.model.step_scheme_dict.uniform_dds = uniform_step_scheme_dds

  config.model.step_scheme = config.model.step_scheme_dict[
      config.model.step_scheme_key]

  config.model.reference_process_key = "oustl"
  config.model.reference_process_dict = configdict.ConfigDict()
  config.model.reference_process_dict.oustl = AugmentedOUFollmerSDESTL
  config.model.reference_process_dict.oudstl = AugmentedOUDFollmerSDESTL
  config.model.reference_process_dict.pisstl = AugmentedBrownianFollmerSDESTL
  config.model.reference_process_dict.oududp = AugmentedOUDFollmerSDEUDP

  config.model.sigma = 0.25
  config.model.alpha = 0.5
  config.model.m = 1.0

  config.model.val = False

  config.trainer.learning_rate = 0.0001

  config.trainer.epochs = 2500
  config.trainer.log_every_n_epochs = 1

  config.trainer.lr_sch_base_dec = 0.95 # For funnel as per PIS repo
  config.model.stop_grad = True
  config.trainer.notebook = False
  config.trainer.simple_gaus_mean = 6.0

  config.trainer.objective = relative_kl_objective
  config.trainer.lnz_is_estimator = importance_weighted_partition_estimate
  config.trainer.lnz_pf_estimator = prob_flow_lnz
  config.model.detach_stl_drift = True
  config.model.detach_path = False
  config.model.log = False

  config.model.exp_dds = False

  config.trainer.random_seed = 42

  config.eval = configdict.ConfigDict()
  config.eval.seeds = 30

  return config


def set_task(config, task="lr_sonar", div=1, c=1):
  """Sets up task specific attributes for config.

  Args:
    config:
    task:

  Raises:
    BaseException: raises exception if config class not implemented

  Returns:
    task processed config
  """
  config.task = task

  if task == "lr_sonar" or task == "ion":
    config = log_reg_config.make_log_reg_config(config)
  elif task == "lgcp":
    config = lgcp_config.make_config(config)
  elif task == "vae":
    config = vae_config.make_config(config)
  elif task == "nice":
    config = pretrained_nice_config.make_config(config)
    config.model.fully_connected_units = [512, 256, 64]
    config.model.learn_betas = True
  elif task == "brownian":
    config = brownian_config.make_config(config)
  elif task == "funnel":
    config.model.input_dim = 10
    log_prob_funn, _ = toy_targets.funnel(d=config.model.input_dim , sig=1)

    config.model.elbo_batch_size = 2000
    config.trainer.lnpi = log_prob_funn
    config.model.target = log_prob_funn

  elif task == "mlpw":
    config.model.input_dim = 1
    config.model.elbo_batch_size = 2000
    config.model.target_class = toy_targets.mlpw_target_class(div, c)
    mlpw_target = config.model.target_class.mlpw
    config.trainer.lnpi = mlpw_target
    config.model.target = mlpw_target

  elif task == "carillo":
    config.model.input_dim = 1
    config.model.elbo_batch_size = 2000
    config.model.target_class = toy_targets.carillo_target_class(div, c)
    target = config.model.target_class.carillo
    config.trainer.lnpi = target
    config.model.target = target

  elif task == "layeb01":
    config.model.input_dim = 1
    config.model.elbo_batch_size = 2000
    layeb01_target = toy_targets.layeb01
    config.trainer.lnpi = layeb01_target
    config.model.target = layeb01_target

  elif task == "layeb10":
    config.model.input_dim = 1
    config.model.elbo_batch_size = 2000
    config.model.target_class = toy_targets.layeb10_target_class(div, c)
    layeb01_target = config.model.target_class.layeb10
    config.trainer.lnpi = layeb01_target
    config.model.target = layeb01_target

  elif task == "xor":
    config.model.input_dim = 9
    config.model.elbo_batch_size = 2000
    config.model.target_class = toy_targets.xor_target_class(div, c)
    target = config.model.target_class.xor
    config.trainer.lnpi = target
    config.model.target = target

  elif task == "breastcancer":
    config.model.input_dim = 64
    config.model.elbo_batch_size = 2000
    config.model.val = True
    config.model.target_class = toy_targets.breast_target_class(div, c)
    target = config.model.target_class.breast
    config.trainer.lnpi = target
    config.model.target = target

  elif task == "forest":
    config.model.input_dim = 1247
    config.model.elbo_batch_size = 2000
    config.model.val = True
    config.model.target_class = toy_targets.forest_target_class(div, c)
    target = config.model.target_class.forest
    config.trainer.lnpi = target
    config.model.target = target

  elif task == "mnist":
    config.model.input_dim = 21645
    config.model.elbo_batch_size = 2000
    config.model.val = True
    config.model.target_class = toy_targets.mnist_target_class(div, c)
    target = config.model.target_class.mnist
    config.trainer.lnpi = target
    config.model.target = target

  elif task == "moons":
    config.model.input_dim = 41
    config.model.elbo_batch_size = 32
    config.model.target_class = toy_targets.moons_target_class(div, c)
    target = config.model.target_class.moon
    config.trainer.lnpi = target
    config.model.target = target

  elif task == "michalewicz":
    config.model.input_dim = 2
    config.model.elbo_batch_size = 2000
    config.model.target_class = toy_targets.michalewicz_target_class(div, c)
    target = config.model.target_class.michalewicz
    config.trainer.lnpi = target
    config.model.target = target

  elif task == "levy":
    config.model.input_dim = 2
    config.model.elbo_batch_size = 2000
    config.model.target_class = toy_targets.levy_target_class(div, c)
    target = config.model.target_class.levy
    config.trainer.lnpi = target
    config.model.target = target

  elif task == "levy2":
    config.model.input_dim = 2
    config.model.elbo_batch_size = 2000
    config.model.target_class = toy_targets.levy_target_class2(div, c)
    target = config.model.target_class.levy
    config.trainer.lnpi = target
    config.model.target = target

  elif task == "anneal":
    config.model.input_dim = 1
    config.model.elbo_batch_size = 4
    config.model.target_class = toy_targets.anneal_target_class(div, c)
    target = config.model.target_class.anneal
    config.trainer.lnpi = target
    config.model.target = target

  elif task == "booth":
    config.model.input_dim = 2
    config.model.elbo_batch_size = 2000
    config.model.target_class = toy_targets.booth_target_class(div, c)
    target = config.model.target_class.booth
    config.trainer.lnpi = target
    config.model.target = target

    config.trainer.learning_rate = 5 * 10**(-3)
  elif task == "mixture_well":
    config.model.input_dim = 2
    config.model.sigma = 1.15
    config.model.alpha = 0.4

    log_prob_funn, _, plot_dist = toy_targets.mixture_well()

    config.trainer.lnpi = log_prob_funn
    config.model.target = log_prob_funn
    config.model.plot_dist = plot_dist
  elif task == "simple_gaussian":
    config.model.input_dim = 2

    log_prob_funn, _, plot_dist = toy_targets.simple_gaussian()

    config.trainer.lnpi = log_prob_funn
    config.model.target = log_prob_funn
    config.model.plot_dist = plot_dist
  elif task == "far_gaussian":
    config.model.input_dim = 2
    log_prob_funn, _, plot_dist = toy_targets.far_gaussian(
        mean=config.trainer.simple_gaus_mean)

    config.trainer.lnpi = log_prob_funn
    config.model.target = log_prob_funn
    config.model.plot_dist = plot_dist
  else:
    raise BaseException("Task config not implemented")

  tpu = config.model.tpu

  dtype = np.float32 if tpu else np.float64
  # import pdb; pdb.set_trace()
  step_scheme = config.model.step_scheme_dict[config.model.step_scheme_key]
  config.model.ts = step_scheme(
      0, config.model.tfinal, config.model.dt, dtype=dtype)

  config.model.stop_grad = True
  return config
