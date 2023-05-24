"""Main training file.

For training diffusion based samplers (OU reversal SDE and Follmer SDE )
"""
import functools
import timeit
from typing import Any, List, Tuple
from absl import flags

from absl import logging
import haiku as hk
import distrax
import jax
import jax.numpy as jnp

from ml_collections import config_dict as configdict
from ml_collections import config_flags
import jax.experimental.host_callback as hcb
import numpy as onp
import optax

from jaxline import utils

import visualisation
from dds.configs.config import set_task
from dds.data_paths import results_path
from utility_func import get_smallest_loss, get_smallest_validation_loss, get_highest_accuracy, \
    get_highest_averaged_accuracy

FLAGS = flags.FLAGS
Writer = Any

# lr_sonar, funnel, lgcp, ion, nice, vae, brownian
_TASK = flags.DEFINE_string("task", "lr_sonar", "Inference task name.")
config_flags.DEFINE_config_file(
    "config",
    "//dds/config.py",
    lock_config=False,
    help_string="Path to ConfigDict file."
)


class WandbWriterWrapper:

  def __init__(self, *args, **kwargs):
    pass

  def write(self, *args):
    pass


def update_detached_params(trainable_params, non_trainable_params,
                           attached_network_name="simple_drift_net",
                           detached_network_name="stl_detach"):
  """Auxiliary function updating detached params for STL.

  Args:
      trainable_params:
      non_trainable_params:
      attached_network_name:
      detached_network_name:
  Returns:
    Returns non trainable params
  """

  if len(trainable_params) != len(non_trainable_params):
    return non_trainable_params

  for key in trainable_params.keys():
    if attached_network_name in key:
      key_det = key.replace(attached_network_name, detached_network_name)
    else:
      key_det = key.replace("diffusion_network",
                            detached_network_name + "_diff")
    non_trainable_params[key_det] = trainable_params[key]  # pytype: disable=unsupported-operands

  return non_trainable_params


def train_dds(
    config: configdict.ConfigDict):
# ) -> Tuple[hk.Params, hk.State, hk.TransformedWithState, jnp.ndarray,
#            List[float]]:
  """Train Follmer SDE.

  Args:
    config : ConfigDict with model and training parameters.

  Returns:
    Tuple containing params, state, function that runs learned sde, and losses
  """

  # train setup
  data_dim = config.model.input_dim
  device_no = jax.device_count()

  alpha = config.model.alpha
  sigma = config.model.sigma
  m = config.model.m
   
  # post setup model vars
  config.model.source = distrax.MultivariateNormalDiag(
      jnp.zeros(config.model.input_dim),
      config.model.sigma * jnp.ones(config.model.input_dim)).log_prob

  batch_size_ = int(config.model.batch_size / device_no)
  batch_size_elbo = int(config.model.elbo_batch_size / device_no)

  step_scheme = config.model.step_scheme_dict[config.model.step_scheme_key]

  dt = config.model.dt

  if config.model.reference_process_key == "oududp":
    key_conversion = {
        "pis": "pisudp",
        "vanilla": "vanilla_udp",
        "tmpis": "tmpis_udp"
    }
    # "pisudp"
    config.model.network_key = key_conversion[config.model.network_key]

  net_key = config.model.network_key
  network = config.model.network_dict[net_key]

  tpu = config.model.tpu

  detach_dif_path, detach_dritf_path = (
      config.model.detach_path, config.model.detach_path)

  # some target distribution
  target = config.model.target


  tfinal = config.model.tfinal
  lnpi = config.trainer.lnpi

  ref_proc_key = config.model.reference_process_key
  ref_proc = config.model.reference_process_dict[ref_proc_key]

  trim = (2 if "stl" in str(ref_proc).lower() or "udp" in str(ref_proc).lower()
          else 1)
 
  stl = config.model.stl

  brown = "brown" in str(ref_proc).lower()

  seed = config.trainer.random_seed  if "random_seed" in config.trainer else 42

  # task directory (currently not in use)
  task = config.task
  method = config.model.reference_process_key
  task_path = results_path + f"/{task}" + f"/{ref_proc_key}" + f"/{net_key}"
  task_path += f"/{method}"


  # checkpoiting variables for wandb
  nsteps = config.model.ts.shape[0]
  keep_every_nth = int(config.trainer.epochs / 125)
  file_name = (f"/alpha_{alpha}_sigma_{sigma}_epochs_{config.trainer.epochs}" +
               f"_task_{task}_seed_{seed}_steps_{nsteps}_stl_{stl}_{method}" +
               f"_scheme_{config.model.step_scheme_key}_ddpm_test11_chk")
  _ = task_path + file_name

  detach_stl_drift = (
      config.model.detach_stl_drift if
      "detach_stl_drift" in config.model else False
  )

  drift_network = lambda: network(config.model, data_dim, "simple_drift_net")

  ############## wandb logging  place holder ################
  data_id = "denoising_diffusion_samplers"  # Project name
  training_writer = WandbWriterWrapper(data_id, dataframe="elbo_results")
  training_writer_eval = WandbWriterWrapper(data_id, dataframe="elbo_results_eval")
  is_writer = WandbWriterWrapper(data_id, dataframe="is_results")
  is_writer_eval = WandbWriterWrapper(data_id, dataframe="is_results_eval")
  pf_writer = WandbWriterWrapper(data_id, dataframe="pf_results")
  pf_writer_eval = WandbWriterWrapper(data_id, dataframe="pf_results_eval")
  lr_writer = WandbWriterWrapper(data_id, dataframe="lr")

  some_var = 9

  def _forward_fn(batch_size: int,
                  training: bool = True,
                  ode=False, exact=False, dt_=dt, new_target=None) -> jnp.ndarray:

    #import jax.experimental.host_callback as hcb


    # things = (
    #     sigma, data_dim, drift_network, tfinal, dt_,
    #     step_scheme, alpha, target, tpu,
    #     detach_stl_drift, None,
    #     detach_dritf_path, detach_dif_path,
    #     m, config.model.log, config.model.exp_dds, exact
    # )

    # print("START")
    # for i, x in enumerate(things):
    #     print(i)
    #     try:
    #         hcb.id_print(x)
    #     except:
    #         print(x)
    # print("END")


    model_def = ref_proc(
        sigma, data_dim, drift_network, tfinal=tfinal, dt=dt_,
        step_scheme=step_scheme, alpha=alpha, target=target, tpu=tpu,
        detach_stl_drift=detach_stl_drift, diff_net=None,
        detach_dritf_path=detach_dritf_path, detach_dif_path=detach_dif_path,
        m=m, log=config.model.log, exp_bool=config.model.exp_dds, exact=exact
    )

    # print(batch_size)
    # print(training)
    # print(ode)

    return model_def(batch_size, training, ode=ode)

  # Transforms a function using Haiku modules into a pair of pure functions ???
  forward_fn = hk.transform_with_state(_forward_fn)

  # opt and loss setup
  seq = hk.PRNGSequence(seed)
  rng_key = next(seq)
  # subkeys = jax.random.split(rng_key, device_no)
  subkeys = utils.bcast_local_devices(rng_key)

  p_init = jax.pmap(
      functools.partial(forward_fn.init, batch_size=batch_size_,
                        training=True), axis_name="num_devices")

  params, model_state = p_init(subkeys)

  #hcb.id_print(params)

  trainable_params, non_trainable_params = hk.data_structures.partition(
      lambda module, name, value: "stl_detach" not in module, params)

  #hcb.id_print(non_trainable_params)



  clipper = optax.clip(1.0)
  base_dec = config.trainer.lr_sch_base_dec # 0.95 for funnel
  scale_by_adam = optax.scale_by_adam()
  # if base_dec == 0:
  #   scale_by_lr = optax.scale(-config.trainer.learning_rate)
  #   opt = optax.chain(clipper, scale_by_adam, scale_by_lr)
  # else:
  transition_steps = 10#50
  exp_lr = optax.exponential_decay(config.trainer.learning_rate,
                                   transition_steps, base_dec)
  scale_lr = optax.scale_by_schedule(exp_lr)
  opt = optax.chain(clipper, scale_by_adam, scale_lr, optax.scale(-1))
  #opt = optax.adam(0.0005)

  opt = optax.adam(learning_rate=config.trainer.learning_rate)
  opt_state = jax.pmap(opt.init)(trainable_params)

  @functools.partial(
      jax.pmap, axis_name="num_devices", static_broadcasted_argnums=(3, 4, 5, 6))
  def forward_fn_jit(
      params,
      model_state: hk.State,
      subkeys: jnp.ndarray,
      batch_size: jnp.ndarray, ode=False, exact=False,  dt_=dt, new_target=None):

    samps, _ = forward_fn.apply(
        params,
        model_state,
        subkeys,
        int(batch_size / device_no),
        False,
        ode=ode, exact=exact, dt_=dt_, new_target=new_target)
    samps = jax.device_get(samps)

    augmented_trajectory, ts = samps
    return (augmented_trajectory, ts), _

  def forward_fn_wrap(
      params,
      model_state: hk.State,
      rng_key: jnp.ndarray,
      batch_size: jnp.ndarray, ode=False, exact=False, dt_=dt, new_target=None):
    subkeys = jax.random.split(rng_key, device_no)
    (augmented_trajectory, ts), _ = forward_fn_jit(params, model_state,
                                                   subkeys, batch_size, ode, exact,
                                                   dt_, new_target=new_target)

    dv, ns, t, _ = augmented_trajectory.shape
    augmented_trajectory = augmented_trajectory.reshape(dv*ns, t, -1)
    return (augmented_trajectory, utils.get_first(ts)), _

  def full_objective(
      trainable_params,
      non_trainable_params,
      model_state: hk.State,
      rng_key: jnp.ndarray,
      batch_size: int,
      is_training: bool = True,
      ode: bool = False,
      stl: bool = False,
      exact: bool = False,
      new_target=None
    ):

    params = hk.data_structures.merge(trainable_params, non_trainable_params)
    (augmented_trajectory, _), model_state = forward_fn.apply(
        params, model_state, rng_key, batch_size, True, ode, exact, new_target=new_target
    )

    #hcb.id_print(augmented_trajectory.shape)

    # import pdb; pdb.set_trace()
    gpartial = functools.partial(
        config.model.terminal_cost,
        lnpi=lnpi, sigma=sigma, tfinal=tfinal, brown=brown)

    if is_training:
      loss = config.trainer.objective(
          augmented_trajectory, gpartial, stl=stl, trim=trim, dim=data_dim)
      #print(stl, trim)
    elif not ode:
      loss = config.trainer.lnz_is_estimator(
          augmented_trajectory, gpartial, dim=data_dim)
    else:
      loss = config.trainer.lnz_pf_estimator(
          augmented_trajectory, config.model.source, config.model.target)
    return loss, model_state

  @functools.partial(
      jax.pmap, axis_name="num_devices", static_broadcasted_argnums=(5,))
  def update(
      trainable_params,
      non_trainable_params,
      model_state: hk.State,
      opt_state: Any,
      rng_key: jnp.ndarray,
      batch_size: jnp.ndarray):
    grads, new_model_state = jax.grad(
        full_objective, has_aux=True)(
            trainable_params,
            non_trainable_params,
            model_state,
            rng_key,
            batch_size,
            is_training=True,
            stl=stl)

    grads = jax.lax.pmean(grads, axis_name="num_devices")
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(trainable_params, updates)
    return new_params, opt_state, new_model_state

  @functools.partial(
      jax.pmap, axis_name="num_devices", static_broadcasted_argnums=(4, 5, 6, 7))
  def jited_val_loss(
      trainable_params,
      non_trainable_params,
      model_state: hk.State,
      rng_key: jnp.ndarray,
      batch_size: jnp.ndarray,
      is_training: bool = True,
      ode: bool = False,
      exact: bool = False,
      new_target = None):

    new_target = jnp.array([new_target])

    loss, new_model_state = full_objective(
        trainable_params,
        non_trainable_params,
        model_state,
        rng_key,
        batch_size,
        is_training=is_training, ode=ode,
        stl=False, exact=exact,
        new_target=new_target)

    loss = jax.lax.pmean(loss, axis_name="num_devices")
    return loss, new_model_state

  def eval_report(
      trainable_params,
      non_trainable_params,
      model_state: hk.State,
      rng_key: jnp.ndarray,
      batch_size: int,
      epoch: int,
      writer: Writer,
      loss_list: List[float],
      is_training: bool = True,
      print_flag: bool = False,
      ode: bool = False,
      exact: bool = False,
      write=True
  ) -> None:


    loss, model_state = jited_val_loss(
        trainable_params, non_trainable_params,
        model_state, rng_key, batch_size, is_training, ode, exact)
    loss = jax.device_get(loss)
    loss = onp.asarray(utils.get_first(loss).item()).item()

    log_string = "epoch: %s %s  loss: %s", epoch, "TRAIN", loss
    logging.info(log_string)
    if config.trainer.notebook and print_flag: print(log_string)

    if write:
        loss_list.append(loss)
        writer.write({"epoch": epoch, "loss": loss})
    # writer.flush()

  loss_list = []
  loss_list_is = []
  loss_list_pf = []

  start = 0
  times = []

  # change based on task
  import utility_func

  accumulated_training_loss = []
  accumulated_validation_loss = []


  accumulated_training_acc_avg = []
  accumulated_validation_acc_avg = []

  accumulated_training_acc = []
  accumulated_validation_acc = []

  best_training_weights = None
  best_training_weights_avg = None
  best_weights = None
  best_weights_avg = None


  for epoch in range(start, config.trainer.epochs):
    #hcb.id_print(epoch)
    rng_key = next(seq)
    subkeys = jax.random.split(rng_key, device_no)

    trainable_params, opt_state, model_state = update(trainable_params,
                                                      non_trainable_params,
                                                      model_state, opt_state,
                                                      subkeys, batch_size_)

    # if epoch % 1 == 0:
    #     eval_report(trainable_params, non_trainable_params,
    #                 model_state, subkeys, batch_size_elbo, epoch,
    #                 training_writer, loss_list, print_flag=False, write=False)
    #
    #     update_detached_params(trainable_params, non_trainable_params,
    #                            "simple_drift_net", "stl_detach")
    #     params = hk.data_structures.merge(trainable_params, non_trainable_params)
    #     (augmented_trajectory, _), _ = forward_fn_wrap(params, model_state, jax.random.PRNGKey(1), 1000)
    #     b, w = get_smallest_loss(augmented_trajectory, config, type="non-batch")
    #     print(f"Best training loss: {b}")
    #     accumulated_training_loss.append(b)

    # #Training loss
    if epoch % 10000 == 9000: # 2

        eval_report(trainable_params, non_trainable_params,
                    model_state, subkeys, batch_size_elbo, epoch,
                    training_writer, loss_list, print_flag=False, write=False)


        update_detached_params(trainable_params, non_trainable_params,
                               "simple_drift_net", "stl_detach")
        params = hk.data_structures.merge(trainable_params, non_trainable_params)
        (augmented_trajectory, _), _ = forward_fn_wrap(params, model_state, jax.random.PRNGKey(1), 1000)



        b, w = get_smallest_loss(augmented_trajectory, config, type="non-batch")
        print(f"Best training loss: {b}")
        accumulated_training_loss.append(b)


        b1, w1 = get_smallest_loss(augmented_trajectory, config, type="validation")
        print(f"Best validation loss: {b1}")
        accumulated_validation_loss.append(b1)


        best_training_acc, w_best_t_acc = get_highest_accuracy(augmented_trajectory, config, type="training")
        #best_averaged_training_acc, w_best_t_acc_avg = get_highest_averaged_accuracy(augmented_trajectory, config, type="training")

        best_val_acc, w_best_v_acc = get_highest_accuracy(augmented_trajectory, config, type="validation")
        #best_averaged_val_acc, w_best_v_acc_avg = get_highest_averaged_accuracy(augmented_trajectory, config, type="validation")


        accumulated_training_acc.append(best_training_acc)
        accumulated_validation_acc.append(best_val_acc)

        #accumulated_training_acc_avg.append(best_averaged_training_acc)
        #accumulated_validation_acc_avg.append(best_averaged_val_acc)


        print(f"Best training accuracy: {best_training_acc}")
        print(f"Best validation accuracy: {best_val_acc}")
        #print(f"Best training accuracy (AVG): {best_averaged_training_acc}")
        #print(f"Best validation accuracy (AVG): {best_averaged_val_acc}")

        if best_training_acc >= max(accumulated_training_acc):
            best_training_weights = w_best_t_acc

        # if best_averaged_training_acc >= max(accumulated_training_acc_avg):
        #     best_training_weights_avg = w_best_t_acc_avg

        if best_val_acc >= max(accumulated_validation_acc):
            best_weights = w_best_v_acc

        # if best_averaged_val_acc >= max(accumulated_validation_acc_avg):
        #     best_weights_avg = w_best_v_acc_avg

        # # historically best
        # accuracy = config.model.target_class.accuracy(best_weights, "test")
        # accuracy_avg = config.model.target_class.accuracy(best_weights_avg, "test")
        # accuracy_train = config.model.target_class.accuracy(best_training_weights, "test")
        # accuracy_train_avg = config.model.target_class.accuracy(best_training_weights_avg, "test")

        # curr best
        accuracy = config.model.target_class.accuracy(w_best_v_acc, "test")
        #accuracy_avg = config.model.target_class.accuracy(w_best_v_acc_avg, "test")
        accuracy_train = config.model.target_class.accuracy(w_best_t_acc, "test")
        #accuracy_train_avg = config.model.target_class.accuracy(w_best_t_acc_avg, "test")

        print("WITH VAL")
        print(f"Test Accuracy: {accuracy}")
        #print(f"Test Accuracy (AVG): {accuracy_avg}")
        print("WITH TRAIN")
        print(f"Test Accuracy: {accuracy_train}")
        #print(f"Test Accuracy (AVG): {accuracy_train_avg}")





    if config.trainer.timer:
      def func():
        return jax.block_until_ready(
            update(trainable_params, non_trainable_params, model_state,
                   opt_state, subkeys, batch_size_))

      delta_time = timeit.timeit(func, number=1)
      times.append(delta_time)

    update_detached_params(trainable_params, non_trainable_params,
                           "simple_drift_net", "stl_detach")

    if epoch % config.trainer.log_every_n_epochs == 0:

      eval_report(trainable_params, non_trainable_params,
                  model_state, subkeys, batch_size_elbo, epoch,
                  training_writer, loss_list, print_flag=True)

      eval_report(trainable_params, non_trainable_params,
                  model_state, subkeys, batch_size_elbo, epoch,
                  is_writer, loss_list_is, is_training=False)

      eval_report(trainable_params, non_trainable_params,
                  model_state, subkeys, batch_size_elbo, epoch,
                  pf_writer, loss_list_pf, is_training=False, ode=True)

      lr = onp.asarray(exp_lr(epoch).item()).item()
      lr_writer.write({"epoch": epoch, "lr": lr})

  # checking fine-tune
  # print(f"TEST ACCURACY: {config.model.target_class.breast_task.get_test_accuracy(best_weights)}")
  # config.model.target_class.breast_task.fine_tune(best_weights)

  # accuracy = 0
  # accuracy_avg = 0
  # test_loss = 0
  # accuracy = config.model.target_class.accuracy(best_weights, "test")
  # accuracy_avg = config.model.target_class.accuracy(best_weights_avg, "test")
  #
  # accuracy_train = config.model.target_class.accuracy(best_training_weights, "test")
  # accuracy_train_avg = config.model.target_class.accuracy(best_training_weights_avg, "test")
  #
  # print("WITH VAL")
  # print(f"Test Accuracy: {accuracy}")
  # print(f"Test Accuracy (AVG): {accuracy_avg}")
  # print("WITH TRAIN")
  # print(f"Test Accuracy: {accuracy_train}")
  # print(f"Test Accuracy (AVG): {accuracy_train_avg}")

  import numpy as np
  import matplotlib.pyplot as plt
  if False:
      list_of_weights = [best_weights, best_weights_avg, best_training_weights, best_training_weights_avg]
      for weight in list_of_weights:
          X_test, y_pred = config.model.target_class.pred(weight)
          y_pred_1d = y_pred.flatten().astype(int)
          colors = np.array(['r', 'b'])
          plt.scatter(X_test[:, 0], X_test[:, 1], c=colors[y_pred_1d], alpha=0.5)
        # Add axis labels
          plt.xlabel('x')
          plt.ylabel('y')
        # Add a legend for the labels
          plt.legend(handles=[plt.Line2D([0], [0], linestyle='none', marker='o', color='r', label='Label 0'),
          plt.Line2D([0], [0], linestyle='none', marker='o', color='b', label='Label 1')],
               title='Predicted Labels',
               loc='best')
        # Show the plot
          plt.show()


  loss_list_is_eval, loss_list_eval, loss_list_pf_eval = [], [], []
  for i in range(config.eval.seeds):
    rng_key = next(seq)
    subkeys = jax.random.split(rng_key, device_no)
    eval_report(
        trainable_params,
        non_trainable_params,
        model_state,
        subkeys,
        batch_size_elbo,
        i,
        training_writer_eval,
        loss_list_eval,
        print_flag=True)

    eval_report(
        trainable_params,
        non_trainable_params,
        model_state,
        subkeys,
        batch_size_elbo,
        i,
        is_writer_eval,
        loss_list_is_eval,
        is_training=False)

    eval_report(
        trainable_params,
        non_trainable_params,
        model_state,
        subkeys,
        batch_size_elbo,
        i,
        pf_writer_eval,
        loss_list_pf_eval,
        is_training=False, ode=True, exact=False)

  params = hk.data_structures.merge(trainable_params, non_trainable_params)
  if config.trainer.timer:
    print(times[1:])

  samps = 5000
  if method == "lgcp" and tfinal >= 12:
    samps = 1000

  (augmented_trajectory, _), _ = forward_fn_wrap(params, model_state, rng_key,
                                                 samps)

  (augmented_trajectory_det, _), _ = forward_fn_wrap(params, model_state,
                                                     rng_key, samps, True, False)

  (augmented_trajectory_det_ext, _), _ = forward_fn_wrap(params, model_state,
                                                         rng_key, samps, True, True)


  results_dict = {
      "elbo": loss_list,
      "is": loss_list_is,
      "pf": loss_list_pf,
      "elbo_eval": loss_list_eval,
      "is_eval": loss_list_is_eval,
      "pf_eval": loss_list_pf_eval,
      "aug": augmented_trajectory,
      "aug_ode": augmented_trajectory_det,
      "aug_ode_ext": augmented_trajectory_det_ext,
      "training_loss": accumulated_training_loss,
      "validation_loss": accumulated_validation_loss,
      "test_loss": test_loss,
      "validation_acc": accumulated_validation_acc,
      "training_acc": accumulated_training_acc,
      "validation_acc_avg": accumulated_validation_acc_avg,
      "training_acc_avg": accumulated_training_acc_avg
  }
  return params, model_state, forward_fn_wrap, rng_key, results_dict


def main(_):

  config_file = FLAGS.config
  config_file = set_task(config_file, task=_TASK.value)
  logging.info(config_file)
  train_pmap(config_file)


if __name__ == "__main__":
  pass
