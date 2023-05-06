import utility_func
from dds.configs.config import set_task, get_config
from dds.train_dds import train_dds
from utility_func import *

"""
HYPERPARAMETERS:

sigma:
alpha:
div:

epochs:
samples:

"""
min_elbos = []
final_elbos = []

#divs = [1, 0.8, 0.5, 0.2]
divs = [1]
sigmas = [0.5, 0.75, 1, 1.5, 2]
alphas = [0.05, 0.25, 0.5, 1, 1.5, 2]

for div in divs:
    for sigma in sigmas:
        for alpha in alphas:
            funnel_config = get_config()

            # Time and step settings (Need to be done before calling set_task)
            funnel_config.model.tfinal = 6.4
            funnel_config.model.dt = 0.05  # 0.05

            if funnel_config.model.reference_process_key == "oudstl":
                funnel_config.model.step_scheme_key = "cos_sq"

            from dds.targets.toy_targets import get_attr

            ### SET TASK
            task = "carillo"
            # div, e, other_dim = get_attr()
            # div = str(div).replace(".", "")
            # e = str(e).replace(".", "")


            save_name = f"{task}_div{div}_sigma{sigma}_alpha{alpha}"

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
                funnel_config.model.sigma = sigma#1.075
                funnel_config.model.alpha = alpha#0.6875
                funnel_config.model.m = 1.0

                # Path opt settings
                funnel_config.model.exp_dds = False

            # funnel_config.model.stl = False
            # funnel_config.model.detach_stl_drift = False

            funnel_config.model.stl = True
            funnel_config.model.detach_stl_drift = True

            funnel_config.trainer.notebook = True
            # Opt settings we use
            # funnel_config.trainer.learning_rate = 0.0001
            funnel_config.trainer.learning_rate = 5 * 10 ** (-3)
            funnel_config.trainer.lr_sch_base_dec = 0.95  # For funnel

            funnel_config.trainer.epochs = 500
            out_dict = train_dds(funnel_config)


            plot_training_loss(out_dict[-1]["elbo"], save_name)
            elbo_final_loss = out_dict[-1]["elbo"][-1]
            min_elbo_loss = min(out_dict[-1]["elbo"])

            min_elbos.append(min_elbo_loss)
            final_elbos.append(elbo_final_loss)

utility_func.save_array_to_pickle(min_elbos, "notebooks/div_files/min_elbo_loss.pickle")
utility_func.save_array_to_pickle(final_elbos, "notebooks/div_files/final_elbo_loss.pickle")