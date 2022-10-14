import copy
import logging
import os
import yaml
import time

import submitit

from mdsim.common import distutils
from mdsim.common.flags import flags
from mdsim.common.registry import registry
from mdsim.common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
    setup_imports,
    setup_logging,
    compose_data_cfg
)


class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        setup_logging()
        self.config = copy.deepcopy(config)

        if config['distributed']:
            distutils.setup(config)

        try:
            setup_imports()
            
            # compose dataset configs.
            train_data_cfg = config['dataset']
            train_data_cfg = compose_data_cfg(train_data_cfg)
            config['dataset'] = [
                train_data_cfg,
                {'src': os.path.join(os.path.dirname(train_data_cfg['src']), 'val')}
            ]
            
            self.config = copy.deepcopy(config)
            
            # initialize trainer.
            self.trainer = registry.get_trainer_class(
                config.get("trainer", "energy")
            )(
                task=config["task"],
                model=config["model"],
                dataset=config["dataset"],
                optimizer=config["optim"],
                identifier=config["identifier"],
                timestamp_id=config.get("timestamp_id", None),
                run_dir=config.get("run_dir", None),
                is_debug=config.get("is_debug", False),
                print_every=config.get("print_every", 100),
                seed=config.get("seed", 0),
                logger=config.get("logger", "wandb"),
                local_rank=config["local_rank"],
                amp=config.get("amp", False),
                cpu=config.get("cpu", False),
                slurm=config.get("slurm", {}),
                no_energy=config.get("no_energy", False)
            )
            
            # save config.
            with open(os.path.join(self.trainer.config["cmd"]["checkpoint_dir"], 'config.yml'), 'w') as yf:
                yaml.dump(self.config, yf, default_flow_style=False)

            self.task = registry.get_task_class(config["mode"])(self.config)
            self.task.setup(self.trainer)
            start_time = time.time()
            self.task.run()
            distutils.synchronize()
            if distutils.is_master():
                logging.info(f"Total time taken: {time.time() - start_time}")
        finally:
            if config['distributed']:
                distutils.cleanup()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        return submitit.helpers.DelayedSubmission(new_runner, self.config)


if __name__ == "__main__":
    setup_logging() 
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    if args.nequip:
        os.system(f'nequip-train {args.config_yml}')
    else:
        config = build_config(args, override_args)
        if args.submit:  # Run on cluster
            slurm_add_params = config.get(
                "slurm", None
            )  # additional slurm arguments
            if args.sweep_yml:  # Run grid search
                configs = create_grid(config, args.sweep_yml)
            else:
                configs = [config]

            logging.info(f"Submitting {len(configs)} jobs")
            executor = submitit.AutoExecutor(
                folder=args.logdir / "%j", slurm_max_num_timeout=3
            )
            executor.update_parameters(
                name=args.identifier,
                mem_gb=args.slurm_mem,
                timeout_min=args.slurm_timeout * 60,
                slurm_partition=args.slurm_partition,
                gpus_per_node=args.num_gpus,
                cpus_per_task=(config["optim"]["num_workers"] + 1),
                tasks_per_node=(args.num_gpus if args.distributed else 1),
                nodes=args.num_nodes,
                slurm_additional_parameters=slurm_add_params,
            )
            for config in configs:
                config["slurm"] = copy.deepcopy(executor.parameters)
                config["slurm"]["folder"] = str(executor.folder)
            jobs = executor.map_array(Runner(), configs)
            logging.info(
                f"Submitted jobs: {', '.join([job.job_id for job in jobs])}"
            )
            log_file = save_experiment_log(args, jobs, configs)
            logging.info(f"Experiment log saved to: {log_file}")

        else:  # Run locally
            Runner()(config)
