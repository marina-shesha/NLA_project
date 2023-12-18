from dataclasses import dataclass
from typing import Union

from src.utils.storage import StateDict


@dataclass
class TrainConfig:
    train_batch_size = 512
    eval_batch_size = 512

    num_epoches = 50
    momentum_beta = 0.9
    rmsprop_beta = 0.999
    label_smoothig = 0.1
    learning_rate = 2000
    scheduler_step = 0.933
    decay = 0.01

    base_regularization_coeff = 1e-8
    final_regularization_coeff = 1e-14
    coeff_adjusting_policy = "exp"
    num_regularizer_decreasing_steps = 3

    checkpoint_path = "checkpoints/"

    def to_dict(self):
        return {
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_epoches": self.num_epoches,
            "learning_rate": self.learning_rate,
            "momentum_beta": self.momentum_beta,
            "label_smoothig": self.label_smoothig,
        }


@dataclass
class TuneConfig:
    num_tunning_runs = 2
    num_run_epochs = 2

    relation_rank_inc = 0
    entity_rank_inc = 1

    def to_dict(self):
        return {
            "num_tunning_runs": self.num_tunning_runs,
            "num_run_epochs": self.num_run_epochs,
            "relation_rank_inc": self.relation_rank_inc,
            "entity_rank_inc": self.entity_rank_inc,
        }

@dataclass
class ModelConfig:
    manifold_rank = (10, 200, 200)

    use_pretrained = False
    pretrained_path = "./checkpoints/rk_20_903"

    def to_dict(self):
        return {
            "manifold_rank": self.manifold_rank,
            "use_pretrained": self.use_pretrained,
            "pretrained_path": self.pretrained_path
        }


@dataclass
class LogConfig:
    project_name = None
    entity_name = None
    run_name = "(200, 20) 10, 5 tune"
    log_dir = "wandb_logs"

    watch_log_freq = 500
    watch_log = "all"


@dataclass
class Config:
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()
    log_cfg = LogConfig()
    tune_cfg = TuneConfig()

    state_dict: Union[None, StateDict]

