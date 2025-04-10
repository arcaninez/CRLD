from .trainer import BaseTrainer, CRDTrainer, AugTrainer, CRLDTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "mlld": AugTrainer,
    "crld": CRLDTrainer,
}
