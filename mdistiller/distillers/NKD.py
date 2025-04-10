import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def kd_loss_nkd(logits_student, logits_teacher, target, temperature, gamma):
    B, _ = logits_student.shape
    log_pred_student = F.log_softmax(logits_student, dim=1)
    pred_teacher = F.softmax(logits_teacher, dim=1)
    target_log_pred_student = torch.gather(log_pred_student, 1, target.unsqueeze(-1))
    target_pred_teacher = torch.gather(pred_teacher, 1, target.unsqueeze(-1))
    loss_t = - (target_pred_teacher * target_log_pred_student).mean()
    nontarget_mask = _get_other_mask(logits_teacher, target)
    nontarget_logits_student = logits_student[nontarget_mask].reshape(B,-1)
    nontarget_logits_teacher = logits_teacher[nontarget_mask].reshape(B,-1)
    nontarget_log_pred_student = F.log_softmax(nontarget_logits_student / temperature)
    nontarget_pred_teacher = F.softmax(nontarget_logits_teacher / temperature, dim=1)
    loss_nt = - (nontarget_pred_teacher * nontarget_log_pred_student).sum(dim=1).mean()*(temperature**2)
    loss_nt = loss_nt * gamma
    loss_kd = loss_t + loss_nt
    return loss_kd

class NKD(Distiller):
    """From Knowledge Distillation to Self-Knowledge Distillation: A Unified Approach with Normalized Loss and Customized Soft Labels (ICCV 2023)"""

    def __init__(self, student, teacher, cfg):
        super(NKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.NKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.NKD.LOSS.KD_WEIGHT
        self.temperature = cfg.NKD.T
        self.gamma = cfg.NKD.GAMMA

    def forward_train(self, image, target, **kwargs):
        logits_student, feat_student = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_nkd = self.kd_loss_weight * kd_loss_nkd(logits_student, logits_teacher, target, self.temperature, self.gamma)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_nkd,
        }
        return logits_student, losses_dict