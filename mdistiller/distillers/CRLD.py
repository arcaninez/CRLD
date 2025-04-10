import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

def kd_loss(logits_student, logits_teacher, temperature, reduce=False):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd

class CRLD(Distiller):
    """Cross-View Consistency Regularisation for Knowledge Distillation (ACMMM2024)"""

    def __init__(self, student, teacher, cfg):
        super(CRLD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.CRLD.CE_WEIGHT
        self.wv_loss_weight = cfg.CRLD.WV_WEIGHT
        self.cv_loss_weight = cfg.CRLD.CV_WEIGHT
        self.t = cfg.CRLD.TEMPERATURE
        self.tau_w = cfg.CRLD.TAU_W
        self.tau_s = cfg.CRLD.TAU_S

    def forward_train(self, image_w, image_s, target, **kwargs):
        logits_student_w, _ = self.student(image_w)
        logits_student_s, _ = self.student(image_s)

        with torch.no_grad():
            logits_teacher_w, _ = self.teacher(image_w)
            logits_teacher_s, _ = self.teacher(image_s)
            # logits_teacher_w = self.teacher(image_w) # for vit teacher
            # logits_teacher_s = self.teacher(image_s) # for vit teacher

        pred_teacher_w = F.softmax(logits_teacher_w.detach(), dim=1)
        conf_w, _ = pred_teacher_w.max(dim=1)
        conf_w = conf_w.detach()
        mask_w = conf_w.ge(self.tau_w).bool()

        pred_teacher_s = F.softmax(logits_teacher_s.detach(), dim=1)
        conf_s, _ = pred_teacher_s.max(dim=1)
        conf_s = conf_s.detach()
        mask_s = conf_s.ge(self.tau_s).bool()

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_w, target) + F.cross_entropy(logits_student_s, target))
        # CRLD losses
        loss_kd_wv = self.wv_loss_weight * ((kd_loss(logits_student_w, logits_teacher_w.detach(), self.t)
                                             + kd_loss(logits_student_s, logits_teacher_s.detach(), self.t)) * mask_w).mean()
        loss_kd_cv = self.cv_loss_weight * ((kd_loss(logits_student_s, logits_teacher_w.detach(), self.t)
                                             + kd_loss(logits_student_w, logits_teacher_s.detach(), self.t)) * mask_s).mean()

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd_wv": loss_kd_wv,
            "loss_kd_cv": loss_kd_cv,
        }

        return logits_student_w, losses_dict