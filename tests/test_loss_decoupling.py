import torch

from mobile_fetal_clip.training.loss import DistillCLIPLoss


def test_decoupled_terms_match_coupled_kl_when_weight_one() -> None:
    torch.manual_seed(0)
    teacher = torch.randn(8, 8)
    student = torch.randn(8, 8)

    coupled = DistillCLIPLoss._kl_distill_loss(teacher, student, confidence_penalty=0.0)
    tckd, nckd = DistillCLIPLoss._decoupled_kl_distill_loss(teacher, student)

    assert torch.allclose(coupled, tckd + nckd, atol=1e-6)
