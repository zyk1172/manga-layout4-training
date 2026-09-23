import yaml, torch
from pathlib import Path
from manga_layout4.model import build_model,RawExportWrapper
from manga_layout4.losses import compute_loss
from manga_layout4.decode import balloon_instance_masks
ROOT=Path(__file__).resolve().parents[1]

def cfg(): return yaml.safe_load((ROOT/'configs/base.yaml').read_text())

def test_forward_shapes():
 m=build_model(cfg(),pretrained_backbone=False).eval(); x=torch.rand(1,3,640,640)
 with torch.no_grad(): o=m(x)
 assert [tuple(v.shape) for v in o['cls_logits']]==[(1,4,160,160),(1,4,80,80),(1,4,40,40),(1,4,20,20)]
 assert [tuple(v.shape) for v in o['mask_coeff']]==[(1,8,160,160),(1,8,80,80),(1,8,40,40),(1,8,20,20)]
 assert tuple(o['mask_prototypes'].shape)==(1,8,320,320)
 assert len(RawExportWrapper(m)(x))==13

def test_loss_finite_and_backward():
 c=cfg(); m=build_model(c,pretrained_backbone=False); x=torch.rand(1,3,640,640); o=m(x)
 masks=torch.zeros(1,320,320); masks[0,50:110,50:100]=1
 targets=[{'boxes':torch.tensor([[100.,100.,200.,220.],[300.,200.,350.,260.]]),'labels':torch.tensor([2,3]),'balloon_masks':masks}]
 loss=compute_loss(o,targets,c); assert torch.isfinite(loss['total']); assert float(loss['mask_instances'])==1.0; loss['total'].backward()

def test_instance_mask_decode_shape():
 c=cfg(); m=build_model(c,pretrained_backbone=False).eval(); x=torch.rand(1,3,640,640)
 with torch.no_grad(): o=m(x)
 det={'boxes':torch.tensor([[10.,20.,210.,230.]]),'scores':torch.tensor([0.9]),'labels':torch.tensor([2]),'mask_coeffs':torch.zeros((1,8))}
 ids,masks=balloon_instance_masks(o,det,0,c)
 assert ids.tolist()==[0]
 assert tuple(masks.shape)==(1,320,320)
