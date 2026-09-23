import yaml, torch
from pathlib import Path
from manga_layout4.model import build_model,RawExportWrapper
from manga_layout4.losses import compute_loss
ROOT=Path(__file__).resolve().parents[1]

def cfg(): return yaml.safe_load((ROOT/'configs/base.yaml').read_text())

def test_forward_shapes():
 m=build_model(cfg(),pretrained_backbone=False).eval(); x=torch.rand(1,3,640,640)
 with torch.no_grad(): o=m(x)
 assert [tuple(v.shape) for v in o['cls_logits']]==[(1,4,160,160),(1,4,80,80),(1,4,40,40),(1,4,20,20)]
 assert tuple(o['balloon_mask_logits'].shape)==(1,1,320,320)

def test_loss_finite_and_backward():
 m=build_model(cfg(),pretrained_backbone=False); x=torch.rand(1,3,640,640); o=m(x)
 mask=torch.zeros(1,320,320); mask[:,50:100,50:100]=1
 targets=[{'boxes':torch.tensor([[100.,100.,200.,220.],[300.,200.,350.,260.]]),'labels':torch.tensor([2,3]),'balloon_mask':mask}]
 loss=compute_loss(o,targets,cfg()); assert torch.isfinite(loss['total']); loss['total'].backward()
