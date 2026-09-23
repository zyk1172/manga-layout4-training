#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from manga_layout4.config import load_config
from manga_layout4.data import MangaLayoutDataset,collate
from manga_layout4.manifest import iter_manifest
from manga_layout4.model import build_model
from manga_layout4.losses import compute_loss
from manga_layout4.metrics import evaluate
from manga_layout4.train_utils import seed_everything

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'configs/base.yaml'); p.add_argument('--manifest',type=Path,default=ROOT/'data/manifests/pages.jsonl'); p.add_argument('--device',default='cuda'); a=p.parse_args(); cfg=load_config(a.config)
 import torch
 seed_everything(int(cfg['project']['seed']))
 rows=list(iter_manifest(a.manifest,'train')); want=int(cfg['preflight']['overfit_pages'])
 # Prioritize SFX/balloon pages, then deterministic fill.
 rows=sorted(rows,key=lambda r:(not r.get('has_onomatopoeia'),not r.get('has_balloon'),r['image_id']))[:want]
 ds=MangaLayoutDataset(a.manifest,cfg,'train',training=False,rows=rows); pf=ROOT/'outputs/preflight/report.json'; recommended=json.loads(pf.read_text()).get('recommended_micro_batch') if pf.is_file() else int(cfg['training']['micro_batch_size']); bs=min(int(recommended),len(ds))
 model=build_model(cfg,None).to(a.device).train(); opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']))
 nw=int(cfg['training']['num_workers']); pin=bool(cfg['training']['pin_memory'])
 loader=torch.utils.data.DataLoader(ds,batch_size=bs,shuffle=False,num_workers=nw,
  pin_memory=pin,persistent_workers=bool(cfg['training']['persistent_workers']) and nw>0,
  collate_fn=collate,drop_last=True)
 batches=iter(loader)
 steps=int(cfg['preflight']['overfit_steps']); window=max(10,steps//10)
 loss_start=torch.zeros(4,device=a.device); loss_end=torch.zeros(4,device=a.device)
 for step in range(steps):
  try: images,targets=next(batches)
  except StopIteration: batches=iter(loader); images,targets=next(batches)
  images=images.to(a.device,non_blocking=pin)
  for t in targets:
   for k in ('boxes','labels','balloon_masks'): t[k]=t[k].to(a.device,non_blocking=pin)
  opt.zero_grad(set_to_none=True); loss_dict=compute_loss(model(images),targets,cfg); l=loss_dict['total'];
  if not torch.isfinite(l): raise SystemExit(f'non-finite loss at {step}: {l}')
  l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['gradient_clip_norm'])); opt.step()
  values=torch.stack([loss_dict[k].detach() for k in ('total','classification','bbox','mask')])
  if step<window: loss_start.add_(values)
  if step>=steps-window: loss_end.add_(values)
 start_values=(loss_start/window).cpu().tolist(); end_values=(loss_end/window).cpu().tolist()
 start=float(start_values[0]); end=float(end_values[0]); drop=(start-end)/max(start,1e-9)
 component_losses={key:{'start_mean':float(start_values[i]),'end_mean':float(end_values[i]),
                        'drop_fraction':(float(start_values[i])-float(end_values[i]))/max(float(start_values[i]),1e-9)}
                   for i,key in enumerate(('classification','bbox','mask'),start=1)}
 need=float(cfg['preflight']['require_loss_drop_fraction'])
 eval_loader=torch.utils.data.DataLoader(ds,batch_size=bs,shuffle=False,num_workers=max(0,min(2,nw)),
  pin_memory=pin,persistent_workers=False,collate_fn=collate)
 metrics=evaluate(model,eval_loader,cfg,torch.device(a.device))
 box_floor=float(cfg['preflight']['require_overfit_box_ap50']); mask_floor=float(cfg['preflight']['require_overfit_mask_ap50']); boundary_floor=float(cfg['preflight']['require_overfit_boundary_ap50'])
 box_ok=all(float(metrics[f'{name}_ap50' if name!='balloon' else 'balloon_box_ap50'])>=box_floor for name in ('frame','text','balloon','onomatopoeia'))
 mask_ok=float(metrics['balloon_mask_ap50'])>=mask_floor; boundary_ok=float(metrics['balloon_boundary_ap50'])>=boundary_floor
 status='PASS' if drop>=need and box_ok and mask_ok and boundary_ok else 'FAIL'
 report={'schema_version':'manga-layout4-overfit-gate-v2','status':status,'pages':[r['image_id'] for r in rows],'steps':steps,'start_loss':start,'end_loss':end,'drop_fraction':drop,'component_losses':component_losses,'component_loss_window_steps':window,'required_drop_fraction':need,'metrics':metrics,'requirements':{'box_ap50_each':box_floor,'balloon_mask_ap50':mask_floor,'balloon_boundary_ap50':boundary_floor}}
 out=ROOT/'outputs/overfit'; out.mkdir(parents=True,exist_ok=True); (out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
 if status!='PASS': raise SystemExit(2)
if __name__=='__main__': main()
