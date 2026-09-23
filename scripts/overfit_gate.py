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
 losses=[]; steps=int(cfg['preflight']['overfit_steps'])
 for step in range(steps):
  ids=[(step*bs+i)%len(ds) for i in range(bs)]; images,targets=collate([ds[i] for i in ids]); images=images.to(a.device)
  for t in targets:
   for k in ('boxes','labels','balloon_masks'): t[k]=t[k].to(a.device)
  opt.zero_grad(set_to_none=True); l=compute_loss(model(images),targets,cfg)['total'];
  if not torch.isfinite(l): raise SystemExit(f'non-finite loss at {step}: {l}')
  l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['gradient_clip_norm'])); opt.step(); losses.append(float(l.detach().cpu()))
 start=sum(losses[:max(10,steps//10)])/max(10,steps//10); end=sum(losses[-max(10,steps//10):])/max(10,steps//10); drop=(start-end)/max(start,1e-9)
 need=float(cfg['preflight']['require_loss_drop_fraction'])
 loader=torch.utils.data.DataLoader(ds,batch_size=bs,shuffle=False,num_workers=0,collate_fn=collate)
 metrics=evaluate(model,loader,cfg,torch.device(a.device))
 box_floor=float(cfg['preflight']['require_overfit_box_ap50']); mask_floor=float(cfg['preflight']['require_overfit_mask_ap50']); boundary_floor=float(cfg['preflight']['require_overfit_boundary_ap50'])
 box_ok=all(float(metrics[f'{name}_ap50' if name!='balloon' else 'balloon_box_ap50'])>=box_floor for name in ('frame','text','balloon','onomatopoeia'))
 mask_ok=float(metrics['balloon_mask_ap50'])>=mask_floor; boundary_ok=float(metrics['balloon_boundary_ap50'])>=boundary_floor
 status='PASS' if drop>=need and box_ok and mask_ok and boundary_ok else 'FAIL'
 report={'schema_version':'manga-layout4-overfit-gate-v2','status':status,'pages':[r['image_id'] for r in rows],'steps':steps,'start_loss':start,'end_loss':end,'drop_fraction':drop,'required_drop_fraction':need,'metrics':metrics,'requirements':{'box_ap50_each':box_floor,'balloon_mask_ap50':mask_floor,'balloon_boundary_ap50':boundary_floor}}
 out=ROOT/'outputs/overfit'; out.mkdir(parents=True,exist_ok=True); (out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
 if status!='PASS': raise SystemExit(2)
if __name__=='__main__': main()
