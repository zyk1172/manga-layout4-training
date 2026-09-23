#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time,traceback,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from manga_layout4.config import load_config
from manga_layout4.data import MangaLayoutDataset,collate
from manga_layout4.model import build_model
from manga_layout4.losses import compute_loss
from manga_layout4.train_utils import seed_everything

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'configs/base.yaml'); p.add_argument('--manifest',type=Path,default=ROOT/'data/manifests/pages.jsonl'); p.add_argument('--device',default='cuda'); a=p.parse_args(); cfg=load_config(a.config)
 import torch
 seed_everything(int(cfg['project']['seed']))
 if a.device.startswith('cuda') and not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
 ds=MangaLayoutDataset(a.manifest,cfg,'train',training=False)
 batches=[int(x) for x in cfg['preflight']['real_model_batches']]
 results=[]; selected=None
 total_mem=torch.cuda.get_device_properties(0).total_memory if a.device.startswith('cuda') else 0
 for bs in batches:
  try:
   if a.device.startswith('cuda'): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
   model=build_model(cfg,pretrained_backbone=None).to(a.device).train()
   items=[ds[i%len(ds)] for i in range(bs)]; images,targets=collate(items); images=images.to(a.device)
   for t in targets: t['boxes']=t['boxes'].to(a.device); t['labels']=t['labels'].to(a.device); t['balloon_masks']=t['balloon_masks'].to(a.device)
   opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
   # warm one step + timed two steps on actual graph
   timings=[]
   for _ in range(5):
    t0=time.perf_counter(); opt.zero_grad(set_to_none=True); loss=compute_loss(model(images),targets,cfg)['total']; loss.backward(); opt.step()
    if a.device.startswith('cuda'): torch.cuda.synchronize()
    timings.append(time.perf_counter()-t0)
   reserved=torch.cuda.max_memory_reserved() if a.device.startswith('cuda') else 0
   frac=reserved/max(total_mem,1)
   measured=timings[1:]
   median=statistics.median(measured)
   row={'batch':bs,'status':'PASS','loss':float(loss.detach().cpu()),'seconds_per_step_median':median,'seconds_per_step_samples':measured,'samples_per_second':bs/median,'peak_reserved_gib':reserved/(1024**3),'vram_fraction':frac}
   results.append(row)
   del model,opt,images,targets
  except RuntimeError as e:
   results.append({'batch':bs,'status':'FAIL','error':str(e)[:500]})
   if a.device.startswith('cuda'): torch.cuda.empty_cache()
 safe=[r for r in results if r['status']=='PASS' and (not a.device.startswith('cuda') or r['vram_fraction']<=float(cfg['preflight']['max_reserved_vram_fraction']))]
 if safe: selected=int(max(safe,key=lambda r:r['samples_per_second'])['batch'])
 if selected is None: raise SystemExit('no safe batch size found')
 eff=int(cfg['training']['effective_batch_size']); accum=max(1,(eff+selected-1)//selected)
 report={'schema_version':'manga-layout4-preflight-v1','device':a.device,'gpu':torch.cuda.get_device_name(0) if a.device.startswith('cuda') else None,'torch':torch.__version__,'results':results,'batch_selection':'highest measured samples_per_second under the configured VRAM safety ceiling','recommended_micro_batch':selected,'recommended_gradient_accumulation':accum,'recommended_effective_batch_size':selected*accum,'formal_gate':'PASS'}
 out=ROOT/'outputs/preflight'; out.mkdir(parents=True,exist_ok=True); (out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
