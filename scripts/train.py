#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from manga_layout4.config import load_config
from manga_layout4.data import MangaLayoutDataset,collate,weighted_sampler
from manga_layout4.losses import compute_loss
from manga_layout4.metrics import evaluate
from manga_layout4.model import build_model
from manga_layout4.train_utils import EMA,lr_at,seed_everything

def _params(model,base_lr,backbone_mult,wd):
 groups=[]
 buckets={}
 for name,p in model.named_parameters():
  if not p.requires_grad: continue
  is_backbone=name.startswith('backbone.')
  no_decay=(p.ndim==1 or name.endswith('.bias'))
  key=(is_backbone,no_decay)
  buckets.setdefault(key,[]).append(p)
 for (bb,nd),ps in buckets.items():
  groups.append({'params':ps,'lr':base_lr*(backbone_mult if bb else 1.0),'weight_decay':0.0 if nd else wd,'lr_mult':backbone_mult if bb else 1.0})
 return groups

def _move(targets,device):
 for t in targets:
  for k in ('boxes','labels','balloon_mask'): t[k]=t[k].to(device,non_blocking=True)
 return targets

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'configs/base.yaml'); p.add_argument('--manifest',type=Path,default=ROOT/'data/manifests/pages.jsonl'); p.add_argument('--device',default='cuda'); p.add_argument('--output',type=Path,required=True); p.add_argument('--micro-batch',type=int); p.add_argument('--grad-accum',type=int); p.add_argument('--resume',type=Path)
 a=p.parse_args(); cfg=load_config(a.config)
 audit=ROOT/'data/manifests/audit.json'
 if not audit.is_file() or json.loads(audit.read_text()).get('status')!='PASS': raise SystemExit('dataset audit gate not PASS')
 pf_path=ROOT/'outputs/preflight/report.json'; of_path=ROOT/'outputs/overfit/report.json'
 if not pf_path.is_file() or json.loads(pf_path.read_text()).get('formal_gate')!='PASS': raise SystemExit('real-model preflight gate not PASS')
 if not of_path.is_file() or json.loads(of_path.read_text()).get('status')!='PASS': raise SystemExit('overfit gate not PASS')
 if cfg['split_policy'].get('allow_legacy_test_during_training'): raise SystemExit('legacy test must remain blocked during training')
 if cfg.get('export',{}).get('require_pretrain_coreml_gate',False):
  coreml_gate=ROOT/str(cfg['export']['pretrain_coreml_gate_report'])
  if not coreml_gate.is_file() or json.loads(coreml_gate.read_text()).get('status')!='PASS':
   raise SystemExit(f'pre-training Core ML equivalence gate not PASS: {coreml_gate}')
 import torch
 seed=int(cfg['project']['seed']); seed_everything(seed); device=torch.device(a.device)
 pf=json.loads(pf_path.read_text()); bs=a.micro_batch or int(pf['recommended_micro_batch']); accum=a.grad_accum or int(pf['recommended_gradient_accumulation'])
 train_ds=MangaLayoutDataset(a.manifest,cfg,'train',True); val_ds=MangaLayoutDataset(a.manifest,cfg,'val',False)
 sampler=weighted_sampler(train_ds,cfg,seed)
 nw=int(cfg['training']['num_workers'])
 train_loader=torch.utils.data.DataLoader(train_ds,batch_size=bs,sampler=sampler,num_workers=nw,pin_memory=bool(cfg['training']['pin_memory']),persistent_workers=bool(cfg['training']['persistent_workers']) and nw>0,collate_fn=collate,drop_last=True)
 val_loader=torch.utils.data.DataLoader(val_ds,batch_size=bs,shuffle=False,num_workers=max(0,min(2,nw)),pin_memory=bool(cfg['training']['pin_memory']),persistent_workers=False,collate_fn=collate)
 model=build_model(cfg,None).to(device)
 base_lr=float(cfg['training']['learning_rate']); bb_mult=float(cfg['training']['backbone_lr_multiplier']); wd=float(cfg['training']['weight_decay'])
 opt=torch.optim.AdamW(_params(model,base_lr,bb_mult,wd),betas=(0.9,0.999))
 ema=EMA(model,float(cfg['training']['ema_decay']))
 start_epoch=0; global_step=0; best=-1.0; history=[]; bad_evals=0
 if a.resume:
  ck=torch.load(a.resume,map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); ema.model.load_state_dict(ck.get('ema',ck['model'])); opt.load_state_dict(ck['optimizer']); start_epoch=int(ck['epoch']); global_step=int(ck['global_step']); best=float(ck.get('best_composite',-1)); history=list(ck.get('history',[]))
 epochs=int(cfg['training']['epochs']); total_opt_steps=math.ceil(len(train_loader)/accum)*epochs; stage1_epochs=round(epochs*float(cfg['augmentation']['stage1_fraction'])); warm=int(cfg['training']['warmup_optimizer_steps']); min_ratio=float(cfg['training']['min_lr_ratio'])
 a.output.mkdir(parents=True,exist_ok=True)
 for epoch in range(start_epoch,epochs):
  train_ds.set_stage1(epoch<stage1_epochs); model.train(); sums={'total':0.0,'classification':0.0,'bbox':0.0,'mask':0.0}; micro=0; opt.zero_grad(set_to_none=True); t0=time.perf_counter()
  for images,targets in train_loader:
   images=images.to(device,non_blocking=True); targets=_move(targets,device); losses=compute_loss(model(images),targets,cfg); loss=losses['total']/accum
   if not torch.isfinite(loss): raise RuntimeError(f'non-finite loss epoch={epoch+1} micro={micro}')
   loss.backward(); micro+=1
   for k in sums: sums[k]+=float(losses[k].detach().cpu())
   if micro%accum==0 or micro==len(train_loader):
    torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['gradient_clip_norm']))
    lr=lr_at(global_step,total_opt_steps,warm,base_lr,min_ratio)
    for g in opt.param_groups: g['lr']=lr*float(g.get('lr_mult',1.0))
    opt.step(); opt.zero_grad(set_to_none=True); ema.update(model); global_step+=1
  row={'epoch':epoch+1,'global_step':global_step,'stage':'strong' if epoch<stage1_epochs else 'weak','train_loss':{k:v/max(1,micro) for k,v in sums.items()},'lr':opt.param_groups[-1]['lr'],'seconds':time.perf_counter()-t0}
  do_val=((epoch+1)%int(cfg['training']['val_every_epochs'])==0 or epoch+1==epochs or epoch+1==stage1_epochs)
  if do_val:
   metrics=evaluate(ema.model,val_loader,cfg,device); row['val']=metrics
   score=float(metrics['composite'])
   if score>best:
    best=score; bad_evals=0
    torch.save({'schema_version':'manga-layout4-checkpoint-v1','model':model.state_dict(),'ema':ema.model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'global_step':global_step,'best_composite':best,'config':cfg,'history':history+[row]},a.output/'best.pt')
   else: bad_evals+=1
  history.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
  if (epoch+1)%int(cfg['training']['checkpoint_every_epochs'])==0 or epoch+1==epochs:
   torch.save({'schema_version':'manga-layout4-checkpoint-v1','model':model.state_dict(),'ema':ema.model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'global_step':global_step,'best_composite':best,'config':cfg,'history':history},a.output/'last.pt')
   (a.output/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  if do_val and epoch+1>=int(cfg['training']['early_stop_min_epoch']) and bad_evals>=int(cfg['training']['early_stop_patience_evals']):
   break
 print(json.dumps({'status':'DONE','best_composite':best,'epochs_completed':history[-1]['epoch'],'output':str(a.output)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
