#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from manga_layout4.config import load_config

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'configs/base.yaml'); p.add_argument('--trace',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); cfg=load_config(a.config)
 import torch, coremltools as ct
 model=torch.jit.load(str(a.trace),map_location='cpu').eval(); size=int(cfg['input']['image_size'])
 names=['p2_cls','p2_bbox','p2_mask_coeff','p3_cls','p3_bbox','p3_mask_coeff','p4_cls','p4_bbox','p4_mask_coeff','p5_cls','p5_bbox','p5_mask_coeff','mask_prototypes']
 target=getattr(ct.target,str(cfg['export']['minimum_deployment_target']))
 precision=ct.precision.FLOAT32 if str(cfg['export']['coreml_precision']).lower()=='fp32' else ct.precision.FLOAT16
 ml=ct.convert(model,convert_to='mlprogram',inputs=[ct.TensorType(name='image',shape=(1,3,size,size))],outputs=[ct.TensorType(name=n) for n in names],minimum_deployment_target=target,compute_precision=precision,compute_units=ct.ComputeUnit.ALL)
 a.output.parent.mkdir(parents=True,exist_ok=True); ml.save(str(a.output))
 report={'schema_version':'manga-layout4-coreml-export-v1','trace':str(a.trace),'output':str(a.output),'outputs':names,'precision':str(cfg['export']['coreml_precision']),'minimum_deployment_target':str(cfg['export']['minimum_deployment_target']),'status':'PASS'}
 (a.output.parent/'coreml-export.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
