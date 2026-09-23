#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from manga_layout4.config import load_config
from manga_layout4.manifest import build_manifest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'configs/base.yaml'); p.add_argument('--output',type=Path,default=ROOT/'data/manifests/pages.jsonl')
    a=p.parse_args(); r=build_manifest(load_config(a.config),a.output); print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
