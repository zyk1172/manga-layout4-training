import yaml
from pathlib import Path
from manga_layout4 import CLASSES
ROOT=Path(__file__).resolve().parents[1]
def test_exact_four_classes():
 cfg=yaml.safe_load((ROOT/'configs/base.yaml').read_text())
 assert tuple(cfg['classes']['names'])==CLASSES
 assert cfg['model']['num_classes']==4
 assert 'face' not in cfg['classes']['names'] and 'body' not in cfg['classes']['names']
 assert cfg['split_policy']['allow_legacy_test_during_training'] is False
