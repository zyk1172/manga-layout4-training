from __future__ import annotations
from typing import Any
import torch
from torch import Tensor
from torchvision.ops import batched_nms
from . import CLASSES
from .losses import boxes_from_ltrb, locations
from .model import STRIDES

@torch.no_grad()
def decode_one(outputs:dict[str,Any], batch_index:int, cfg:dict[str,Any], pre_nms_topk:int=1200):
    size=int(cfg["input"]["image_size"]); all_boxes=[]; all_scores=[]; all_labels=[]
    for li,stride in enumerate(STRIDES):
        cls=outputs["cls_logits"][li][batch_index].permute(1,2,0).reshape(-1,len(CLASSES))
        reg=outputs["bbox_reg"][li][batch_index].permute(1,2,0).reshape(-1,4)
        h=outputs["cls_logits"][li].shape[-2]; w=outputs["cls_logits"][li].shape[-1]
        pts=locations(h,w,stride,cls.device,cls.dtype)
        probs=cls.sigmoid(); flat=probs.flatten()
        k=min(pre_nms_topk,flat.numel())
        scores,idx=torch.topk(flat,k)
        keep=scores>=float(cfg["postprocess"]["score_threshold"])
        scores=scores[keep]; idx=idx[keep]
        if not len(scores): continue
        point_idx=torch.div(idx,len(CLASSES),rounding_mode="floor"); labels=idx%len(CLASSES)
        boxes=boxes_from_ltrb(pts[point_idx],reg[point_idx],stride).clamp(0,size)
        all_boxes.append(boxes); all_scores.append(scores); all_labels.append(labels)
    if not all_boxes:
        z=outputs["cls_logits"][0].new_empty
        return {"boxes":z((0,4)),"scores":z((0,)),"labels":torch.empty((0,),device=outputs["cls_logits"][0].device,dtype=torch.long)}
    boxes=torch.cat(all_boxes); scores=torch.cat(all_scores); labels=torch.cat(all_labels)
    # class-specific NMS by applying NMS separately; batched_nms only supports one IoU.
    kept=[]
    for cid,name in enumerate(CLASSES):
        ids=torch.nonzero(labels==cid).squeeze(1)
        if len(ids):
            k=batched_nms(boxes[ids],scores[ids],torch.zeros_like(ids),float(cfg["postprocess"]["nms_iou"][name]))
            kept.append(ids[k])
    if kept:
        keep=torch.cat(kept); keep=keep[scores[keep].argsort(descending=True)[:int(cfg["postprocess"]["max_detections"])]]
        boxes,scores,labels=boxes[keep],scores[keep],labels[keep]
    return {"boxes":boxes,"scores":scores,"labels":labels}
