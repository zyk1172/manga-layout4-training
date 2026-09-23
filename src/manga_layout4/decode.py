from __future__ import annotations
from typing import Any
import torch
from torch import Tensor
from torchvision.ops import batched_nms
from . import CLASSES, CLASS_TO_ID
from .losses import boxes_from_ltrb, locations
from .model import STRIDES


@torch.no_grad()
def decode_one(outputs:dict[str,Any], batch_index:int, cfg:dict[str,Any], pre_nms_topk:int=1200):
    size=int(cfg["input"]["image_size"]); all_boxes=[]; all_scores=[]; all_labels=[]; all_coeff=[]
    for li,stride in enumerate(STRIDES):
        cls=outputs["cls_logits"][li][batch_index].permute(1,2,0).reshape(-1,len(CLASSES))
        reg=outputs["bbox_reg"][li][batch_index].permute(1,2,0).reshape(-1,4)
        coeff=outputs["mask_coeff"][li][batch_index].permute(1,2,0).reshape(-1,outputs["mask_coeff"][li].shape[1])
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
        all_boxes.append(boxes); all_scores.append(scores); all_labels.append(labels); all_coeff.append(coeff[point_idx])
    if not all_boxes:
        z=outputs["cls_logits"][0].new_empty
        return {"boxes":z((0,4)),"scores":z((0,)),"labels":torch.empty((0,),device=outputs["cls_logits"][0].device,dtype=torch.long),
                "mask_coeffs":z((0,outputs["mask_prototypes"].shape[1]))}
    boxes=torch.cat(all_boxes); scores=torch.cat(all_scores); labels=torch.cat(all_labels); coeffs=torch.cat(all_coeff)
    kept=[]
    for cid,name in enumerate(CLASSES):
        ids=torch.nonzero(labels==cid).squeeze(1)
        if len(ids):
            k=batched_nms(boxes[ids],scores[ids],torch.zeros_like(ids),float(cfg["postprocess"]["nms_iou"][name]))
            kept.append(ids[k])
    if kept:
        keep=torch.cat(kept); keep=keep[scores[keep].argsort(descending=True)[:int(cfg["postprocess"]["max_detections"])]]
        boxes,scores,labels,coeffs=boxes[keep],scores[keep],labels[keep],coeffs[keep]
    return {"boxes":boxes,"scores":scores,"labels":labels,"mask_coeffs":coeffs}


@torch.no_grad()
def balloon_instance_masks(outputs:dict[str,Any], detection:dict[str,Tensor], batch_index:int, cfg:dict[str,Any]) -> tuple[Tensor, Tensor]:
    """Return (detection_indices, bool masks) for balloon detections at prototype resolution."""
    balloon_id=CLASS_TO_ID["balloon"]
    ids=torch.nonzero(detection["labels"]==balloon_id).squeeze(1)
    proto=outputs["mask_prototypes"][batch_index]
    h,w=proto.shape[-2:]
    if not len(ids):
        return ids, torch.empty((0,h,w),device=proto.device,dtype=torch.bool)
    coeff=detection["mask_coeffs"][ids].tanh()
    logits=torch.einsum("nk,khw->nhw",coeff,proto)
    masks=logits.sigmoid()>=float(cfg["postprocess"]["balloon_mask_threshold"])
    size=float(cfg["input"]["image_size"])
    yy=torch.arange(h,device=proto.device).view(1,h,1)
    xx=torch.arange(w,device=proto.device).view(1,1,w)
    boxes=detection["boxes"][ids]
    x1=(boxes[:,0]*w/size).floor().clamp(0,w).view(-1,1,1)
    y1=(boxes[:,1]*h/size).floor().clamp(0,h).view(-1,1,1)
    x2=(boxes[:,2]*w/size).ceil().clamp(0,w).view(-1,1,1)
    y2=(boxes[:,3]*h/size).ceil().clamp(0,h).view(-1,1,1)
    crop=(xx>=x1)&(xx<x2)&(yy>=y1)&(yy<y2)
    return ids, masks & crop
