from __future__ import annotations
from typing import Any, Sequence
import torch
from torch import Tensor
from torch.nn import functional as F
from . import CLASSES, CLASS_TO_ID
from .model import STRIDES


def locations(h:int,w:int,stride:int,device,dtype):
    ys=(torch.arange(h,device=device,dtype=dtype)+0.5)*stride
    xs=(torch.arange(w,device=device,dtype=dtype)+0.5)*stride
    yy,xx=torch.meshgrid(ys,xs,indexing="ij")
    return torch.stack([xx.reshape(-1),yy.reshape(-1)],1)


def assign_targets(image_size:int,target:dict[str,Any],size_ranges:Sequence[Sequence[float]],center_radius:float=1.5):
    boxes=target["boxes"]; labels=target["labels"]; device=boxes.device; dtype=boxes.dtype
    areas=(boxes[:,2]-boxes[:,0]).clamp_min(0)*(boxes[:,3]-boxes[:,1]).clamp_min(0) if len(boxes) else boxes.new_empty((0,))
    out=[]
    for stride,(lower,upper) in zip(STRIDES,size_ranges):
        h=(image_size+stride-1)//stride; w=h; pts=locations(h,w,stride,device,dtype); n=len(pts)
        assigned=torch.full((n,),-1,device=device,dtype=torch.long)
        assigned_gt=torch.full((n,),-1,device=device,dtype=torch.long)
        gt_boxes=torch.zeros((n,4),device=device,dtype=dtype)
        if len(boxes):
            l=pts[:,0,None]-boxes[None,:,0]; t=pts[:,1,None]-boxes[None,:,1]
            r=boxes[None,:,2]-pts[:,0,None]; b=boxes[None,:,3]-pts[:,1,None]
            ltrb=torch.stack([l,t,r,b],-1)
            inside=ltrb.min(-1).values>=0
            maxd=ltrb.max(-1).values
            inrange=(maxd>=float(lower))&(maxd<=float(upper))
            cx=(boxes[:,0]+boxes[:,2])/2; cy=(boxes[:,1]+boxes[:,3])/2; rad=stride*center_radius
            cbox=torch.stack([torch.maximum(boxes[:,0],cx-rad),torch.maximum(boxes[:,1],cy-rad),torch.minimum(boxes[:,2],cx+rad),torch.minimum(boxes[:,3],cy+rad)],1)
            cl=pts[:,0,None]-cbox[None,:,0]; ct=pts[:,1,None]-cbox[None,:,1]
            cr=cbox[None,:,2]-pts[:,0,None]; cb=cbox[None,:,3]-pts[:,1,None]
            center=torch.stack([cl,ct,cr,cb],-1).min(-1).values>=0
            cand=inside&inrange&center
            has=cand.any(0); cand=torch.where(has[None,:],cand,inside&inrange)
            costs=torch.where(cand,areas[None,:],torch.full_like(areas[None,:],float("inf")))
            minarea,idx=costs.min(1); pos=torch.isfinite(minarea)
            assigned[pos]=labels[idx[pos]]; assigned_gt[pos]=idx[pos]; gt_boxes[pos]=boxes[idx[pos]]
        out.append({"points":pts,"labels":assigned,"gt_indices":assigned_gt,"gt_boxes":gt_boxes})
    return out


def box_iou_aligned(a:Tensor,b:Tensor)->Tensor:
    ix1=torch.maximum(a[:,0],b[:,0]); iy1=torch.maximum(a[:,1],b[:,1]); ix2=torch.minimum(a[:,2],b[:,2]); iy2=torch.minimum(a[:,3],b[:,3])
    inter=(ix2-ix1).clamp_min(0)*(iy2-iy1).clamp_min(0)
    aa=(a[:,2]-a[:,0]).clamp_min(0)*(a[:,3]-a[:,1]).clamp_min(0)
    bb=(b[:,2]-b[:,0]).clamp_min(0)*(b[:,3]-b[:,1]).clamp_min(0)
    return inter/(aa+bb-inter).clamp_min(1e-6)


def giou_loss(a:Tensor,b:Tensor)->Tensor:
    iou=box_iou_aligned(a,b)
    ex1=torch.minimum(a[:,0],b[:,0]); ey1=torch.minimum(a[:,1],b[:,1]); ex2=torch.maximum(a[:,2],b[:,2]); ey2=torch.maximum(a[:,3],b[:,3])
    enc=(ex2-ex1).clamp_min(0)*(ey2-ey1).clamp_min(0)
    aa=(a[:,2]-a[:,0]).clamp_min(0)*(a[:,3]-a[:,1]).clamp_min(0)
    bb=(b[:,2]-b[:,0]).clamp_min(0)*(b[:,3]-b[:,1]).clamp_min(0)
    ix1=torch.maximum(a[:,0],b[:,0]); iy1=torch.maximum(a[:,1],b[:,1]); ix2=torch.minimum(a[:,2],b[:,2]); iy2=torch.minimum(a[:,3],b[:,3])
    inter=(ix2-ix1).clamp_min(0)*(iy2-iy1).clamp_min(0); union=aa+bb-inter
    giou=iou-(enc-union)/enc.clamp_min(1e-6)
    return 1-giou


def boxes_from_ltrb(points:Tensor,raw:Tensor,stride:int)->Tensor:
    d=F.softplus(raw)*float(stride)
    return torch.stack([points[:,0]-d[:,0],points[:,1]-d[:,1],points[:,0]+d[:,2],points[:,1]+d[:,3]],1)


def quality_focal(logits:Tensor,targets:Tensor,beta:float)->Tensor:
    p=logits.sigmoid()
    bce=F.binary_cross_entropy_with_logits(logits,targets,reduction="none")
    return bce*(targets-p).abs().pow(beta)


def _instance_mask_loss(logits: Tensor, target: Tensor, crop: Tensor) -> tuple[Tensor, Tensor]:
    if not crop.any():
        zero = logits.sum() * 0
        return zero, zero
    bce = F.binary_cross_entropy_with_logits(logits[crop], target[crop])
    p = logits.sigmoid() * crop
    t = target * crop
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
    return bce, dice


def compute_loss(outputs:dict[str,Any],targets:list[dict[str,Any]],cfg:dict[str,Any])->dict[str,Tensor]:
    size=int(cfg["input"]["image_size"]); ranges=cfg["model"]["size_ranges"]
    beta=float(cfg["loss"]["quality_focal_beta"]); box_w=float(cfg["loss"]["box_giou_weight"])
    class_weights=outputs["cls_logits"][0].new_tensor([float(cfg["loss"]["positive_class_weights"][c]) for c in CLASSES])
    cls_total=outputs["cls_logits"][0].sum()*0; box_total=cls_total.clone(); pos_count=cls_total.clone()
    mask_bce_total=cls_total.clone(); mask_dice_total=cls_total.clone(); mask_instances=0
    balloon_id=CLASS_TO_ID["balloon"]

    for bi,target in enumerate(targets):
        boxes=target["boxes"].to(outputs["cls_logits"][0].device); labels=target["labels"].to(boxes.device)
        balloon_masks=target["balloon_masks"].to(boxes.device)
        assigns=assign_targets(size,{"boxes":boxes,"labels":labels},ranges,float(cfg["assignment"]["center_sampling_radius"]))
        coeff_candidates: dict[int, list[tuple[Tensor, Tensor]]] = {}
        for li,stride in enumerate(STRIDES):
            cls=outputs["cls_logits"][li][bi].permute(1,2,0).reshape(-1,len(CLASSES))
            reg=outputs["bbox_reg"][li][bi].permute(1,2,0).reshape(-1,4)
            coeff=outputs["mask_coeff"][li][bi].permute(1,2,0).reshape(-1,outputs["mask_coeff"][li].shape[1])
            a=assigns[li]; pos=a["labels"]>=0
            qtargets=torch.zeros_like(cls)
            if pos.any():
                pred=boxes_from_ltrb(a["points"][pos],reg[pos],stride)
                gt=a["gt_boxes"][pos]
                quality=box_iou_aligned(pred.detach(),gt).clamp(0,1)
                plabel=a["labels"][pos]
                gt_indices=a["gt_indices"][pos]
                qtargets[pos,plabel]=quality
                bw=class_weights[plabel]
                box_total=box_total+(giou_loss(pred,gt)*bw).sum()*box_w
                pos_count=pos_count+bw.sum()
                pos_indices=torch.nonzero(pos).squeeze(1)
                for gt_index in torch.unique(gt_indices[plabel == balloon_id]).tolist():
                    local=torch.nonzero((gt_indices == gt_index) & (plabel == balloon_id)).squeeze(1)
                    if not len(local):
                        continue
                    q=quality[local]
                    chosen_local=local[int(torch.argmax(q))]
                    chosen_point=pos_indices[chosen_local]
                    coeff_candidates.setdefault(int(gt_index), []).append(
                        (quality[chosen_local].detach(), coeff[chosen_point])
                    )
            qfl=quality_focal(cls,qtargets,beta)
            if pos.any():
                plabel=a["labels"][pos]
                qfl[pos,plabel]*=class_weights[plabel]
            cls_total=cls_total+qfl.sum()

        balloon_gt_indices=torch.nonzero(labels == balloon_id).squeeze(1).tolist()
        if len(balloon_gt_indices) != len(balloon_masks):
            raise RuntimeError("balloon instance-mask count does not match balloon GT count")
        mask_index_by_gt={gt_index:i for i,gt_index in enumerate(balloon_gt_indices)}
        prototypes=outputs["mask_prototypes"][bi]
        mh,mw=prototypes.shape[-2:]
        for gt_index, mask_index in mask_index_by_gt.items():
            candidates=coeff_candidates.get(int(gt_index))
            if not candidates:
                continue
            qualities=torch.stack([item[0] for item in candidates])
            coeffs=torch.stack([item[1] for item in candidates])
            coeff=coeffs[torch.argmax(qualities)].tanh()
            logits=(prototypes * coeff[:,None,None]).sum(0)
            target_mask=balloon_masks[mask_index]
            box=boxes[gt_index]
            x1=max(0,min(mw,int(torch.floor(box[0] * mw / size).item())))
            y1=max(0,min(mh,int(torch.floor(box[1] * mh / size).item())))
            x2=max(0,min(mw,int(torch.ceil(box[2] * mw / size).item())))
            y2=max(0,min(mh,int(torch.ceil(box[3] * mh / size).item())))
            crop=torch.zeros((mh,mw),device=boxes.device,dtype=torch.bool)
            if x2>x1 and y2>y1: crop[y1:y2,x1:x2]=True
            bce,dice=_instance_mask_loss(logits,target_mask,crop)
            mask_bce_total=mask_bce_total+bce; mask_dice_total=mask_dice_total+dice; mask_instances+=1

    norm=pos_count.clamp_min(1.0)
    cls_loss=cls_total/norm; box_loss=box_total/norm
    if mask_instances:
        bce=mask_bce_total/mask_instances; dice=mask_dice_total/mask_instances
    else:
        bce=mask_bce_total; dice=mask_dice_total
    mask_loss=bce*float(cfg["loss"]["mask_bce_weight"])+dice*float(cfg["loss"]["mask_dice_weight"])
    total=cls_loss+box_loss+mask_loss
    return {"total":total,"classification":cls_loss,"bbox":box_loss,"mask":mask_loss,
            "mask_bce":bce,"mask_dice":dice,"weighted_positive":pos_count.detach(),
            "mask_instances":torch.tensor(float(mask_instances),device=total.device)}
