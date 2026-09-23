from __future__ import annotations
from collections import defaultdict
import math
import torch
from torch.nn import functional as F

from . import CLASSES
from .decode import decode_one, balloon_instance_masks


def _iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0.0,ix2-ix1)*max(0.0,iy2-iy1)
    aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]); bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
    return inter/max(aa+bb-inter,1e-9)


def _match_box_predictions(preds, gts, threshold: float):
    matched={key:set() for key in gts}; rows=[]; duplicate=0
    for score,page,box in sorted(preds,key=lambda x:-x[0]):
        candidates=gts.get(page,[]); best=-1; best_iou=0.0; best_any=0.0
        for j,gt in enumerate(candidates):
            value=_iou(box,gt); best_any=max(best_any,value)
            if j in matched.setdefault(page,set()): continue
            if value>best_iou: best_iou=value; best=j
        tp=best>=0 and best_iou>=threshold
        if tp: matched[page].add(best)
        elif best_any>=threshold: duplicate+=1
        rows.append((score,1.0 if tp else 0.0,0.0 if tp else 1.0))
    return rows,matched,duplicate


def _ap_from_rows(rows,total:int):
    if total==0: return float("nan")
    if not rows: return 0.0
    tp=torch.tensor([r[1] for r in rows]).cumsum(0); fp=torch.tensor([r[2] for r in rows]).cumsum(0)
    rec=tp/total; prec=tp/(tp+fp).clamp_min(1e-9); vals=[]
    for recall_point in torch.linspace(0,1,101):
        c=prec[rec>=recall_point]; vals.append(float(c.max()) if len(c) else 0.0)
    return sum(vals)/101.0


def _box_ap(preds,gts,threshold:float):
    total=sum(len(v) for v in gts.values())
    rows,_,_=_match_box_predictions(preds,gts,threshold)
    return _ap_from_rows(rows,total)


def _box_diagnostics(preds,gts,page_count:int):
    total=sum(len(v) for v in gts.values()); rows,matched,duplicates=_match_box_predictions(preds,gts,0.5)
    tp=sum(len(v) for v in matched.values()); fp=sum(int(r[2]) for r in rows)
    return {"gt":total,"predictions":len(preds),"recall50":tp/max(total,1),"fp":fp,
            "fp_per_page":fp/max(page_count,1),"duplicate_fp":duplicates}


def _boundary_band(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    x=mask.float()[None,None]
    eroded=1-F.max_pool2d(1-x,3,1,1)
    edge=((x-eroded)>0.5).float()
    if radius>0: edge=F.max_pool2d(edge,2*radius+1,1,radius)
    return edge[0,0].bool()


def _mask_similarity(pred:torch.Tensor,gt:torch.Tensor)->float:
    inter=(pred&gt).sum().item(); union=(pred|gt).sum().item()
    return inter/max(union,1)


def _boundary_similarity(pred:torch.Tensor,gt:torch.Tensor)->float:
    p=_boundary_band(pred); g=_boundary_band(gt)
    inter=(p&g).sum().item(); union=(p|g).sum().item()
    if union==0: return 1.0
    return inter/union


def _pairwise_similarity(pred_masks: torch.Tensor, gt_masks: torch.Tensor, boundary: bool=False) -> list[list[float]]:
    result=[]
    for p in pred_masks:
        row=[]
        for g in gt_masks:
            row.append(_boundary_similarity(p,g) if boundary else _mask_similarity(p,g))
        result.append(row)
    return result


def _similarity_ap(records, gt_counts:dict[str,int], threshold:float):
    total=sum(gt_counts.values()); matched={page:set() for page in gt_counts}; rows=[]; duplicate=0
    for score,page,sims in sorted(records,key=lambda x:-x[0]):
        best=-1; best_value=0.0; best_any=max(sims,default=0.0)
        for j,value in enumerate(sims):
            if j in matched.setdefault(page,set()): continue
            if value>best_value: best_value=value; best=j
        tp=best>=0 and best_value>=threshold
        if tp: matched[page].add(best)
        elif best_any>=threshold: duplicate+=1
        rows.append((score,1.0 if tp else 0.0,0.0 if tp else 1.0))
    return _ap_from_rows(rows,total), duplicate


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval(); preds={c:[] for c in CLASSES}; gts={c:defaultdict(list) for c in CLASSES}
    mask_records=[]; boundary_records=[]; mask_gt_counts={}; page_count=0
    for images,targets in loader:
        images=images.to(device); out=model(images)
        for bi,target in enumerate(targets):
            page_count+=1; page=str(target["image_id"]); det=decode_one(out,bi,cfg)
            for box,label,score in zip(det["boxes"].cpu().tolist(),det["labels"].cpu().tolist(),det["scores"].cpu().tolist()):
                preds[CLASSES[label]].append((float(score),page,[float(x) for x in box]))
            for box,label in zip(target["boxes"].tolist(),target["labels"].tolist()):
                gts[CLASSES[label]][page].append([float(x) for x in box])
            det_ids,pred_masks=balloon_instance_masks(out,det,bi,cfg)
            gt_masks=target["balloon_masks"].bool()
            pred_masks=pred_masks.cpu(); gt_masks=gt_masks.cpu(); mask_gt_counts[page]=len(gt_masks)
            mask_sims=_pairwise_similarity(pred_masks,gt_masks,False)
            boundary_sims=_pairwise_similarity(pred_masks,gt_masks,True)
            for local_i,det_i in enumerate(det_ids.cpu().tolist()):
                score=float(det["scores"][det_i].cpu())
                mask_records.append((score,page,mask_sims[local_i]))
                boundary_records.append((score,page,boundary_sims[local_i]))

    metrics={}
    for class_name in CLASSES:
        metrics[f"{class_name}_ap50"]=_box_ap(preds[class_name],gts[class_name],0.50)
        metrics[f"{class_name}_ap75"]=_box_ap(preds[class_name],gts[class_name],0.75)
    metrics["balloon_box_ap50"]=metrics.pop("balloon_ap50")
    metrics["balloon_box_ap75"]=metrics.pop("balloon_ap75")
    mask_ap,mask_dup=_similarity_ap(mask_records,mask_gt_counts,0.50)
    boundary_ap,boundary_dup=_similarity_ap(boundary_records,mask_gt_counts,0.50)
    metrics["balloon_mask_ap50"]=mask_ap; metrics["balloon_boundary_ap50"]=boundary_ap
    metrics["page_count"]=page_count
    metrics["detection_diagnostics"]={c:_box_diagnostics(preds[c],gts[c],page_count) for c in CLASSES}
    metrics["balloon_instance_diagnostics"]={
        "gt":sum(mask_gt_counts.values()),"predictions":len(mask_records),
        "mask_duplicate_fp":mask_dup,"boundary_duplicate_fp":boundary_dup,
        "mask_fp_per_page":max(0,len(mask_records)-sum(mask_gt_counts.values()))/max(page_count,1),
    }
    weights=cfg["selection"]["composite_weights"]
    terms={
        "frame_ap50":metrics["frame_ap50"],"text_ap50":metrics["text_ap50"],
        "balloon_box_ap50":metrics["balloon_box_ap50"],"onomatopoeia_ap50":metrics["onomatopoeia_ap50"],
        "balloon_mask_ap50":metrics["balloon_mask_ap50"],"balloon_boundary_ap50":metrics["balloon_boundary_ap50"],
    }
    numerator=sum(float(weights[k])*float(v) for k,v in terms.items() if not math.isnan(float(v)))
    denominator=sum(float(weights[k]) for k,v in terms.items() if not math.isnan(float(v)))
    metrics["composite"]=numerator/max(denominator,1e-9)
    return metrics
