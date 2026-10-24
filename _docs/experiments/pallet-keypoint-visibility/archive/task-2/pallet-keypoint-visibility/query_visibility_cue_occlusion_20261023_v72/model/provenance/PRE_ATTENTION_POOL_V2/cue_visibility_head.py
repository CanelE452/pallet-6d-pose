"""Small source semantic-cue pilot head; backbone features are frozen upstream.

No reference, mask, class, source ID or native semantic ID enters forward().
P3 is sampled from each actual full RGB at the fixed q/L native9 lattice.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CueVisibilityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(4,8,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(8,16,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(16,16,3,stride=2,padding=1),nn.SiLU())
        self.fusion = nn.Sequential(nn.Conv2d(88,32,3,padding=1),nn.SiLU(),
                                    nn.Conv2d(32,32,3,padding=1),nn.SiLU())
        self.classifier = nn.Sequential(nn.Linear(68,32),nn.SiLU(),nn.Linear(32,1))
        self.localizer = nn.Conv2d(32,1,1)
        yy,xx = torch.meshgrid(torch.arange(9),torch.arange(9),indexing='ij')
        self.register_buffer('native_offsets',torch.stack((xx-4,yy-4)).float()[None]/9)

    def forward(self, p3_roi, rgb96, crop_valid96, query_xy, box_xyxy, side_L):
        """p3_roi:N65x9x9 (P3=64 + native-valid=1); RGB uint8 N3x96x96.

        Geometry has no reference association. The returned binary score is a
        D/E-conditional score, not calibrated semantic-E or MISS/U evidence.
        """
        n = p3_roi.shape[0]
        if p3_roi.shape != (n,65,9,9) or rgb96.shape != (n,3,96,96):
            raise ValueError('Expected native9 P3 and CHW RGB96; no silent layout conversion')
        if crop_valid96.shape != (n,1,96,96) or query_xy.shape != (n,2) or box_xyxy.shape != (n,4) or side_L.shape != (n,):
            raise ValueError('Input-only geometry/valid shapes do not match')
        if rgb96.dtype != torch.uint8:
            raise ValueError('RGB96 must be raw uint8; normalization is fixed inside head')
        if not torch.isfinite(p3_roi).all() or not torch.isfinite(query_xy).all() or not torch.isfinite(box_xyxy).all() or not torch.isfinite(side_L).all() or not torch.all(side_L>0):
            raise ValueError('Nonfinite feature/geometry or nonpositive L')
        native_valid = p3_roi[:,64:65]
        if not torch.all((native_valid==0)|(native_valid==1)):
            raise ValueError('P3 channel64 must be the native sampling validity flag')
        valid = crop_valid96.to(p3_roi.dtype)
        if not torch.all((valid==0)|(valid==1)):
            raise ValueError('Crop valid must be binary')
        # Only a small RGB branch is trained; full-image YOLO P3 stays frozen.
        raw_rgb = rgb96.to(p3_roi.dtype)/255
        rgb = self.rgb_encoder(torch.cat((raw_rgb*valid,valid),dim=1))
        rgb = F.interpolate(rgb,size=(9,9),mode='bilinear',align_corners=False)
        valid9 = F.adaptive_avg_pool2d(valid,(9,9))
        geom = (torch.cat((box_xyxy[:,:2]-query_xy,box_xyxy[:,2:]-query_xy),dim=1)/side_L[:,None]).to(p3_roi.dtype)
        geom = torch.clamp(geom,-8,8)/8  # fixed numerical scale, no source/current fitting
        geom_map = geom[:,:,None,None].expand(-1,-1,9,9)
        offsets = self.native_offsets.to(p3_roi.dtype).expand(n,-1,-1,-1)
        fused = self.fusion(torch.cat((p3_roi[:,:64]*native_valid,native_valid,rgb,valid9,offsets,geom_map),dim=1))
        visible_cells = valid9>0
        avg = (fused*valid9).sum((2,3))/valid9.sum((2,3)).clamp_min(1e-8)
        mx = fused.masked_fill(~visible_cells,-1e4).amax((2,3))
        mx = torch.where(visible_cells.any((2,3)),mx,torch.zeros_like(mx))
        logit = self.classifier(torch.cat((avg,mx,geom),dim=1)).squeeze(1)
        heatmap = self.localizer(fused).squeeze(1)
        return {'logit_D_given_DE':logit,'cue_heatmap_logits':heatmap,
                'heatmap_input_valid':visible_cells.squeeze(1)}


def uncertainty_heatmap_target(uncertainty_bbox_xyxy, query_xy, side_L, dtype=torch.float32):
    """GT-only area-overlap target on native9 cells, not a forward input.

    Cell centers are q+(col/row-4)*L/9 and cell edges cover q +/- L/2.
    No Gaussian radius or fitted tolerance is introduced. Only D/FULL targets
    with positive in-crop area may use this output in the loss.
    """
    box = uncertainty_bbox_xyxy.to(dtype)
    q = query_xy.to(dtype)
    side = side_L.to(dtype)
    edges = (torch.arange(10,device=q.device,dtype=dtype)/9-.5)[None]
    ex = q[:,0:1]+side[:,None]*edges
    ey = q[:,1:2]+side[:,None]*edges
    dx = (torch.minimum(ex[:,1:],box[:,2:3])-torch.maximum(ex[:,:-1],box[:,0:1])).clamp_min(0)
    dy = (torch.minimum(ey[:,1:],box[:,3:4])-torch.maximum(ey[:,:-1],box[:,1:2])).clamp_min(0)
    mass = dy[:,:,None]*dx[:,None,:]
    return mass/mass.sum((1,2),keepdim=True).clamp_min(1e-12)


def semantic_cue_loss(outputs, class_target, full_available, input_valid,
                      source_index, heatmap_target, heatmap_weight=.25):
    """Source-equal BCE + .25 source-equal D-only uncertainty heatmap CE.

    class_target=-1 U, 0 E, 1 D. Non-FULL and invalid must be U. U and E
    heatmaps are unknown, not zero-valued negative heatmap supervision.
    Source IDs are loss metadata only; absent sources are excluded from means.
    """
    z,h = outputs['logit_D_given_DE'],outputs['cue_heatmap_logits']
    if not torch.all((class_target==-1)|(class_target==0)|(class_target==1)):
        raise ValueError('Only U=-1, E=0, D=1 are permitted')
    known = class_target>=0
    if torch.any(known & (~full_available.bool() | ~input_valid.bool())):
        raise ValueError('D/E supervision requires FULL availability and input-valid query')
    zero = z.sum()*0+h.sum()*0
    bce_sources,hm_sources = [],[]
    for source in torch.unique(source_index):
        mask = known & (source_index==source)
        if mask.any():
            bce_sources.append(F.binary_cross_entropy_with_logits(z[mask],class_target[mask].to(z.dtype)))
        positive = mask & (class_target==1)
        if positive.any():
            target = heatmap_target[positive].to(h.dtype)
            valid = outputs['heatmap_input_valid'][positive]
            if not torch.isfinite(target).all() or torch.any(target<0) or torch.any(target.masked_select(~valid)>0):
                raise ValueError('D heatmap target must be finite/nonnegative and input-valid')
            if not torch.allclose(target.sum((1,2)),torch.ones(len(target),device=h.device,dtype=h.dtype),atol=1e-5,rtol=0):
                raise ValueError('Each D/FULL heatmap must have unit GT target mass')
            logp = F.log_softmax(h[positive].masked_fill(~valid,-1e4).flatten(1),dim=1)
            hm_sources.append(-(target.flatten(1)*logp).sum(1).mean())
    bce = torch.stack(bce_sources).mean() if bce_sources else zero
    hm = torch.stack(hm_sources).mean() if hm_sources else zero
    return {'loss':bce+heatmap_weight*hm,'binary_BCE':bce,'D_heatmap_CE':hm,
            'known_count':int(known.sum()),'D_heatmap_count':int((known&(class_target==1)).sum()),
            'binary_sources':len(bce_sources),'D_heatmap_sources':len(hm_sources)}
