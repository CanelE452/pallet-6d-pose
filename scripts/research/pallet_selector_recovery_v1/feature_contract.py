"""Predeclared, prediction-only feature names and units."""
from . import common as C

def names():
    n=['reprojection_rmse_px','reprojection_rmse_bboxnorm','cheirality_fraction',
       'lr_violations','tb_violations','front_rear_violations','invariant_violations','upright_alignment','spread_ratio',
       'cf_width_m','cf_height_m','cf_depth_m']
    n += [f't_{k}_m' for k in 'xyz']+[f't_{k}_objectnorm' for k in 'xyz']+[f'R_cf_{i}{j}' for i in range(3) for j in range(3)]
    n += [f'{f}_area_bboxnorm' for f in ('front','rear')]
    n += [f'{family}_edge_{s}_bboxnorm' for family in ('LR','TB','FR') for s in ('mean','min','max')]
    n += [f'front_rear_depth_{s}_{u}' for u in ('m','objectnorm') for s in ('mean','min','max')]
    n += [f'residual{k}_{u}' for u in ('px','bboxnorm') for k in range(9)]
    n += [f'residual_{s}_{u}' for u in ('px','bboxnorm') for s in ('mean','median','P90','max','front4_mean','rear4_mean','center','confidence_weighted_mean')]
    n += ['bbox_conf']+[f'kpt_conf{k}' for k in range(9)]+['kpt_conf_mean','kpt_conf_min','kpt_conf_P10','kpt_conf_P50','bbox_aspect','bbox_area_image_fraction']
    return n

def main():
    C.freeze(C.DOC/'SELECTOR_FEATURE_CONTRACT.json',dict(created_at=C.now(),features=names(),n_features=len(names()),
        families=['GEO','GEO_IMG'],image_feature='final Pose26 module forward_pre_hook; GAP each incoming feature scale then concatenate',
        candidate_scorer_shared=True,lower_is_better=True,candidate_order=list(C.HYP),tie='lexicographic candidate name',
        real_GT_used=False,feature_selection_after_real_GT=False,normalization='training-only mean/std, shared over both candidates; std floor1e-6',
        missing='only successful finite two-candidate samples used for synthetic learning; inference one-success chooses it, no-success retains production fallback',
        inference=dict(conf=.001,imgsz=640,rect=True,augment=False,half=False,real_padding=100,synthetic_already_padded=True),
        train=dict(seed=42,lr=.001,weight_decay=.0001,batch=256,max_epoch=30,patience=5,selection='VAL parity accuracy; earliest best; variant tie GEO_LINEAR'),
        split=dict(seed=20260925,requested=[4096,1024,1024],minimum=[1024,256,256],priority='renderer group disjoint then SHA; same renderer derivative group never cross split'),
        no_pointwise_remapping=True))

if __name__=='__main__':main()
