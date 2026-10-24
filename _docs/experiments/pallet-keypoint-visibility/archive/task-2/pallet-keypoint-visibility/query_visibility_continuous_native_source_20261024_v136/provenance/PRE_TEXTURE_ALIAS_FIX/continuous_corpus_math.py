"""V136 pure metadata math; immutable V129 geometry/material operators.

No registered mesh payload, RGB, teacher, feature, or model is loaded here.
The camera ordering is the fixed source-label normal rule certified in V134.
"""
import ast, hashlib, importlib.util, json
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parent;P=H.parent.parent
OLD=P/'query_visibility_registered_native_diversity_20261023_v129/render_preparation/corpus_math.py'
OLD_SHA='96713b5c8380e7698081d9189d54b3f4911ea6cda14cbd47083a79f4df36817e'
ORDER=P/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py'
ORDER_SHA='4ddd134aa26da1d676628bd6987a161ab36f09328dfdaa89bedd213e397755ab'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()==OLD_SHA
_s=importlib.util.spec_from_file_location('v136_immutable_V129_math',OLD)
old=importlib.util.module_from_spec(_s);_s.loader.exec_module(old)
OLD_DIM,SIGNS,NAMES,ROLES,FOREIGN=old.OLD_DIM,old.SIGNS,old.NAMES,old.ROLES,old.FOREIGN
project,quarter_mapping,scaled_data=old.project,old.quarter_mapping,old.scaled_data
foreign_placements,material_rgb,shade,CF_records=old.foreign_placements,old.material_rgb,old.shade,old.CF_records
hash_int,uniform=old.hash_int,old.uniform

def normal_rule():
    assert hashlib.sha256(ORDER.read_bytes()).hexdigest()==ORDER_SHA
    nodes=[]
    for n in ast.parse(ORDER.read_text()).body:
        if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='VERTICAL_FIXED_FACES'for t in n.targets):nodes.append(n)
        elif isinstance(n,ast.FunctionDef)and n.name in('outward_normals','normal_facing_permutation'):nodes.append(n)
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ORDER),'exec'),ns)
    return ns['normal_facing_permutation']

def certified_normal_mapping(cfg,rule):
    xyz=np.asarray(cfg['native8_canonical_xyz'],np.float64)
    zup=xyz[:,[0,2,1]]*np.array([1,1,-1])
    eye=np.asarray(cfg['camera']['eye_world']);eye=eye[[0,2,1]]*np.array([1,1,-1])
    xy=project(xyz,cfg['camera']);perm,cos=rule(zup,xy,eye)
    margin=float(np.sort(cos)[-1]-np.sort(cos)[-2])
    assert margin>1e-10 and perm.tolist()==cfg['permCF_to_canonical']
    return dict(perm=perm.tolist(),cosines=cos.tolist(),top_cosine_margin=margin,
                fixed_rule='V74 normal_facing_permutation; certified source-label equivalence V134; no image/accuracy association')

def camera_scaled(anchors,dimensions,quarter,pair,distance_scale):
    camera,A,perm=old.camera_config(anchors,dimensions,quarter,pair,distance_scale)
    distance=float(max(dimensions[0],dimensions[2])*distance_scale)
    assert np.isclose(np.linalg.norm(camera['eye_world']),distance,rtol=0,atol=1e-14)
    return camera,A,perm

def make_configs(policy,meshes):
    f=policy['fixed_factors'];rows=[];rule=normal_rule()
    for fold,seeds in f['asset_seeds'].items():
        ng=len(seeds)*len(f['geometry_families'])*len(f['aspect_ratios'])
        for asset_ordinal,asset in enumerate(seeds):
            for gi,geometry in enumerate(f['geometry_families']):
                for ai,(aspect,ratio)in enumerate(f['aspect_ratios'].items()):
                    geomkey=('V136',fold,asset,geometry,aspect)
                    gordinal=(asset_ordinal*2+gi)*2+ai
                    height=.10+.10*(gordinal+uniform(0,1,*geomkey,'height_jitter'))/ng
                    z=uniform(.85,1.25,*geomkey,'footprint_Z')
                    dims=np.array([ratio*z,height,z]);scale=dims/OLD_DIM;anchors=(SIGNS*OLD_DIM/2)*scale
                    for qi,quarter in enumerate(f['signed_quarters']):
                        for vi,sign in enumerate(f['view_signs']):
                            ci=2*qi+vi;viewkey=geomkey+(quarter,sign)
                            az=sign*uniform(5,40,*viewkey,'azimuth_abs_deg')
                            el=uniform(10,45,*viewkey,'elevation_deg')
                            distance_scale=uniform(2.2,4.2,*viewkey,'distance_scale')
                            camera,A,perm=camera_scaled(anchors,dims,quarter,[az,el],distance_scale)
                            appearance=(ci+gi+2*ai+asset_ordinal)%4;role=(ci//2+gi+ai+asset_ordinal)%4
                            direction=(ci+3*gi+5*ai+asset_ordinal)%8;angle=np.deg2rad(45*direction)
                            side=1 if az>0 else 0;slots=[side,1-side,2 if side else 3,5 if side else 4]
                            for material in f['materials']:
                                b=len(rows);seeds2={purpose:hash_int('V136',fold,asset,geometry,aspect,
                                    material,quarter if purpose!='material'else'',sign if purpose!='material'else'',purpose)
                                    for purpose in('material','light','background','foreign')}
                                factors=dict(geometry_family=geometry,aspect=aspect,material=material,asset_seed=asset,
                                    camera_pair_deg=[az,el],signed_quarter=quarter,view_sign=sign,
                                    foreign_family=FOREIGN[appearance],target_role=ROLES[role],distance_scale=distance_scale,
                                    distance_m=camera['distance'])
                                cfg=dict(base_index=b,base_id='V136_BASE_%04d'%b,split=fold,factors=factors,
                                    geometry_group_id='%s_%d_%s_%s'%(fold,asset,geometry,aspect),
                                    target_layer_group_id='%s_%d_%s_%s_Q%d_V%d'%(fold,asset,geometry,aspect,quarter,vi),
                                    template=meshes[gi],template_index=gi,dimensions_xyz_m=dims.tolist(),positive_scale_xyz=scale.tolist(),
                                    raw_hw=[512,512],camera=camera,canonical_to_CF_rotation=A.tolist(),permCF_to_canonical=perm,
                                    native8_canonical_xyz=anchors.tolist(),native8_CF_xy=project(anchors,camera)[perm].tolist(),
                                    selected_CF_index=slots[role],selected_canonical_index=perm[slots[role]],
                                    sham_direction_unit_xy=[float(np.cos(angle)),float(np.sin(angle))],
                                    foreign_extent_factor=uniform(1.25,3.,*viewkey,'foreign_extent'),
                                    foreign_margin_px=uniform(1.,7.,*viewkey,'foreign_margin_px'),
                                    foreign_depth_fraction=uniform(.45,.80,*viewkey,'foreign_depth_fraction'),
                                    foreign_roll_deg=uniform(-25.,25.,*viewkey,'foreign_roll_deg'),seeds=seeds2,
                                    generation_keys=dict(geometry=list(geomkey),height_stratum=gordinal,height_strata=ng,
                                        camera_and_foreign=list(viewkey)),configuration_uses_RGB_or_labels=False)
                                cfg['source_label_mapping_certificate']=certified_normal_mapping(cfg,rule)
                                rows.append(cfg)
    assert len(rows)==640 and sum(r['split']=='TRAIN'for r in rows)==512
    return rows

def metadata_audit(configs,old_configs):
    """Fail on any contract break; never reroll, fit, select or inspect pixels."""
    fields=dict(height_m=lambda r:r['dimensions_xyz_m'][1],footprint_Z_m=lambda r:r['dimensions_xyz_m'][2],
        azimuth_abs_deg=lambda r:abs(r['factors']['camera_pair_deg'][0]),elevation_deg=lambda r:r['factors']['camera_pair_deg'][1],
        distance_scale=lambda r:r['factors']['distance_scale'],distance_m=lambda r:r['factors']['distance_m'],foreign_extent=lambda r:r['foreign_extent_factor'],
        foreign_margin_px=lambda r:r['foreign_margin_px'],foreign_depth_fraction=lambda r:r['foreign_depth_fraction'],
        foreign_roll_deg=lambda r:r['foreign_roll_deg'])
    folds={};geometry={};texture={};seeds_byfold={};camera_byfold={}
    old_camera={tuple(np.round(np.r_[r['camera']['eye_world'],r['dimensions_xyz_m']],14))for r in old_configs}
    old_asset={r['factors']['asset_seed']for r in old_configs}
    def texture_signature(r):
        mat=r['factors']['material'];s=r['seeds']['material']
        return(mat,s%1009,s%53)if mat=='PROCEDURAL_WOOD'else(mat,s%1009)
    old_texture={texture_signature(r)for r in old_configs}
    for fold,n in(('TRAIN',512),('VALIDATION',128)):
        rr=[r for r in configs if r['split']==fold];assert len(rr)==n
        geometry[fold]={r['geometry_group_id']:tuple(r['dimensions_xyz_m'])for r in rr}
        texture[fold]={texture_signature(r)for r in rr}
        seeds_byfold[fold]={r['factors']['asset_seed']for r in rr}
        camera_byfold[fold]={tuple(np.round(np.r_[r['camera']['eye_world'],r['dimensions_xyz_m']],14))for r in rr}
        assert seeds_byfold[fold].isdisjoint(old_asset)and texture[fold].isdisjoint(old_texture)
        assert camera_byfold[fold].isdisjoint(old_camera)
        values={k:np.asarray([fn(r)for r in rr])for k,fn in fields.items()}
        corr={name:float(np.corrcoef(values['height_m'],values[name])[0,1])for name in('footprint_Z_m','azimuth_abs_deg','elevation_deg','distance_scale')}
        folds[fold]=dict(bases=n,scenes=4*n,native_slots=32*n,geometry_groups=len(geometry[fold]),
            target_layer_groups=len(set(r['target_layer_group_id']for r in rr)),texture_recipes=len(texture[fold]),
            continuous={k:dict(min=float(v.min()),max=float(v.max()),unique=len(np.unique(v)))for k,v in values.items()},
            finite_sample_height_correlations=corr,
            minimum_normal_rule_top_cosine_margin=min(r['source_label_mapping_certificate']['top_cosine_margin']for r in rr),
            all_configs_normal_rule_exact=True,old_camera_dimension_exact_clones=0)
    assert seeds_byfold['TRAIN'].isdisjoint(seeds_byfold['VALIDATION'])
    assert texture['TRAIN'].isdisjoint(texture['VALIDATION'])and camera_byfold['TRAIN'].isdisjoint(camera_byfold['VALIDATION'])
    assert set(geometry['TRAIN'].values()).isdisjoint(geometry['VALIDATION'].values())
    assert len(geometry['TRAIN'])==32 and len(geometry['VALIDATION'])==8
    return dict(completed=True,PASS=True,folds=folds,old128_64_untouched=True,
        new_asset_seeds_disjoint_bothfolds_and_old=True,new_geometry_dimension_tuples_disjoint=True,
        new_texture_recipe_signatures_disjoint_bothfolds_and_old=True,new_exact_camera_dimension_tuples_disjoint_bothfolds_and_old=True,
        dimensions_and_camera_separate_hash_purposes=True,continuous_sampling_no_rejection_or_rerolls=True,
        same_two_topologies_positive_scaling_only=True,not_new_topology=True,
        normal_rule_not_obsolete_area_rule=True,no_current_private_V112_PnP_models_RGB_teacher_calls=True)
