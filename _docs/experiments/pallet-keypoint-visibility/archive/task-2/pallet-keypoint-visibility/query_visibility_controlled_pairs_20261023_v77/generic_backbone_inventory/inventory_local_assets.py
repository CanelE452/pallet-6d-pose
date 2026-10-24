"""Read-only local asset/code inventory; no model, data, RGB or network imports."""
from pathlib import Path
import hashlib,json,subprocess

OUT=Path(__file__).resolve().parent
P=OUT.parents[1]
REPO=Path('/home/minjae/Documents/github/pallet-pose')
SAM2=P/'foundation_runtime_sam21_20261023'
V63=P/'query_visibility_public_real_20261022_v63'
DINO_ASSETS=REPO/'outputs/pallet_type_selftrain_v1/selftrain_recovery_v1/dino_localization_assets'
MOBILE=V63/'model/deployment_assets/MobileSAM'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()
def binding(p):return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    assert not p.exists();p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def weight(p,expected,expected_size=None,fresh=True):
    d={'path':str(p),'exists':p.is_file(),'sha256_expected':expected,'hash_scope':'Fresh raw-byteSHA256; no torchload'if fresh else'Previously frozenSHA256; onlysize/stat recheckedtoday'}
    if p.is_file():
        d['bytes']=p.stat().st_size;d['size_matches_previous']=None if expected_size is None else d['bytes']==expected_size
        if fresh:d['sha256_actual']=sha(p);d['matches_previous_sha256']=d['sha256_actual']==expected
    return d
def head(p):return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()

def main():
    dino_receipt=REPO/'_docs/experiments/pallet_type_selftrain_v1/selftrain_recovery_v1/dino_localization/BACKBONE.json'
    db=read(dino_receipt);sam2seal=SAM2/'RUNTIME_SEAL.json';s2=read(sam2seal)
    mobile_receipt=V63/'contract/MOBILESAM_STANDALONE_AUDIT.json';mb=read(mobile_receipt)
    dw=DINO_ASSETS/'torch_hub/checkpoints/dinov2_vits14_pretrain.pth'
    mw=MOBILE/'weights/mobile_sam.pt'
    rw=REPO/'data/pallet/results/pallet_resnet18_refiner_20261001_v1/pretrained/resnet18-f37072fd.pth'
    sw=Path(s2['checkpoint']['path'])
    dino_weight=weight(dw,db['checkpoint']['sha256'],db['checkpoint']['bytes'])
    mobile_weight=weight(mw,'6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f',40728226)
    resnet_weight=weight(rw,'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec')
    sam2_weight=weight(sw,s2['checkpoint']['sha256'],s2['checkpoint']['bytes'],fresh=False)
    dino_repo=DINO_ASSETS/'dinov2';sam2_repo=SAM2/'sam2_repo'
    assert head(dino_repo)==db['commit']and head(sam2_repo)==s2['repo_revision']
    assert all(w['exists']and w['matches_previous_sha256']for w in [dino_weight,mobile_weight,resnet_weight])
    assert sam2_weight['exists']and sam2_weight['size_matches_previous']
    code_paths={
        'DINO_asset_loader':REPO/'scripts/research/pallet_type_selftrain_v1/dino_localization_assets.py',
        'DINO_direct_local_feature_loader':REPO/'scripts/research/pallet_pose_dino_input_audit_20261001_v1/freeze_train.py',
        'DINO_original_feature_extractor':REPO/'scripts/research/pallet_type_selftrain_v1/dino_localization.py',
        'DINO_mid_feature_option':REPO/'scripts/research/pallet_type_selftrain_v1/dino_mid_feature.py',
        'DINO_official_backbone':dino_repo/'dinov2/hub/backbones.py',
        'DINO_official_ViT':dino_repo/'dinov2/models/vision_transformer.py',
        'DINO_license':dino_repo/'LICENSE',
        'SAM2_predictor':sam2_repo/'sam2/sam2_image_predictor.py',
        'SAM2_transforms':sam2_repo/'sam2/utils/transforms.py',
        'SAM2_base':sam2_repo/'sam2/modeling/sam2_base.py',
        'SAM2_decoder':sam2_repo/'sam2/modeling/sam/mask_decoder.py',
        'SAM2_config':sam2_repo/'sam2/configs/sam2.1/sam2.1_hiera_b+.yaml',
        'SAM2_frozen_source_runner':P/'query_visibility_sam21_frozen_20261023_v68/extract_source_masks.py',
        'MobileSAM_builder':MOBILE/'mobile_sam/build_sam.py',
        'MobileSAM_predictor':MOBILE/'mobile_sam/predictor.py',
        'MobileSAM_preprocessing':MOBILE/'mobile_sam/modeling/sam.py',
        'MobileSAM_resize':MOBILE/'mobile_sam/utils/transforms.py',
        'MobileSAM_standalone_source_runner':V63/'sam/standalone_mobile_sam.py',
        'ResNet_ImageNet_file_binding':REPO/'scripts/research/pallet_resnet18_refiner_20261001_v1/base_common.py',
        'ResNet_feature_taps_and_local_loader':REPO/'scripts/research/pallet_resnet18_refiner_20261001_v1/model.py'}
    codes={k:binding(v)for k,v in code_paths.items()}
    entries=[{
        'model':'DINOv2 ViT-S/14 official generic checkpoint','available':True,'weight':dino_weight,
        'used_in_V63_to_V76':False,'historical_local_use':'Earlier pallet-pose research source code; official backbone asset remains separate from task-specific heads/caches.',
        'primary_source':{'repository':db['repository'],'pinned_commit':db['commit'],'actual_git_HEAD_matches':True,'receipt':binding(dino_receipt),'license_code_file':codes['DINO_license']},
        'local_code':['DINO_asset_loader','DINO_direct_local_feature_loader','DINO_original_feature_extractor','DINO_mid_feature_option','DINO_official_backbone','DINO_official_ViT'],
        'no_download_loading':'Use dinov2_vits14(pretrained=False), then local torch.load(weights_only=True) and strictload, as existing freeze_train.py. Historical A.load(pretrained=True) depends on torchhubcache, so do not blindly reuse that path.',
        'preprocess':'Existing code: RGBfloat/255, bilinear align_corners=False resizeHW392x294 orwide784x588, ImageNetmean[.485,.456,.406]/std[.229,.224,.225]. Existing affine crops are from R0predictedbox, not q-centeredV76L crops.',
        'spatial_features':{'API':'model.forward_features(x)[x_norm_patchtokens]','channels':384,'patch_size':14,'shape_392x294':'384x28x21','shape_784x588':'384x56x42','storage_historical':'FP16 patchtokens; fp32 model inference','CLS_and_register_tokens':'Excluded by x_norm_patchtokens API','mid_layer_optional':'Existing separate fixedblock5-of12 code path; availability only, no layerselection or run here.'},
        'runtime_verification_today':'Not loaded/imported/executed. Only localfilehash+officialrepoHEAD+source inspected.',
        'compatibility':'Not drop-in for65channel specialistP3. Need new frozenfeature producer with exact source-only affine/inverseq mapping andvalidity; newinputprojection/headchannels before fit. OldDINOtaskheads, localization/PnP runners and oldfeaturecaches are not V76/V77 inputcompatible.'
    },{
        'model':'SAM2.1 Hiera base-plus official generic segmenter/imageencoder','available':True,'weight':sam2_weight,'used_in_V63_to_V76':True,'local_use':'V68 fixedbox-only source mask baseline; no prior imageencoder-latentfeature cache established forV76.',
        'primary_source':{'repository':'https://github.com/facebookresearch/sam2','pinned_commit':s2['repo_revision'],'actual_git_HEAD_matches':True,'runtime_seal':binding(sam2seal),'license':s2['license']},
        'runtime':{'python':s2['python'],'recorded_versions':s2['pinned_versions'],'prior_blankCPU_andCUDA_smoke':'Bound in RUNTIME_SEAL; not rerun today'},
        'local_code':['SAM2_predictor','SAM2_transforms','SAM2_base','SAM2_decoder','SAM2_config','SAM2_frozen_source_runner'],
        'no_download_loading':'Existing isolatedruntime build_sam2(localconfig,localcheckpoint,apply_postprocessing=False); avoid from_pretrained/HFfallback.',
        'preprocess':'SAM2Transforms ToTensor→Resize1024x1024 (square, not SAM1longest-sidepad)→ImageNet mean/std. Nativeq normalized byoriginalW/H thenresolution. ExistingV68 fixedfp32weights+bf16autocast,batch1,nocompile,hole/sprinkle0.',
        'spatial_features':{'API':'predictor.set_image(RGB); predictor.get_image_embedding()','image_embed':'1x256x64x64 from inspectedconfig/predictor, not runtime-tested today','high_res':'predictor._features[high_res_feats] exposes256x256 and128x128 spatialgrids; decoderprojectedchannels32/64 underuse_high_res_features_in_sam','backbone_only':'model.forward_image plus_prepare_backbone_features; use exactsame config/no_memembed behavior if matching predictor','prompt_scope':'Imageencoder latentfeatures computed beforeprompt/maskchoice; maskdecoder additionallytakes predictedbox.'},
        'runtime_verification_today':'Size/stat+priorboundSHAonly for323606802Bcheckpoint; no freshlargeweightbodyhash, load, orforward.',
        'compatibility':'V68saved nativechosenlogits are maskoutputs, not encoderfeatures. Newrawimagefeaturecache required forV77 changedRGB.256/32/64 channels andSAM2coordinatewarp differ fromR0P3, so headandspatialproducer need separately frozen mapping.'
    },{
        'model':'MobileSAM vit_t generic imageencoder and promptable mask decoder','available':True,'weight':mobile_weight,'used_in_V63_to_V76':True,'local_use':'V63standalonebox-only sourcepipeline and laterSAM-feature recipes.',
        'primary_source':{'repository':mb['official_repository'],'pinned_commit_in_previous_audit':mb['pinned_commit'],'audit':binding(mobile_receipt),'license':mb['license'],'deployment_copy':str(MOBILE)},
        'local_code':['MobileSAM_builder','MobileSAM_predictor','MobileSAM_preprocessing','MobileSAM_resize','MobileSAM_standalone_source_runner'],
        'no_download_loading':'sam_model_registry[vit_t](checkpoint=None), strict localstate load; SamPredictor. Standalone directsource runner avoids oldCAL/PnP wrappers.',
        'preprocess':'RGBuint8→ResizeLongestSide1024→float normalize by[123.675,116.28,103.53]/[58.395,57.12,57.375]→right/bottomzeropad1024square. Preserve inverse resize andpadding before nativeq sampling.',
        'spatial_features':{'API':'predictor.set_image(RGB); predictor.get_image_embedding()','embedding':'1x256x64x64 perlocalbuilder/predictor','existing_features':'ExistingV63+later rawSAM native9/masklogitmaps are outputs, not these256 latentchannels.'},
        'runtime_verification_today':'Fresh40728226BweightSHA+sourcefiles only. Previous sourcepipeline used pallet-poseTorch2.1; runtime not imported/tested today.',
        'compatibility':'Generic latentfeatures available withoutCALmodel orpose solve. A visiblemasknegative is not semantic E/S; boxprompt doesnotguarantee targetownership or cornerassociation. Newfeatureproducer needed for latentuse.'
    },{
        'model':'torchvision ResNet18 ImageNet pretrained body','available':True,'weight':resnet_weight,'used_in_V63_to_V76':False,'historical_local_use':'Earlier independentResNet18 estimator/refiner research; inventory points only to originalImageNetfile, not fittedpalletcheckpoints.',
        'primary_source':{'weight_binding_in_code':codes['ResNet_ImageNet_file_binding'],'feature_adapter':codes['ResNet_feature_taps_and_local_loader'],'source_type':'Localcode explicitly binds torchvisionImageNetfile SHA; separate checkpointlicense metadata not independently researched today.'},
        'local_code':['ResNet_ImageNet_file_binding','ResNet_feature_taps_and_local_loader'],
        'no_download_loading':'torchvision.models.resnet18(weights=None); suppliedlocalstate only. Existing SimpleBaselineResNet18.load_imagenet_state checks allbodytensors, handles legacy BNcounters, excludes optionalfc pair.',
        'preprocess':'Existing estimator uses RGBImageNetconstants and384x512 aspectfit/centeredletterbox. This is not automaticallythe correct V77 featurecrop; any futureproducer must freeze nativeq inversemap.',
        'spatial_features':{'API':'Existing eval().features_only(images) returnslayer2,layer3','channels':[128,256],'strides':[8,16],'input384x512_shapes':['128x48x64','256x24x32'],'no_heatmap_needed':'features_only runs prefix only, not layer4/deconv/keypointhead'},
        'runtime_verification_today':'FreshsmallweightSHAandcode only; no state_dict load/modelrun.',
        'compatibility':'Stride8 resembles P3spacing but channelcount, receptivefield andletterbox differ; cannot reuse specializedP3cache or41kheadweights unchanged. Use only the pretrainedbodyfile, not old estimator trainedstates.'
    }]
    roots=[Path('/home/minjae/.cache/huggingface/hub'),Path('/home/minjae/.cache/torch/hub'),Path('/home/minjae/.cache/clip'),Path('/home/minjae/.cache/open_clip'),Path('/home/minjae/Documents/models'),Path('/home/minjae/Documents/weights'),REPO,P,Path('/home/minjae/Documents/pallet-research')]
    hf=roots[0];hfdirs=sorted(x.name for x in hf.iterdir()if x.is_dir()and x.name.startswith('models--'))
    search={'roots':[{'path':str(x),'exists':x.exists()}for x in roots],'method':'rg --files --hidden forpt/pth/safetensors/bin/ckpt names +priorcode/provenance references; no PNG/NPZfeaturebody/results access',
        'HF_model_directory_names_metadata_only':hfdirs,
        'not_found_in_searched_roots':['DINOv3 pretrainedvisionweights','CLIP/OpenCLIP/SigLIP imageencoderweights','originalSAMvit_b/l/hweights'],
        'not_global_absence_claim':True,'SAM_vit_b_l_h_note':'BuilderfunctionspresentinMobileSAMvendorregistry do notmean correspondingpretrainedweights are on disk.',
        'not_image_backbones':'Cached BGE text and time-series model directories were seen byname only; not reclassified as imageassets.',
        'specialist_baseline':'Existing YOLO26P3code/weights are specialistcurrentbaseline, not an independent genericrepresentation. No checkpoint is chosen here.'}
    report={'schema':'V77_LOCAL_GENERIC_IMAGE_BACKBONE_READONLY_INVENTORY','completed':True,'model_ranking_or_selection':False,
        'source_scope':'Availabilityandcode/provenance only, not new comparison. V77 single frozen data intervention remains unchanged.',
        'assets':entries,'code_bindings':codes,'search':search,
        'future_contract_requirements':['Keep originalq8/L/box/nativechannel/instancequeryset andsamebefore/afterinputRGB; GTref/masks/teacherlabels neverforwarded.',
            'New genericfeaturecache must cover changedafterRGB; no reusing oldfeatures for newcomposites.',
            'Freeze featurelayer, imagewarp, normalization, padding, spatialcenter/readout anddtype before sourcefit; use same sourcegroups andsource-onlyselector.',
            'Output latentrepresentation is not querypixel/semanticvisibility GT. Virtualcuboidreference neednotexactlylieonmesh; mask0 mustnotbecomeE/S automatically.',
            'Maskqualityargmax or prompt cannot be selected fromGT/currentoutcomes. SourceGTonly supervises declaredlocal-support loss.'],
        'actions':{'new_download':0,'environment_changes':0,'model_import_load_or_run':0,'training_or_PnP':0,'source_or_current_RGB_PNG_reads':0,'current_or_private_metric_result_files_read':0,'featurecache_NPZbody_reads':0,'hugefilecopies':0,
            'weight_rawbytehashes':'DINO88MB/MobileSAM40MB/ResNet46MB only; SAM2large323MBpreviousSHA+stat'},
        'limitations':'Assets stat/hash/codeverified; runtimecompatibility/newfeaturemapping not executed. PreviousDINO/ResNet experiment descriptions are not reused as V77 training evidence or currentperformance.'}
    save(OUT/'GENERIC_BACKBONE_INVENTORY.json',report)
    save(OUT/'SEAL.json',{'completed':True,'inventory':binding(OUT/'GENERIC_BACKBONE_INVENTORY.json'),'code':binding(Path(__file__)),'scope':'Read-only localmetadata/code/smallweightsha, no model/data execution'})
    print(json.dumps({'inventory':binding(OUT/'GENERIC_BACKBONE_INVENTORY.json'),'seal':binding(OUT/'SEAL.json'),'weights':[{ 'model':x['model'],'weight':x['weight']}for x in entries]},indent=2))
if __name__=='__main__':main()
