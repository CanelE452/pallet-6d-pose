"""Metadata-only fixed eight by ten selection. Does not stat any PNG/NPZ path."""
from pathlib import Path
import hashlib,json,os,stat
H=Path(__file__).parent;V=H.parent;P=V.parent
S=V/'ROOT_PROSPECTIVE_RAW96_WEAK_WITNESS_CONTROL_PILOT_SCOPE.json'

def read(path):
    p=Path(path)
    assert p.is_absolute() and P in p.parents and '..'not in p.parts and 'v112'not in str(p).casefold()
    assert p.suffix in {'.json','.jsonl'},'No payload or header access'
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for x in p.parts[1:-1]:
            n=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(n).st_mode)
    with os.fdopen(n,'rb')as f:return f.read()

def get(binding):
    data=read(binding['path']);assert hashlib.sha256(data).hexdigest()==binding['sha256'];return json.loads(data)
def digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def inert(b,kind):
    # Lexical validation only: no filesystem operation, including existence check.
    p=Path(b['path']);assert p.is_absolute() and '..'not in p.parts and 'v112'not in str(p).casefold()
    assert p.suffix==('.npz'if kind=='COMPACT_RGBA'else'.png') and len(b['sha256'])==64
    out=dict(path=b['path'],sha256=b['sha256'],kind=kind)
    if 'bytes'in b:out['size_bytes']=b['bytes']
    return out
def put(name,obj):
    with (H/name).open('x',encoding='utf-8')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    scope_bytes=read(S);assert hashlib.sha256(scope_bytes).hexdigest()=='a61ee52c41e4ad8e8e21d514ec8801cdaf6d32863438cc8c084cc4cbf37cf342'
    scope=json.loads(scope_bytes);joined=get(scope['witness_join']);qa=get(scope['root_actual_witness_visual_QA']);ready=get(scope['donor_ready'])
    assert joined['completed'] and joined['PASS'] and len(joined['rows'])==8
    assert qa['completed'] and qa['PASS'] and qa['actual_toolview_fixed_cards']==8 and qa['original_clean_and_valid_copied_PNG_bytes_exact']
    assert ready['completed'] and ready['PASS'] and ready['donor_count']==59
    display=get(qa['display_manifest']);assert display['completed'] and display['PASS'] and len(display['cards'])==8
    donorbytes=read(scope['donor_metadata']['path']);assert hashlib.sha256(donorbytes).hexdigest()==scope['donor_metadata']['sha256']
    all_donors=[json.loads(line)for line in donorbytes.splitlines()if line.strip()]
    assert len(all_donors)==59 and all(r['role']=='YCB_V_TRAIN_FOREIGN_DONOR_NOT_PALLET_TARGET'for r in all_donors)
    objects=sorted({r['object_id']for r in all_donors});assert len(objects)==10
    selected=[]
    for object_id in objects:
        donor=min((r for r in all_donors if r['object_id']==object_id),key=lambda r:digest('V207_RAW96_DONOR_V1|'+r['donor_id']))
        selected.append(dict(donor_id=donor['donor_id'],object_id=object_id,scene_id=donor['scene_id'],frame_id=donor['frame_id'],instance_id=donor['instance_id'],
            selection_hash=digest('V207_RAW96_DONOR_V1|'+donor['donor_id']),role=donor['role'],compact_RGBA=inert(donor['compact_RGBA_cutout'],'COMPACT_RGBA'),
            recorded_array_proof={k:donor['existing_array_proof'][k]for k in ('rgb','alpha')},source_asset_stat_hash_header_decode=False))
    cases=[]
    for i,(case,card,q)in enumerate(zip(joined['rows'],display['cards'],qa['rows'])):
        assert case['row_id']==card['row_id']==q['row_id'] and card['position']==i
        assert q['weak_corresponding_local_visible_feature_complex_supported']is True and q['observer_extents_preserved_not_adjusted']is True
        assert case['corrected_keypoint_or_new_E_label']is False and case['training_admission']is False
        cases.append(dict(position=i,**{k:case[k]for k in ('row_id','frame_key','native_slot','original_query_xy','original_L','original_query_available','state_A','state_C','ROI_A','ROI_C')},
            CLEAN96=inert(card['images']['clean96'],'CLEAN96'),VALID96=inert(card['images']['valid96'],'VALID96'),
            authority='INDEPENDENT_ASSISTANT_WEAK_VISIBLE_FEATURE_RECTANGLES_NOT_NATIVE_GOLD',fit_admission=False,new_E_label=False))
    assignments=[]
    for case in cases:
        for j,donor in enumerate(selected):
            assignments.append(dict(attempt_id=f"CASE{case['position']:02d}_OBJ{donor['object_id']:02d}",case_position=case['position'],row_id=case['row_id'],
                donor_id=donor['donor_id'],object_id=donor['object_id'],assignment_hash=digest(f"V207_RAW96_ASSIGN_V1|{case['row_id']}|{donor['donor_id']}")))
    files=[b for c in cases for b in (c['CLEAN96'],c['VALID96'])]+[d['compact_RGBA']for d in selected]
    assert len(files)==26 and len({b['path']for b in files})==26 and len(assignments)==80
    selection=dict(completed=True,PASS=True,selection_before_payload_access=True,scope=dict(path=str(S),sha256=hashlib.sha256(scope_bytes).hexdigest()),
        provenance={k:scope[k]for k in ('witness_join','root_actual_witness_visual_QA','donor_ready','donor_metadata')},display_manifest=qa['display_manifest'],
        cases=cases,donors=selected,assignments=assignments,attempt_count=80,
        fixed_preview_rule='Per original case, minimum V207_RAW96_ASSIGN_V1 SHA among constructed attempts; else factual plus failure. No score/image preference.',
        post_sampler_only=True,new_E_or_fit_admission=False,actual_asset_stat_hash_header_decode=0)
    put('FIXED8_X_TRAIN10_DONOR80_SELECTION.json',selection)
    put('EXACT26_PAYLOAD_WHITELIST.json',dict(completed=True,PASS=True,files=files,CLEAN96_PNG=8,VALID96_PNG=8,compact_RGBA_NPZ=10,
        source_binary_access_before_root_entry=0,source_RGB_GT_mask_nativealpha_or_array_binding_followed=False))
    print(json.dumps(dict(completed=True,PASS=True,cases=8,donors=10,attempts=80,whitelist=26,selection_sha256=hashlib.sha256((H/'FIXED8_X_TRAIN10_DONOR80_SELECTION.json').read_bytes()).hexdigest())))

if __name__=='__main__':main()
