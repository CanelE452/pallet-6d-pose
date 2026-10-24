from pathlib import Path
import json,hashlib,datetime
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');R=P/'query_visibility_source_cue_corpus_20261023_v85';D=R/'annotator_b/batch03'
def b(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
idx=json.loads((D/'UNSEALED_BEFORE_METADATA_INDEX.json').read_text());out=[]
for a in idx:
 pd=Path(a['pair_dir']);ann=pd/'BEFORE_CUES.json';x=json.loads(ann.read_text());assert len(x['slots'])==6
 r={'completed':True,'phase':'BEFORE_ONLY','batch':'03','pair_id':a['pair_id'],'annotation':b(ann),'own_readerpolicy':b(D/'READER_POLICY_BEFORE_BATCH_RGB.json'),'root_policy':b(R/'SOURCE_CUE_CORPUS_SCOPE_BEFORE_NEW_RGB.json'),'format':b(R/'ANNOTATION_RECORD_FORMAT_BEFORE_LABELS.json'),'source_before_RGB':x['source_before_RGB'],'visual_receipt':b(pd/'BEFORE_VISUAL_RECEIPT.json'),'before_contact_actually_viewed':True,'fullscene_and_native_detail_actually_viewed':True,'after_RGB_decoded_or_viewed_before_this_seal':False,'native8UNKNOWN':True,'coordinates_and_uncertainty_frozen_before_after':True,'training_admitted':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code':b(Path(__file__))}
 p=pd/'BEFORE_SEAL.json';assert not p.exists();p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');out.append({'pair_id':a['pair_id'],'pair_dir':str(pd),'before_annotation':b(ann),'before_seal':b(p)})
p=D/'ALL5_BEFORE_SEALED_INDEX.json';assert not p.exists();p.write_text(json.dumps({'completed':True,'all5_before_sealed_before_any_batch_afterRGB':True,'pairs':out},indent=2)+'\n');print(b(p))
rows=json.loads((R/'ANNOTATOR_B_BATCH03_BEFORE_NEW_RGB.json').read_text())['rows'];entry=[]
for a,row in zip(out,rows):
 assert a['pair_id']==row['pair_id'];assert b(Path(a['before_annotation']['path']))==a['before_annotation'] and b(Path(a['before_seal']['path']))==a['before_seal'];assert b(Path(row['after_RGB']['path']))['sha256']==row['after_RGB']['sha256'];entry.append({'pair_id':a['pair_id'],'before_seal':a['before_seal'],'source_after_RGB':row['after_RGB']})
e=D/'AFTER_READ_ENTRY_AFTER_ALL5_BEFORE_SEALS.json';assert not e.exists();e.write_text(json.dumps({'completed':True,'global_rootpolicy_authorization':b(R/'SOURCE_CUE_CORPUS_SCOPE_BEFORE_NEW_RGB.json'),'all5before_index':b(p),'pairs':entry,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_after_decode_starts_after_this_entry':True},indent=2)+'\n');print(b(e))
