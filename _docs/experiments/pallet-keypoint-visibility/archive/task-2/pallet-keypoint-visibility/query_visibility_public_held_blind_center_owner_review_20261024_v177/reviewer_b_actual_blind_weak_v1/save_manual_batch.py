"""Serialize manually tool-viewed B observations only; never reads source/model arrays."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
V=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
states={'T':'TARGET_VISIBLE_SUPPORTED','F':'FOREIGN_VISIBLE_SUPPORTED','B':'BACKGROUND_SUPPORTED','U':'UNKNOWN','I':'UNKNOWN'}
pres={'Y':'YES','N':'NO','U':'UNKNOWN'}
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main(path):
    manual=json.loads(Path(path).read_text());inputs=[json.loads(s) for s in (V/'blind_review_evidence_v1/ALL360_BLINDED_REVIEW_INPUTS.jsonl').read_text().splitlines()]
    by={r['original_global_index']:r for r in inputs};out=[];views=[]
    for fi_s,m in manual.items():
        fi=int(fi_s);assert len(m['owners'])==len(m['presence'])==len(m['notes'])==8
        sheet=V/f'blind_review_evidence_v1/FRAME_{fi:04d}/SHEET.png';sb=bind(sheet)
        views.append({'frame_index':fi,**sb,'actual_tool_view':'tools.view_image','all8slots_actually_viewed':True})
        for slot,(oc,pc,note) in enumerate(zip(m['owners'],m['presence'],m['notes'])):
            i=fi*8+slot;r=by[i];assert r['frame_index']==fi and r['native_slot']==slot
            forced=r['forced_center_unknown_reason']
            if forced:assert oc in 'UI';cr=forced
            else:cr={'T':'CLEAR_TARGET','F':'CLEAR_FOREIGN','B':'CLEAR_BG','U':'MIXED_EDGE','I':'INSTANCE_AMBIGUOUS'}[oc]
            if not r['feature_constructed']:assert pc=='U';pr='INPUT_U'
            else:pr={'Y':'CLEAR_TARGET','N':'CLEAR_FOREIGN','U':'INSUFFICIENT_VISUAL_EVIDENCE'}[pc]
            out.append({'original_global_index':i,'query_id':r['query_id'],'frame_index':fi,'native_slot':slot,
                'center_owner':states[oc],'center_reason':cr,'crop_presence':pres[pc],'crop_reason':pr,
                'observation':note,'sheet_path':str(sheet),'sheet_sha256':sb['sha256'],
                'actually_viewed':True,'original_query_available':r['query_available'],
                'center4_valid_bits':r['center4_valid_bits'],'training_admission':False,
                'authority':'assistant weak visual observation; not human/pixel/native/coordinate gold'})
    lo=min(x['frame_index'] for x in views);hi=max(x['frame_index'] for x in views)
    op=D/f'BATCH_{lo:04d}_{hi:04d}_OBSERVATIONS.jsonl'
    with op.open('x') as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    vp=D/f'BATCH_{lo:04d}_{hi:04d}_ACTUAL_VIEWS.json'
    with vp.open('x') as f:json.dump({'UTC':datetime.now(timezone.utc).isoformat(),'actually_viewed_sheets':views},f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'frames_recorded':len(views),'rows_recorded':len(out),'observations':bind(op),'views':bind(vp)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manual');main(p.parse_args().manual)
