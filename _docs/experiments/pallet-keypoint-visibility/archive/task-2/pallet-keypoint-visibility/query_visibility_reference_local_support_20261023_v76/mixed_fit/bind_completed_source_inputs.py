"""Bind completed source caches; no training choices or GT values read here."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent;V=ROOT.parent

def b(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return dict(path=str(p),sha256=h.hexdigest())

def pack(name,folder,seal):return dict(source=name,manifest=b(folder/'PACK_MANIFEST.json'),seal=b(folder/seal),lineage=b(folder/'INDEX_LINEAGE_INPUT_ONLY.jsonl'))
qa=json.loads((V/'public_input_pack/INPUT_PACK_QA.json').read_text());assert qa['PASS']
se=json.loads((V/'public_input_pack/PACK_SEAL.json').read_text());assert se['PASS'] and se['instances']==4200
p=dict(schema='V76_COMPLETED_MIXED_SOURCE_INPUT_BINDINGS',choices='All learning/model/threshold/admission choices remain those of immutable MIXED_FIT_PROTOCOL.json; this file binds completed input artifacts only.',packs=[pack('SYNTHETIC',V/'synth1003_input_pack','FINAL_INPUT_PACK_SEAL.json'),pack('YCB_PUBLIC_REAL',V/'public_input_pack','PACK_SEAL.json')],additional_bindings=[b(V/x)for x in ['public_query_emulation/SEAL.json','public_input_preparation/PUBLIC_IMAGE_AND_P3_INVENTORY.json','public_input_preparation/P3_FIRST_REPLAY/P3_REPLAY_CHECK.json','public_input_pack/INPUT_PACK_QA.json','synth1003_input_pack/INPUT_PACK_QA.json','mixed_fit/GENERATED_NONZERO_FORWARD_ADDENDUM.json','mixed_fit_audit_preparation/DESIGN_AND_PROTOCOL_AUDIT.json']],current_reads=0,PnP=0)
out=ROOT/'SOURCE_INPUT_BINDINGS.json';assert not out.exists();out.write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n');print(json.dumps(b(out)))
