"""Additive V2 wording/scope review; reuses preserved independent V1 count audit."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib,json,copy
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
class Tables(HTMLParser):
    def __init__(self): super().__init__(); self.tables=[];self.table=None;self.row=None;self.cell=None
    def handle_starttag(self,t,a):
        if t=='table': self.table=[]
        elif t=='tr': self.row=[]
        elif t in ('td','th'): self.cell=[]
    def handle_data(self,d):
        if self.cell is not None: self.cell.append(d)
    def handle_endtag(self,t):
        if t in ('td','th'): self.row.append(''.join(self.cell));self.cell=None
        elif t=='tr': self.table.append(self.row);self.row=None
        elif t=='table': self.tables.append(self.table);self.table=None
out=ROOT/'source_audit/PROGRESS_REPORT_REVIEW.json'
prior=ROOT/'source_audit/PROGRESS_REPORT_REVIEW_V1.json'
old_bytes=out.read_bytes(); assert hashlib.sha256(old_bytes).hexdigest()=='b507740db105b29f46f2cf11780c50e040aa649125e23391e15e1035449f0dac'
assert not prior.exists();prior.write_bytes(old_bytes)
old=load(prior)
clarify_path=ROOT/'source_audit/REPORT_SCOPE_CLARIFICATION_V2.json';clarify=load(clarify_path)
for section in ['preserved_V1','revised']:
    for item in clarify[section]: assert sha(item['path'])==item['sha256']
for binding in old['reviewed_report_bindings']:
    preserved=ROOT/'provenance/REPORT_V1'/Path(binding['path']).name
    assert sha(preserved)==binding['sha256']
# No source-row count reconstruction repeated; ensure every previously audited source is unchanged.
for binding in old['source_bindings']: assert sha(binding['path'])==binding['sha256']
current=load(ROOT/'PROGRESS_KO.json'); previous=load(ROOT/'provenance/REPORT_V1/PROGRESS_KO.json')
assert current['counterexample_denominator_scope']==clarify['scope']
assert clarify['scope']['primary_FULL_supervised_scope']=={'D_with_background_query':42,'E_with_target_query':2,'D_supervised_rows':54,'E_supervised_rows':27}
assert clarify['scope']['all_underlying_reference_states_including_MISS_and_stress']=={'D_with_background_query':69,'E_with_target_query':16}
changed=sorted(k for k in set(current)|set(previous) if current.get(k)!=previous.get(k))
assert changed==['counterexample_denominator_scope','previous_report_snapshot','wording_revision_V2'],changed
h=Tables();h.feed((ROOT/'PROGRESS_KO.html').read_text())
table=next(t for t in h.tables if t[0][0]=='원래 reference 상태와 입력 상태')
assert ['FULL primary D 감독행 중 query 픽셀이 배경','42 crop행'] in table
assert ['FULL primary E 감독행 중 query 픽셀이 대상','2 crop행'] in table
result=copy.deepcopy(old)
result.update(schema='V71_PROGRESS_REPORT_INDEPENDENT_REVIEW_V2',status='PASS',
    reviewed_report_bindings=[{'path':str(ROOT/f),'sha256':sha(ROOT/f)} for f in ['PROGRESS_KO.json','PROGRESS_KO.html']],
    previous_independent_audit={'path':str(prior),'sha256':sha(prior)},
    scope_clarification_binding={'path':str(clarify_path),'sha256':sha(clarify_path)},
    V2_supplement_checks={'prior_audit_byte_preserved':True,'prior_report_snapshots_hash_match':True,'all_previous_source_bindings_unchanged':True,'only_json_fields_changed':changed,'HTML_primary_FULL_D_and_E_denominators_explicit':True,'scope_metadata_matches_previous_independent_counts':True,'all_clarification_bindings_match':True},
    minor_clarity_advisory='RESOLVED: FULL primary D54/E27 supervised denominator is explicit; 42/2 distinguished from all underlying reference-state 69/16.',
    audit_code={'path':str(Path(__file__)),'sha256':sha(Path(__file__))})
result['numeric_reconciliation']['SYNTH']['all_underlying_reference_counterexamples_not_reported_as_primary']['now_reported_in_separate_scope_metadata']=True
result['read_scope']['V2_followup']='Read report JSON/HTML and existing audit metadata; hash-only verification of source bindings and V1 PNG/build snapshot. No source-row reconstruction repeated.'
result['limits']+=' V2 follow-up verifies wording/scope changes and unchanged bindings, reusing the preserved V1 numeric reconciliation.'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'path':str(out),'sha256':sha(out),'previous_audit_sha256':sha(prior),'clarification_sha256':sha(clarify_path),'reports':result['reviewed_report_bindings']},ensure_ascii=False))
