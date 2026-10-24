from pathlib import Path
import hashlib,json
import independent_grid_math as m
R=Path(__file__).resolve().parent;V=R.parent;P=V.parent

def b(x):return dict(path=str(x.resolve()),sha256=hashlib.sha256(x.read_bytes()).hexdigest())
assert b(V/'AUDIT_ENTRY.json')['sha256']=='bdefc2af54e4e9e38eb1dc312116ad230421a9e47eadfa4810b85b5b2b3987b6'
meta=P/'query_visibility_dino_query_support_20261023_v78/feature_pack/INDEX_LINEAGE_INPUT_ONLY.jsonl';rows=[json.loads(x)for x in meta.read_text().splitlines()];assert len(rows)==10002
salt='V79_INDEPENDENT_REPRESENTATIVE_SCALAR_V1';selected=[]
for source in ['SYNTHETIC','YCB_PUBLIC_REAL']:
 for split in ['train','val']:
  for domain in [0,1]:
   choices=[x for x in rows if x['source']==source and x['split']==split and x['domain']==domain]
   selected.append(min(choices,key=lambda a:(hashlib.sha256((salt+'\n'+a['id']).encode()).hexdigest(),a['id'])))
x=dict(status='FROZEN_BEFORE_V79_OUTPUT_ARRAYS_OR_SOURCE_MASK_RGB_READS',entry=b(V/'AUDIT_ENTRY.json'),numeric_math=b(R/'independent_grid_math.py'),source_metadata=b(meta),selection=dict(salt=salt,rule='MinimumSHA(salt+newline+id) perSOURCE/split/domain, tieslexid, no scores/quality/coverage',rows=selected),coordinate_policy=dict(native9='Literal q+(((j+.5)/9)-.5)*L. Algebra(j-4)/9 diagnostic only.',rgb96='Literal q+(((j+.5)/96)-.5)*L, no recentering/rounding.',nearest='floor(x+.5), no epsilon/clip. Nativecontinuous0<=x<W plusNNbounds. Cachedvalid+NN andintersection separate.',query_good='Original enclosing V76 compiler uses finite &not[-1,-1] for both RGB96/native9. Missingflag separate; primaryunchanged. If geometric-only samplecounts exclude thisguard, explicitlylabel them separately.',cached_native9='Actual stored P3 channel64 binary, not source GT-derived.',cached_rgb96='Actual crop_valid96, footprint[-.5,W-.5), not equatedto[0,W).',invalid='Nonfinite/outsideNNignored, allframes/queries retained. No invalidnegative label.'),numeric_policy=dict(strict='Canonical literalFP64 NN/discretecounts compareexact aftermatching explicit validity guards; native cached arrays remainexact.',near_tie='Only flag dist(x+.5,nearestinteger)<=8*u64*(abs(x)+abs(y)+abs(L)+1), never epsilon sampling.',float32_phase='Independentlyinverse normalized P3grid float32 cast; bound u32*abs(norm)*320/gain +FP64rounding. ReportNN/validitydifferences separately; no strictFP64=FP32 requirement or receptivefieldpixelgold claim.',no_new_numeric_performance_gate=True),semantics=dict(V='M4',C='M0&~M4 conditional external pixelproxy',O='~M0 target silhouette absent, notsemanticE/S',invalid='Ignored,neverO/Cnegative',labels='Reference-local V/C/U+self remainfixed; coverage cannotreplace labels/admission.',RGB96='Repeatednativepixels, not independentphysicalobservations'),new_performance_gate=False,model_forward_newfit_current_PnP=0)
p=R/'INDEPENDENT_REPLAY_POLICY_BEFORE_OUTPUTS.json';assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n');mock=m.mock();mock['math_code']=b(R/'independent_grid_math.py');(R/'GENERATED_INDEPENDENT_GRID_TESTS.json').write_text(json.dumps(mock,indent=2)+'\n');print(json.dumps(dict(PASS=True,policy=b(p),selected=[(a['frame_index'],a['id'])for a in selected])))
