"""Actual completed TOMIE CSV schema audit; author code is text evidence only."""
from pathlib import Path
import ast,collections,csv,hashlib,json,math,zlib
import numpy as np
V=Path(__file__).resolve().parents[1];H=V/'annotation_preparation';S=V.parent/'query_visibility_external_only_bounded_prior_20261023_v102/public_real_extension_intake'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def numeric_vector(s,n):
 x=ast.literal_eval(s)
 if not isinstance(x,(list,tuple))or len(x)!=n:raise ValueError('wrong vector shape')
 r=[float(a)for a in x]
 if not all(math.isfinite(v)for v in r):raise ValueError('nonfinite vector')
 return r
completion=V/'raw_pilot/EXTRACTION_COMPLETE_V2.json';c=json.loads(completion.read_text());assert c['completed']and len(c['members'])==8
contract=H/'ANNOTATION_ROW_CONTRACT_BEFORE_CSV_READ.json';assert contract.exists()
source_urls={'generator':'https://github.com/FLW-TUDO/TOMIE-Dataset/blob/main/generator.py','converter':'https://github.com/FLW-TUDO/TOMIE-Dataset/blob/main/coco_converter.py','paper':'https://doi.org/10.1186/s13640-024-00623-6','archive_record':'https://zenodo.org/records/7849183'}
files=[];bycamera={};selectedrgb=[m for m in c['members']if m['kind']=='RGB']
for member in c['members']:
 if member['kind']=='RGB':continue
 p=Path(member['extracted_file']['path']);body=p.read_bytes();assert sha(p)==member['extracted_file']['sha256'];assert f'{zlib.crc32(body)&0xffffffff:08x}'==member['central_directory_record']['crc32']
 text=body.decode('utf-8-sig');parsed=list(csv.reader(text.splitlines(),delimiter=','));header=parsed[0];rr=parsed[1:];assert len(set(header))==len(header)and all(len(r)==len(header)for r in rr);data=[dict(zip(header,r))for r in rr]
 camera=member['original_member_name'].split('/')[1];kind='raw_data'if'raw_data.csv'in member['original_member_name']else'data';bycamera.setdefault(camera,{})[kind]=data
 entity=collections.Counter(r['ObjectName']for r in data);images=collections.Counter(Path(r['fileName']).name for r in data);keys=[(r['fileName'],r['ObjectName'])for r in data]
 positions=[];rotations=[];bad=[];bbox=[]
 for i,r in enumerate(data):
  try:positions.append(numeric_vector(r['Position'],3));rotations.append(numeric_vector(r['Rotation'],3));float(r['Delta_Time']);int(r['Occlusion'])
  except Exception as e:bad.append(dict(row=i+2,error=str(e)))
  if'BoundingBox'in r:bbox.append(numeric_vector(r['BoundingBox'],4))
 assert not bad
 pos=np.array(positions);rot=np.array(rotations);stat=dict(Position_xyz_numeric_min=pos.min(0).tolist(),Position_xyz_numeric_max=pos.max(0).tolist(),Rotation_xyz_numeric_min=rot.min(0).tolist(),Rotation_xyz_numeric_max=rot.max(0).tolist())
 boxstat=None
 if bbox:
  a=np.array(bbox);missing=np.all(a==-1,axis=1);positive=(a[:,2]>0)&(a[:,3]>0)&(a[:,:2]>=0).all(1)
  visible=np.array([int(r['Visible'])for r in data]);assert set(visible)<=set([0,1]);boxstat=dict(allminus1=int(missing.sum()),positive_extents=int(positive.sum()),other=int((~(missing|positive)).sum()),Visible_counts={str(k):int((visible==k).sum())for k in sorted(set(visible))},Visible1_with_missing_bbox=int(((visible==1)&missing).sum()),Visible0_with_positive_bbox=int(((visible==0)&positive).sum()),positive_bbox_numeric_min=a[positive].min(0).tolist()if positive.any()else None,positive_bbox_numeric_max=a[positive].max(0).tolist()if positive.any()else None)
 selected=[]
 for im in selectedrgb:
  if camera not in im['original_member_name']:continue
  image=Path(im['original_member_name']).name;matches=[r for r in data if Path(r['fileName']).name==image]
  selected.append(dict(raw_member=im['original_member_name'],rows=len(matches),entities=[r['ObjectName']for r in matches],raw_entity_fields=matches))
 files.append(dict(camera=camera,kind=kind,member=member['original_member_name'],file=bind(p),CRC32_verified=True,header=header,rows=len(rr),unique_image_names=len(images),unique_entities=len(entity),entity_counts=dict(entity),duplicate_image_entity_keys=len(keys)-len(set(keys)),image_index_minmax=[min(int(Path(x).stem)for x in images),max(int(Path(x).stem)for x in images)],Occlusion_counts=dict(collections.Counter(r['Occlusion']for r in data)),field_presence={k:k in header for k in('BoundingBox','Visible','FOV','FieldOfView','Occlusion','Position','Rotation','Delta_Time','AnnotationTime')},vectors=stat,bbox_Visible=boxstat,selected_RGB_metadata=selected))
join=[]
for camera,dd in bycamera.items():
 rawmap={(r['fileName'],r['ObjectName']):r for r in dd['raw_data']};datamap={(r['fileName'],r['ObjectName']):r for r in dd['data']};common=set(rawmap)&set(datamap);cols=('fileName','ObjectName','Position','Rotation','Occlusion','Delta_Time');diff=[k for k in common if any(rawmap[k][v]!=datamap[k][v]for v in cols)]
 join.append(dict(camera=camera,raw_rows=len(rawmap),data_rows=len(datamap),common_image_entity_rows=len(common),raw_without_data_rows=len(set(rawmap)-set(datamap)),data_without_raw_rows=len(set(datamap)-set(rawmap)),first6field_string_mismatch_rows=len(diff),data_image_range=[min(int(Path(r['fileName']).stem)for r in dd['data']),max(int(Path(r['fileName']).stem)for r in dd['data'])]))
code=dict(generator=bind(S/'TOMIE_generator_py.txt'),converter=bind(S/'TOMIE_coco_converter_py.txt'),README=bind(S/'TOMIE_README_md.txt'))
meaning={'entity_ID':{'status':'KNOWN_AS_RAW_INSTANCE_NAME_NOT_PHYSICAL_CUE_ID','evidence':'Actual ObjectName strings persist across frames/cameras; authorgenerator218–219 maps any PALLET name to class0, not nativecorner.','lines':'generator.py204–219'},'Position':{'status':'UNITS_UNKNOWN_FROM_AVAILABLE_PRIMARY_FILES','known':'Actual finite3vector; generator94–109 interprets as obj2vicon translation, combined withinversecamera2vicon. No scalarunit conversion appears197–202.','limit':'Magnitude is consistent with a millimetre-scale lab frame, but numericrange/BOPrenderer naming is not explicit unit certification. Camera locations calibration and raw exporter unit documentation are not in thispilot.'},'Rotation':{'status':'KNOWN_INTENDED_CODE_CONVENTION_ACTUAL_EXPORT_VERSION_UNVERIFIED','known':"Actual finite3vector. generator96 calls scipyRotation.from_euler('XYZ', values, degrees=False), then object2vicon; camera2vicon inverted103–104. Literal code means radians, orderedXYZ. No PnP invoked here.",'limit':'No raw exporter or matching codecommit evidence proves every CSV was generated withthisexactcodeversion; camera0/7outputschemasdiffer.'},'BoundingBox':{'status':'KNOWN_PRESENT_IN_ACTUAL_DATA_CSV_NOT_RAW_DATA','code_meaning':'generator147–164 returns [row_min,col_min,row_extent,col_extent] from rendered positive depth; generator283–284/converter114–121 swap to imageXY. It is a rendered wholeentitybbox, not a localized physicalcue region.','actual_check':'Numericformat/missing[-1]*4/Visibleconsistency checked; pixels/rendering not replayed.'},'Visible':{'status':'KNOWN_ACTUAL_BOOLEAN_FIELD_WITH_CODE_INTENDED_ENTITY_ELIGIBILITY','code_meaning':'generator142–144 first requires sufficient rendereddepthsum and >=90% of large-render depthsum inimage; lines239–258 setVisible1 onlyif depth_image isnotNone and rawOcclusion==0, else0/bboxmissing. Usesdepthsum, not measuredRGBvisiblepixel fraction.','limit':'Object-level render/FOVeligibility and rawflag checks do not certify particularcorner/part cue D. Visible0 mixesreasons; notE orS.'},'FOV':{'status':'NO_EXPLICIT_FIELD_IN_FOUR_ACTUAL_CSVS','code_evidence':'Only implicit generate_depth_image142–144 render-coverage test. Do not invent a per-cue FOV label.'},'Occlusion':{'status':'NUMERIC_RAW_FLAG_OBSERVED_PHYSICAL_SEMANTICS_UNKNOWN','code_meaning':'generator239 reads int and240 rejects nonzero forVisible; source excerpt doesnotdefine RGBexternalpartcoverage versus Vicontrackingocclusion.','actual_counts':'Recorded perfile; evenOcclusion0 coexistswithVisible0/missingbbox, so0doesnotproveRGBcuevisible.'},'Delta_Time':{'status':'FINITE_NUMERIC_FIELD_MEANING_UNITS_NOT_ESTABLISHED_BY_AVAILABLE_CODE','known':'Preserved unmodified bygenerator, not a trustworthyseconds/time-neighbour key without exporterdefinition.'}}
result=dict(completed=True,PASS_schema_execution=True,contract=bind(contract),completion=bind(completion),source_URLs=source_urls,code_read_as_text_only=code,actual_CSV_files=files,data_raw_joins=join,field_meaning=meaning,primary_version_caveat='Author generator readsnew_data.csv withcamids3..7 and writesannotated_data.csv9cols; archiveactualfilesnameddata.csv,cam0has9cols/cam7has8cols. Mechanism is code-intended evidence, not proof thisexactgeneratorversioncreatedallarchiveCSVs.',capture_grouping='Allselectedcamera0/camera7andtime-neighboursinonescenario_3_without_dist_LVL_0episodeconservativelyonefamily; entitynamesandframebasenamejoinsretained. Crosscameraexacttimesyncisnotproven bysharednumericbasename.',physical_cue_readiness='Rawwholeentitymetadata+poses are usefulassociationcandidates, not physicalcueD/E. No localizedreferencepoint/uncertainty/nativecorner label is supplied by these4CSVs.',future_required='Freezeactualinitializer allq/L/candidate/order/failures first, then annotate independent target-ownedphysicalcue evidence anduncertainty; retain naturalEonlywithdefensiblecuecorrespondence. No GTcentre/bestquery selection.',new_human_native_gold_or_training_labels=0,training_admission=False,sourceRGB_pixels_read_by_this_audit=0,author_code_executed_or_copied=False,new_model_fit_private_current_PnP=0)
write(H/'ACTUAL_CSV_SCHEMA_AND_CUE_READINESS.json',result)
write(H/'SEAL.json',dict(completed=True,PASS_metadata_audit=True,contract=bind(contract),actual_CSV_audit=bind(H/'ACTUAL_CSV_SCHEMA_AND_CUE_READINESS.json'),code=bind(__file__),raw_completion=bind(completion),RGB_model_private_current_PnP=0,pilot_training_admission=False))
print(json.dumps(dict(result=bind(H/'ACTUAL_CSV_SCHEMA_AND_CUE_READINESS.json'),seal=bind(H/'SEAL.json'),files=[dict(name=f['member'],rows=f['rows'],images=f['unique_image_names'],entities=f['unique_entities'],Visible=f['bbox_Visible'])for f in files],joins=join)))
