"""Fixed16 public RGB images: MCAP container + previously verified Image CDR."""
from pathlib import Path
from collections import Counter
import csv,hashlib,importlib.util,json,sys,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import yaml

ROOT=Path(__file__).resolve().parent;PREVIOUS=ROOT.parent/'r2b_storage_v3'
sys.path.insert(0,str(ROOT/'reader_overlay'))
from mcap.reader import make_reader
TOPIC='/camera/color/image_raw'
CDR_SOURCE=PREVIOUS/'extract_fixed_rgb.py'
spec=importlib.util.spec_from_file_location('verified_source_image_cdr',CDR_SOURCE);cdr=importlib.util.module_from_spec(spec);spec.loader.exec_module(cdr)
DENY=ROOT.parents[1]/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,d):
    assert not p.exists();p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def identity(message,ordinal):
    return dict(channel_id=message.channel_id,sequence=message.sequence,bag_timestamp_ns=message.log_time,publish_timestamp_ns=message.publish_time,
        topic_record_ordinal=ordinal,source_bag_message_id=f'ch{message.channel_id}:seq{message.sequence}:log{message.log_time}:ord{ordinal}')

def main():
    start=time.monotonic();assert not(ROOT/'INTAKE_SEAL.json').exists()
    verified=json.loads((ROOT/'DOWNLOAD_VERIFICATION.json').read_text());assert verified['PASS']and verified['actual_size']==1033512670 and verified['actual_md5']=='78c96e908e158c5b16213212e8362ad3'
    bag=ROOT/'r2b_galileo2_0.mcap';assert bag.stat().st_size==verified['actual_size']
    with bag.open('rb')as f:
        assert f.read(8)==b'\x89MCAP0\r\n';f.seek(-8,2);assert f.read(8)==b'\x89MCAP0\r\n'
    write(ROOT/'CDR_REUSE_MOCK_TEST.json',dict(**cdr.mock(),reused_code=ref(CDR_SOURCE),previous_mock=ref(PREVIOUS/'CDR_DECODER_MOCK_TEST.json')))
    protocol=dict(schema='R2B_GALILEO2_V1_MCAP_FIXED_RGB_EXTRACTION_V1',code=ref(Path(__file__)),intake_protocol=ref(ROOT/'INTAKE_PROTOCOL.json'),
        official_metadata=ref(ROOT/'metadata.yaml'),reader_overlay=ref(ROOT/'MCAP_READER_OVERLAY_LOCK.json'),CDR_decoder=ref(CDR_SOURCE),CDR_mock=ref(ROOT/'CDR_REUSE_MOCK_TEST.json'),
        container='MCAP ROS2 records, official mcap Python reader in a localoverlay only; validate_crcs=True. No message/image decoder factory installed.',
        sensor='Publisher ros2_benchmark README explicitly lists RealSenseD455 forgalileo2. Actualtopic anddimensions are verified from thisbag, not copied from nominal docs.',
        topic=TOPIC,selection='Sort301RGB message identities by(log_time,topic_file_order_ordinal); np.linspace(0,N-1,min(16,N)).astype(int). Lock selection before CDRImage pixel decoding.',
        payloads='Only selectedRGB CDRImage payloads decoded. MCAP chunks may contain othermessages; no depth/IR/calibration/TF payloads decoded.',
        source_family='Alltimes/sensors fromthiscapture share onefamily. Differentfilename/year is not evidence of independence from2023storage.',
        no_model_or_teacher_or_current_RGB_GT_or_PnP=True,admission='HOLD; RGBcandidate only, no keypoint/visibilityGT')
    write(ROOT/'DECODER_PROTOCOL.json',protocol)
    meta=yaml.safe_load((ROOT/'metadata.yaml').read_text())['rosbag2_bagfile_information'];expected={r['topic_metadata']['name']:r for r in meta['topics_with_message_count']}
    with bag.open('rb')as f:
        reader=make_reader(f,validate_crcs=True);summary=reader.get_summary();header=reader.get_header();assert summary is not None
        assert summary.statistics.message_count==meta['message_count']==6140
        channels=[]
        for cid,ch in summary.channels.items():
            schema=summary.schemas[ch.schema_id];count=summary.statistics.channel_message_counts[cid]
            assert ch.topic in expected and schema.name==expected[ch.topic]['topic_metadata']['type']and count==expected[ch.topic]['message_count']
            channels.append(dict(channel_id=cid,topic=ch.topic,message_encoding=ch.message_encoding,schema_id=ch.schema_id,schema_name=schema.name,
                schema_encoding=schema.encoding,schema_data_sha256=hashlib.sha256(schema.data).hexdigest(),message_count=count))
        assert len(channels)==9
        chosen=next(ch for ch in channels if ch['topic']==TOPIC);assert chosen['schema_name']=='sensor_msgs/msg/Image'and chosen['message_encoding']=='cdr'and chosen['message_count']==301
        comp=sorted(set(ch.compression for ch in summary.chunk_indexes))
        schema_record=dict(PASS=True,header=dict(profile=header.profile,library=header.library),channels=channels,metadata_counts_exact=True,
            chunks=len(summary.chunk_indexes),chunk_compressions=comp,validate_crcs=True,source_bag_message_count=6140,bag_duration_ns=meta['duration']['nanoseconds'])
        write(ROOT/'MCAP_SCHEMA_AND_TOPICS.json',schema_record)
        identities=[]
        for ordinal,(schema,ch,message)in enumerate(reader.iter_messages(topics=TOPIC,log_time_order=False)):
            assert schema.name=='sensor_msgs/msg/Image'and ch.topic==TOPIC
            identities.append(identity(message,ordinal))
    assert len(identities)==301
    ordered=sorted(identities,key=lambda x:(x['bag_timestamp_ns'],x['topic_record_ordinal']));indices=np.linspace(0,len(ordered)-1,16).astype(int)
    selection=[dict(**ordered[int(index)],sensor='D455_COLOR',topic=TOPIC,sensor_slot=slot,index_in_sorted_topic=int(index),topic_message_count=301)for slot,index in enumerate(indices)]
    write(ROOT/'SELECTION_LOCK.json',dict(protocol=ref(ROOT/'DECODER_PROTOCOL.json'),source_bag_sha256=verified['sha256'],RGB_message_count=301,
        image_pixels_decoded_before_lock=0,selected_indices=indices.tolist(),RGB_image_count=16,rows=selection))
    selected_by_ordinal={r['topic_record_ordinal']:r for r in selection};rows=[];denied=set(json.loads(DENY.read_text())['image_sha256']);assert len(denied)==223
    old=[json.loads(x)for x in(PREVIOUS/'RGB_FIXED_MANIFEST.jsonl').read_text().splitlines()]
    old_file={r['image_sha256']for r in old};old_raw={r['RGB_pixels_sha256']for r in old}
    (ROOT/'RGB_FIXED').mkdir(exist_ok=False)
    with bag.open('rb')as f:
        reader=make_reader(f,validate_crcs=True)
        for ordinal,(schema,ch,message)in enumerate(reader.iter_messages(topics=TOPIC,log_time_order=False)):
            if ordinal not in selected_by_ordinal:continue
            s=selected_by_ordinal[ordinal];assert identity(message,ordinal)=={k:s[k]for k in identity(message,ordinal)}
            rgb,m=cdr.decode_image(message.data)
            name=f"D455_COLOR_{s['sensor_slot']:02d}_idx{s['index_in_sorted_topic']:03d}_ord{ordinal:03d}.png";path=ROOT/'RGB_FIXED'/name
            Image.fromarray(rgb,'RGB').save(path,compress_level=6);assert np.array_equal(np.asarray(Image.open(path).convert('RGB')),rgb)
            row=dict(**s,**m,image=str(path),image_sha256=sha(path),RGB_pixels_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),CDR_message_sha256=hashlib.sha256(message.data).hexdigest(),
                header_minus_bag_timestamp_ns=m['header_timestamp_ns']-s['bag_timestamp_ns'],GT_or_mask=None)
            assert row['image_sha256']not in denied and row['RGB_pixels_sha256']not in denied
            rows.append(row)
    rows.sort(key=lambda r:r['sensor_slot']);assert len(rows)==16
    with(ROOT/'RGB_FIXED_MANIFEST.jsonl').open('x')as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    with(ROOT/'RGB_FIXED_MANIFEST.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    contact(rows);family_overview(rows,old)
    record=dict(PASS=True,source='NVIDIA r2bDataset2024v1 r2b_galileo2',topic=TOPIC,sensor_model_publisher='RealSenseD455',
        sensor_model_source='https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark/blob/main/README.md',actual_shapes_and_encodings=sorted(set((r['height'],r['width'],r['encoding'])for r in rows)),
        nominal_dimensions_not_used=True,RGB301_message_logtime_span_ns=ordered[-1]['bag_timestamp_ns']-ordered[0]['bag_timestamp_ns'],selected_frames=16,
        exact_file_or_raw_RGB_overlap_with_current_denylist=0,exact_file_overlap_with_storage_fixed32=sum(r['image_sha256']in old_file for r in rows),
        exact_raw_RGB_overlap_with_storage_fixed32=sum(r['RGB_pixels_sha256']in old_raw for r in rows),
        exact_hash_caveat='Currentdenylistoriginalfilehashes versusnewPNG/rawRGB nonmatch cannot prove recordingfamily independence.',
        depth_IR_right_camera_or_TF_payloads_decoded=0,CDR_RGB_images_decoded=16,source_model_teacher_or_current_data_calls=0)
    write(ROOT/'RGB_AND_DUPLICATE_AUDIT.json',record)
    inventory=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file()and path.name not in ['INTAKE_INVENTORY.json','INTAKE_SEAL.json']:
            inventory.append(dict(path=str(path),relative_path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=verified['sha256']if path==bag else sha(path)))
    write(ROOT/'INTAKE_INVENTORY.json',dict(files=inventory,original_MCAPPublisherMD5_andstreamSHA_verified=True))
    seal=dict(PASS=True,schema='PUBLIC_R2B_GALILEO2_V1_FIXED_RGB_INTAKE_SEAL_V1',original_mcap=dict(path=str(bag),bytes=bag.stat().st_size,sha256=verified['sha256'],md5=verified['actual_md5']),
        source=dict(version='1',publisher='NVIDIA',license='CC-BY-4.0',site='NVIDIA facilities inUnitedStates',card='https://catalog.ngc.nvidia.com/orgs/nvidia/isaac/resources/r2bdataset2024/1'),
        RGB_images=16,total_RGB_messages=301,actual_shapes_and_encodings=record['actual_shapes_and_encodings'],manifest=ref(ROOT/'RGB_FIXED_MANIFEST.jsonl'),selection=ref(ROOT/'SELECTION_LOCK.json'),
        contact_sheet=ref(ROOT/'FIXED_RGB_CONTACT_SHEET.png'),family_overview=ref(ROOT/'CAPTURE_FAMILY_FIXED_OVERVIEW.png'),RGB_audit=ref(ROOT/'RGB_AND_DUPLICATE_AUDIT.json'),inventory=ref(ROOT/'INTAKE_INVENTORY.json'),
        recording_family_admission='HOLD/UNKNOWN until visual/provenance review;different clipname alone not independence.',GT_masks_or_semantic_visibility_labels=False,training_admission='HOLD: conditionalpublicRGBcandidateonly',
        actual_network_download_bytes_approx=bag.stat().st_size+sum((ROOT/n).stat().st_size for n in ['OFFICIAL_API_FILES.json','metadata.yaml','OFFICIAL_CARD.html','LICENSE_CC_BY_4_0.html','OFFICIAL_ROS2_BENCHMARK_README.md'])+21955,
        no_existing_ML_environment_changes=True,new_model_training_forward_current_RGB_GT_PnP=0,seconds=time.monotonic()-start)
    assert seal['actual_network_download_bytes_approx']<1200000000
    write(ROOT/'INTAKE_SEAL.json',seal);print('GALILEO2_FIXED_RGB_INTAKE_PASS',sha(ROOT/'INTAKE_SEAL.json'),record,flush=True)

def contact(rows):
    W,H=384,324;im=Image.new('RGB',(W*4,84+H*4),'#17212b');d=ImageDraw.Draw(im);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    d.text((12,8),'NVIDIA r2b_galileo2 v1:16fixed timestamp RGB frames',font=title,fill='white');d.text((12,42),'D455 color topic / no content selection / no masks or semanticGT / sourcefamily and training HOLD',font=font,fill='#d7e4ed')
    t0=min(r['bag_timestamp_ns']for r in rows)
    for i,r in enumerate(rows):
        x=(i%4)*W;y=84+(i//4)*H;v=Image.open(r['image']).convert('RGB');v.thumbnail((W-8,H-56),Image.Resampling.LANCZOS);im.paste(v,(x+(W-v.width)//2,y))
        d.text((x+6,y+H-52),f"slot{r['sensor_slot']:02d} idx{r['index_in_sorted_topic']:03d}/300",font=font,fill='white')
        d.text((x+6,y+H-28),f"t={(r['bag_timestamp_ns']-t0)/1e9:.3f}s | {r['width']}x{r['height']} {r['encoding']}",font=font,fill='#d7e4ed')
    im.save(ROOT/'FIXED_RGB_CONTACT_SHEET.png')

def family_overview(rows,old):
    old=[r for r in old if r['sensor']=='D455_COLOR'and r['sensor_slot']in[0,5,10,15]];new=[r for r in rows if r['sensor_slot']in[0,5,10,15]]
    write(ROOT/'FAMILY_OVERVIEW_SELECTION.json',dict(selection='Fixedordinal0,5,10,15 fromeachalreadylockedsourceD45516list, independent of imagecontent orscores.',
        STORAGE=[{k:r[k]for k in ['sensor_slot','image','image_sha256']}for r in old],GALILEO2=[{k:r[k]for k in ['sensor_slot','image','image_sha256']}for r in new]))
    W,H=480,330;im=Image.new('RGB',(W*4,82+H*2),'#17212b');d=ImageDraw.Draw(im);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    d.text((10,8),'Fixed capture-family overview: STORAGE2023(top) vs GALILEO2_2024(bottom)',font=font,fill='white');d.text((10,38),'Different sequence/year and different pixelhashes do not prove a different warehouse, pallet or recordingfamily.',font=font,fill='#d7e4ed')
    for row,group in enumerate([old,new]):
        for col,r in enumerate(group):
            x=col*W;y=82+row*H;v=Image.open(r['image']).convert('RGB');v.thumbnail((W-8,H-34),Image.Resampling.LANCZOS);im.paste(v,(x+(W-v.width)//2,y))
            d.text((x+6,y+H-28),f"{'STORAGE2023'if row==0 else'GALILEO2_2024'} slot{r['sensor_slot']}",font=font,fill='white')
    im.save(ROOT/'CAPTURE_FAMILY_FIXED_OVERVIEW.png')

if __name__=='__main__':main()
