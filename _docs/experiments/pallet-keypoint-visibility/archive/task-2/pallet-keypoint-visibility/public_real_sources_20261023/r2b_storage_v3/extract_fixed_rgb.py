"""Decode only preregistered fixed RGB message IDs from the verified public bag.

This is a small schema-specific ROS2 Image CDR1 reader, not a ROS executable.
It has no model, current-image or annotation dependency.
"""
from pathlib import Path
import csv,hashlib,json,sqlite3,struct,sys,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import yaml,zstandard

ROOT=Path(__file__).resolve().parent
DENY=ROOT.parents[1]/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json'
TOPICS=('d455_1_rgb_image','hawk_0_left_rgb_image')
SCHEMAS={'Image':'https://raw.githubusercontent.com/ros2/common_interfaces/rolling/sensor_msgs/msg/Image.msg',
         'Header':'https://raw.githubusercontent.com/ros2/common_interfaces/rolling/std_msgs/msg/Header.msg',
         'ZstdMessage':'https://raw.githubusercontent.com/ros2/rosbag2/rolling/rosbag2_compression_zstd/src/rosbag2_compression_zstd/zstd_decompressor.cpp'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def bh(b):return hashlib.sha256(b).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,d):
    assert not Path(p).exists(),str(p)
    Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

class CDR:
    def __init__(self,b):
        assert b[:4]in [b'\x00\x01\x00\x00',b'\x00\x00\x00\x00'],('Unsupported CDR encapsulation',b[:4].hex())
        self.b=b;self.pos=4;self.endian='<'if b[1]==1 else'>'
    def align(self,n):self.pos+=( -(self.pos-4))%n
    def number(self,fmt,align):
        self.align(align);v=struct.unpack_from(self.endian+fmt,self.b,self.pos)[0];self.pos+=struct.calcsize(fmt);return v
    def string(self):
        n=self.number('I',4);assert 1<=n<=1024
        b=self.b[self.pos:self.pos+n];assert len(b)==n and b[-1]==0
        self.pos+=n;return b[:-1].decode('utf-8')

def decode_image(b):
    c=CDR(b);sec=c.number('i',4);nanosec=c.number('I',4);assert nanosec<1000000000
    frame_id=c.string();h=c.number('I',4);w=c.number('I',4);encoding=c.string();big=c.number('B',1);step=c.number('I',4);n=c.number('I',4)
    assert 0<h<=10000 and 0<w<=10000 and big in [0,1]
    assert n==h*step and c.pos+n<=len(b)
    tail=b[c.pos+n:];assert len(tail)<=3 and not any(tail)
    assert encoding in ['rgb8','bgr8','rgba8','bgra8','mono8'],('Unsupported observed encoding',encoding)
    channels={'rgb8':3,'bgr8':3,'rgba8':4,'bgra8':4,'mono8':1}[encoding]
    assert step>=w*channels
    packed=np.frombuffer(b,dtype=np.uint8,count=n,offset=c.pos).reshape(h,step)[:,:w*channels].reshape(h,w,channels)
    if encoding=='mono8':rgb=np.repeat(packed,3,axis=2)
    elif encoding in ['bgr8','bgra8']:rgb=packed[:,:,[2,1,0]]
    else:rgb=packed[:,:,:3]
    return np.ascontiguousarray(rgb),dict(height=h,width=w,encoding=encoding,is_bigendian=big,step=step,data_bytes=n,
        serialized_bytes=len(b),cdr_encapsulation=b[:4].hex(),header_frame_id=frame_id,header_timestamp_ns=sec*1000000000+nanosec,
        row_padding_bytes=step-w*channels,trailing_serialization_padding=len(tail))

def mock():
    tests=[]
    def build(order,frame,encoding,pixels,step):
        b=bytearray(b'\x00\x01\x00\x00'if order=='<'else b'\x00\x00\x00\x00')
        def put(fmt,v,align):
            b.extend(b'\x00'*((-(len(b)-4))%align));b.extend(struct.pack(order+fmt,v))
        def string(x):
            y=x.encode()+b'\x00';put('I',len(y),4);b.extend(y)
        put('i',12,4);put('I',345,4);string(frame);put('I',2,4);put('I',2,4);string(encoding);put('B',order=='>',1);put('I',step,4);put('I',len(pixels),4);b.extend(pixels)
        return bytes(b)
    expected=np.arange(12,dtype=np.uint8).reshape(2,2,3)
    for endian in ['<','>']:
        for frame in ['x','optical_frame']:
            for encoding in ['rgb8','bgr8']:
                raw=expected if encoding=='rgb8'else expected[:,:,::-1]
                padded=b''.join(row.tobytes()+b'\x77\x88'for row in raw)
                rgb,meta=decode_image(build(endian,frame,encoding,padded,8));assert np.array_equal(rgb,expected)
                tests.append(dict(endian=endian,frame_id=frame,encoding=encoding,padded_stride=True,PASS=True))
    return dict(PASS=True,tests=tests,no_public_or_private_image_reads=True)

def main():
    start=time.monotonic();verified=json.loads((ROOT/'DOWNLOAD_VERIFICATION.json').read_text());assert verified['PASS']and verified['actual_md5']=='5e8f11201fe10dbac7307a8628553e94'
    write(ROOT/'CDR_DECODER_MOCK_TEST.json',mock())
    protocol=dict(schema='FIXED_RGB_CDR1_DECODE_PROTOCOL_V1',code=ref(Path(__file__)),download_verification=ref(ROOT/'DOWNLOAD_VERIFICATION.json'),
        intake_protocol=ref(ROOT/'INTAKE_PROTOCOL.json'),official_schemas=SCHEMAS,
        serialization='ROS2 CDR1 encapsulation; Header(sec,nanosec,frame_id), height,width,encoding,is_bigendian,step,uint8 data sequence; alignment measured after4byte encapsulation.',
        compression='metadata mode MESSAGE+zstd: decompress only the selected RGB message blobs, not entire bag.',
        selected_topics=list(TOPICS),indices='np.linspace(0,N-1,min(16,N)).astype(int); sorted(timestamp,messageid), fixed before decompressing any public image.',
        no_depth_IR_right_camera_or_point_cloud_payload_decode=True,GT_masks_annotations_available=False,
        training_admission='HOLD: source family, label quality and independent scene/session TRAIN/VAL must be resolved separately.',
        runtime=dict(executable=sys.executable,numpy=np.__version__,zstandard=zstandard.__version__),mock=ref(ROOT/'CDR_DECODER_MOCK_TEST.json'))
    write(ROOT/'DECODER_PROTOCOL.json',protocol)
    bag=ROOT/'r2b_storage_0.db3';assert bag.stat().st_size==verified['actual_size']
    c=sqlite3.connect('file:'+str(bag)+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
    integrity=[r[0]for r in c.execute('PRAGMA quick_check')];assert integrity==['ok']
    topic_rows=[dict(r)for r in c.execute('SELECT * FROM topics ORDER BY id')]
    counts={r[0]:dict(count=r[1],min_timestamp=r[2],max_timestamp=r[3])for r in c.execute('SELECT topic_id,count(*),min(timestamp),max(timestamp) FROM messages GROUP BY topic_id')}
    meta=yaml.safe_load((ROOT/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    expected={r['topic_metadata']['name']:r for r in meta['topics_with_message_count']}
    for r in topic_rows:
        assert r['name']in expected and r['type']==expected[r['name']]['topic_metadata']['type']
        assert counts[r['id']]['count']==expected[r['name']]['message_count']
        r['observed_message_stats']=counts[r['id']]
    assert sum(r['count']for r in counts.values())==meta['message_count']==4034
    write(ROOT/'SQLITE_SCHEMA_AND_TOPICS.json',dict(PASS=True,quick_check=integrity,topics=topic_rows,
        schema=[dict(r)for r in c.execute("SELECT name,sql FROM sqlite_master WHERE type IN ('table','index') ORDER BY name")],metadata_counts_exact=True))
    selection=[]
    for name in TOPICS:
        t=next(r for r in topic_rows if r['name']==name);assert t['type']=='sensor_msgs/msg/Image'and t['serialization_format']=='cdr'
        messages=[tuple(r)for r in c.execute('SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp,id',(t['id'],))]
        indices=np.linspace(0,len(messages)-1,min(16,len(messages))).astype(int)
        assert len(set(indices))==len(indices)
        for slot,idx in enumerate(indices):
            mid,ts=messages[idx]
            selection.append(dict(topic=name,topic_id=t['id'],sensor='D455_COLOR'if name.startswith('d455')else'HAWK_LEFT',sensor_slot=slot,
                index_in_sorted_topic=int(idx),topic_message_count=len(messages),bag_message_id=mid,bag_timestamp_ns=ts))
    lock=dict(protocol=ref(ROOT/'DECODER_PROTOCOL.json'),bag_sha256=verified['sha256'],selection_before_any_image_decode=True,RGB_image_count=len(selection),rows=selection)
    write(ROOT/'SELECTION_LOCK.json',lock)
    raw_deny=json.loads(DENY.read_text());denied=set(raw_deny['image_sha256']);assert len(denied)==223
    (ROOT/'RGB_FIXED').mkdir(exist_ok=False);rows=[];zstd=zstandard.ZstdDecompressor()
    for s in selection:
        blob=c.execute('SELECT data FROM messages WHERE id=? AND topic_id=?',(s['bag_message_id'],s['topic_id'])).fetchone()[0]
        assert blob[:4]==b'\x28\xb5\x2f\xfd'
        cdr=zstd.decompress(blob);rgb,m=decode_image(cdr)
        name=f"{s['sensor']}_{s['sensor_slot']:02d}_idx{s['index_in_sorted_topic']:03d}_msg{s['bag_message_id']:04d}.png"
        path=ROOT/'RGB_FIXED'/name;Image.fromarray(rgb,'RGB').save(path,compress_level=6)
        assert np.array_equal(np.asarray(Image.open(path).convert('RGB')),rgb),'PNG roundtrip'
        out={**s,**m,'image':str(path),'image_sha256':sha(path),'RGB_pixels_sha256':bh(rgb.tobytes()),
            'compressed_message_sha256':bh(blob),'compressed_message_bytes':len(blob),'CDR_message_sha256':bh(cdr),
            'header_minus_bag_timestamp_ns':m['header_timestamp_ns']-s['bag_timestamp_ns'],'GT_or_mask':None}
        assert out['image_sha256']not in denied and out['RGB_pixels_sha256']not in denied
        rows.append(out)
    c.close()
    with (ROOT/'RGB_FIXED_MANIFEST.jsonl').open('x')as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    with (ROOT/'RGB_FIXED_MANIFEST.csv').open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    make_contact(rows)
    inventory=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file()and p.name not in ['INTAKE_SEAL.json','INTAKE_INVENTORY.json']:
            h=verified['sha256']if p==bag else sha(p)
            inventory.append(dict(path=str(p),relative_path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=h))
    write(ROOT/'INTAKE_INVENTORY.json',dict(files=inventory,original_bag_streaming_SHA_reused=True))
    seal=dict(PASS=True,schema='PUBLIC_R2B_STORAGE_V3_FIXED_RGB_INTAKE_SEAL_V1',source=dict(version='3',publisher='NVIDIA',site='NVIDIA facilities in United States',license='CC-BY-4.0',card='https://catalog.ngc.nvidia.com/orgs/nvidia/isaac/resources/r2bdataset2023/3'),
        original_db3=dict(path=str(bag),bytes=bag.stat().st_size,sha256=verified['sha256'],md5=verified['actual_md5']),
        RGB_images=len(rows),sensors={t:sum(r['topic']==t for r in rows)for t in TOPICS},
        image_shapes_and_encodings=sorted(set((r['height'],r['width'],r['encoding'])for r in rows)),
        denylist=ref(DENY),exact_file_or_raw_RGB_hash_overlap=0,
        exact_hash_limitation='Current denylist contains original file hashes. Re-encoded PNG and rawRGB hash nonmatches do not alone establish recording-family independence.',
        inventory=ref(ROOT/'INTAKE_INVENTORY.json'),manifest=ref(ROOT/'RGB_FIXED_MANIFEST.jsonl'),selection_lock=ref(ROOT/'SELECTION_LOCK.json'),
        contact_sheet=ref(ROOT/'FIXED_RGB_CONTACT_SHEET.png'),GT_or_pallet_masks_available=False,
        training_admission='HOLD: RGB inspection only; source-family exclusion, trustworthy target labels and scene/session TRAIN/VAL protocol are separate pending gates.',
        model_backbone_feature_forward_training_current_RGB_GT_PnP_reads_or_calls=0,depth_payloads_decoded=0,
        seconds=time.monotonic()-start)
    write(ROOT/'INTAKE_SEAL.json',seal)
    print('FIXED_RGB_INTAKE_PASS',len(rows),seal['image_shapes_and_encodings'],'sealSHA',sha(ROOT/'INTAKE_SEAL.json'),'seconds',seal['seconds'],flush=True)

def make_contact(rows):
    W,H=384,276;canvas=Image.new('RGB',(W*8,H*4+78),'#17212b');d=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    big=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    d.text((16,12),'NVIDIA r2b_storage v3: fixed temporal RGB samples (no content selection)',font=big,fill='white')
    d.text((16,43),'Top 2 rows: D455 COLOR | Bottom 2 rows: HAWK LEFT | 16 evenly spaced topic indices each | No masks / GT',font=font,fill='#c9d6e2')
    t0=min(r['bag_timestamp_ns']for r in rows)
    for j,r in enumerate(rows):
        col=j%8;row=j//8;x=col*W;y=78+row*H
        im=Image.open(r['image']).convert('RGB');im.thumbnail((W-8,H-60),Image.Resampling.LANCZOS)
        canvas.paste(im,(x+(W-im.width)//2,y+(H-60-im.height)//2))
        d.text((x+6,y+H-56),f"{r['sensor']}  idx={r['index_in_sorted_topic']}/{r['topic_message_count']-1}",font=font,fill='white')
        d.text((x+6,y+H-33),f"msg={r['bag_message_id']}  t={(r['bag_timestamp_ns']-t0)/1e9:.3f}s",font=font,fill='#c9d6e2')
    canvas.save(ROOT/'FIXED_RGB_CONTACT_SHEET.png')

if __name__=='__main__':main()
