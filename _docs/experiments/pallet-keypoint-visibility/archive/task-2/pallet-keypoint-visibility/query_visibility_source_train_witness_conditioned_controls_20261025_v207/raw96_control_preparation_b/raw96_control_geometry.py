"""Pure post-sampler pixel geometry; no dataset/model loaders or label creation."""
import hashlib,math
import numpy as np

LONGSIDES=(32,40,48,56,64,72,80)
ROTATIONS=(0,90,180,270)

def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()

def rectangle(roi,shape=(96,96)):
    assert len(roi)==4 and all(type(x)is int for x in roi)
    x0,y0,x1,y1=roi;h,w=shape
    assert 0<=x0<=x1<w and 0<=y0<=y1<h
    out=np.zeros(shape,bool);out[y0:y1+1,x0:x1+1]=True
    return out

def obligation(case):
    if not case['original_query_available']:return None,'ORIGINAL_INPUT_U'
    if case['state_A']!='YES' or case['state_C']!='YES':return None,'UNKNOWN_WITNESS'
    if case['ROI_A'] is None or case['ROI_C'] is None:return None,'UNKNOWN_WITNESS'
    return rectangle(case['ROI_A'])|rectangle(case['ROI_C']),'KNOWN_WEAK_FEATURE_ONLY'

def erode1(alpha):
    assert alpha.ndim==2 and alpha.dtype==np.bool_
    h,w=alpha.shape;p=np.pad(alpha,1,constant_values=False)
    out=np.ones((h,w),bool)
    for y in range(3):
        for x in range(3):out&=p[y:y+h,x:x+w]
    return out

def transform(rgb,alpha,longside,rotation):
    from PIL import Image
    assert rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3
    assert alpha.dtype==np.bool_ and rgb.shape[:2]==alpha.shape
    assert longside in LONGSIDES and rotation in ROTATIONS
    h,w=alpha.shape;scale=longside/max(h,w)
    hh=max(1,math.floor(h*scale+.5));ww=max(1,math.floor(w*scale+.5))
    r=np.array(Image.fromarray(rgb).resize((ww,hh),Image.Resampling.BILINEAR))
    a=np.array(Image.fromarray(alpha.astype(np.uint8)*255).resize((ww,hh),Image.Resampling.NEAREST))>0
    a=erode1(a)
    return np.ascontiguousarray(np.rot90(r,rotation//90)),np.ascontiguousarray(np.rot90(a,rotation//90))

def correlation_counts(image,kernel):
    """All valid top-left integer overlaps, using float64 FFT then exact integer round.

    Inputs are Boolean; independently tested against scalar sums. No scipy needed.
    """
    assert image.dtype==kernel.dtype==np.bool_ and image.ndim==kernel.ndim==2
    H,W=image.shape;h,w=kernel.shape
    if h>H or w>W:return np.empty((0,0),np.int64)
    shape=tuple(1<<(n-1).bit_length() for n in (H+h-1,W+w-1))
    conv=np.fft.irfft2(np.fft.rfft2(image.astype(np.float64),s=shape)*
        np.fft.rfft2(kernel[::-1,::-1].astype(np.float64),s=shape),s=shape)
    view=conv[h-1:H,w-1:W];rounded=np.rint(view)
    assert np.max(np.abs(view-rounded),initial=0)<1e-5,'FFT integer-count precision'
    return rounded.astype(np.int64)

def feasible_positions(alpha,valid,feature):
    assert alpha.dtype==valid.dtype==feature.dtype==np.bool_ and valid.shape==feature.shape
    mass=int(alpha.sum())
    if mass==0:return [],[]
    legal=correlation_counts(valid,alpha)==mass
    overlaps=correlation_counts(feature,alpha)
    cover=np.argwhere(legal&(overlaps==int(feature.sum())))
    sham=np.argwhere(legal&(overlaps==0))
    # Return x,y with full transformed rectangular footprint inside original crop.
    return [(int(x),int(y)) for y,x in cover],[(int(x),int(y)) for y,x in sham]

def ordered_transforms(row_id,donor_id):
    return sorted(((n,r) for n in LONGSIDES for r in ROTATIONS),
        key=lambda z:sha(f'V207_RAW96_TRANSFORM_V1|{row_id}|{donor_id}|{z[0]}|{z[1]}'))

def choose_position(positions,row_id,donor_id,longside,rotation,kind):
    assert kind in ('COVER','SHAM') and positions
    return min(positions,key=lambda p:sha(f'V207_RAW96_PLACE_V1|{row_id}|{donor_id}|{longside}|{rotation}|{kind}|{p[0]}|{p[1]}'))

def paste(before,texture,alpha,origin):
    assert before.dtype==texture.dtype==np.uint8 and before.ndim==texture.ndim==3
    h,w=alpha.shape;x,y=origin;H,W=before.shape[:2]
    assert 0<=x and 0<=y and x+w<=W and y+h<=H,'No clipping'
    out=before.copy();mask=np.zeros((H,W),bool)
    out[y:y+h,x:x+w][alpha]=texture[alpha];mask[y:y+h,x:x+w]=alpha
    return out,mask

def verify_actual(before,cover,sham,cover_alpha,sham_alpha,valid,case):
    feature,status=obligation(case);assert feature is not None,status
    assert before.shape==cover.shape==sham.shape==(96,96,3)
    assert before.dtype==cover.dtype==sham.dtype==np.uint8
    assert cover_alpha.dtype==sham_alpha.dtype==valid.dtype==np.bool_
    assert cover_alpha.shape==sham_alpha.shape==valid.shape==(96,96)
    assert np.all(cover_alpha[feature]) and not np.any(sham_alpha[feature])
    assert not np.any(cover_alpha&~valid) and not np.any(sham_alpha&~valid)
    assert int(cover_alpha.sum())==int(sham_alpha.sum())>0
    assert np.array_equal(cover[~cover_alpha],before[~cover_alpha])
    assert np.array_equal(sham[~sham_alpha],before[~sham_alpha])
    assert np.array_equal(sham[feature],before[feature])
    changed_cover=np.any(cover!=before,axis=2);changed_sham=np.any(sham!=before,axis=2)
    fractions={}
    for k in ('A','C'):
        f=rectangle(case['ROI_'+k]);n=int((changed_cover&f).sum());area=int(f.sum())
        fractions[k]=dict(changed_pixels=n,area=area,changed_fraction=n/area)
    return dict(actual_change_PASS=all(fractions[k]['changed_pixels']>0 for k in fractions),
        feature_changes=fractions,cover_changed_RGB_pixels=int(changed_cover.sum()),
        sham_changed_RGB_pixels=int(changed_sham.sum()),pasted_alpha_pixels=int(cover_alpha.sum()),
        obligations_opaque=True,SHAM_feature_RGB_unchanged=True,outside_alpha_exact=True,
        matched_positive_alpha_mass=True,training_admission=False,new_native_E_label=False)

def validate_gate(entry,expected):
    # Pure, fail-closed; caller must finish this before creating any payload reader.
    assert entry.get('approved')is True and entry.get('actual_payload_access_authorized')is True
    assert entry.get('no_new_E_or_training_admission')is True and entry.get('attempt_count')==80
    for name,binding in expected.items():assert entry.get(name)==binding,name
    return True
