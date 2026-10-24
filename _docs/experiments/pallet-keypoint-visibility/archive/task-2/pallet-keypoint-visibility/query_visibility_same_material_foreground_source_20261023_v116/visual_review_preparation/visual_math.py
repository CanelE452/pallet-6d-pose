"""Inert display math for fixed V116 source cases; no source I/O or models."""
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def select_attempts(attempts):
    if len(attempts) != 448 or [r['attempt_index'] for r in attempts] != list(range(448)):
        raise ValueError('Retain all448 attempts in canonical order')
    if any(type(r['pair_ready']) is not bool for r in attempts):
        raise ValueError('Explicit pair-ready bool')
    accepted = [r['attempt_index'] for r in attempts if r['pair_ready']][:3]
    failed = [r['attempt_index'] for r in attempts if not r['pair_ready']][:3]
    return dict(accepted=accepted, failed=failed, selected_attempt_indices=sorted(accepted+failed),
                missing_groups=[dict(group=k, requested=3, present=len(v), missing=3-len(v))
                                for k, v in [('accepted', accepted), ('failed', failed)] if len(v) < 3],
                policy='First3accepted and first3failed by originalattempt_index; no replacement or other selection')


# Exact AST of frozen V90 RGB crop; caller supplies unchanged original q/L.
def crop(rgb,q,L,n):
    h,w=rgb.shape[:2];u=(np.arange(n,dtype=np.float64)+.5)/n-.5
    xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L);ix=np.floor(xx).astype(int);iy=np.floor(yy).astype(int);dx=xx-ix;dy=yy-iy
    acc=np.zeros((n,n,3),np.float64)
    for ax,ay,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        x,y=ix+ax,iy+ay;inside=(x>=0)&(x<w)&(y>=0)&(y<h)
        acc[inside]+=rgb[y[inside],x[inside]].astype(np.float64)*weight[inside,None]
    valid=(xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5)
    out=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8);out[~valid]=0
    return out.transpose(2,0,1).copy(),valid[None].copy()


def protected_crop(mask, q, L, n=96):
    h,w=mask.shape; u=(np.arange(n,dtype=np.float64)+.5)/n-.5
    x,y=np.meshgrid(q[0]+u*L,q[1]+u*L)
    ix,iy=np.floor(x+.5).astype(int),np.floor(y+.5).astype(int)
    good=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
    out=np.zeros((n,n),bool);out[good]=mask[iy[good],ix[good]]
    return out


def marked(rgb, support, q_xy=None, crop_rectangle=None):
    """Separate overlay copy; never mutate source pixels or support."""
    a=np.array(rgb,dtype=np.uint8,copy=True)
    if support is not None:
        a[support]=np.floor(.55*a[support]+.45*np.array([255,0,160])+.5).astype(np.uint8)
    im=Image.fromarray(a);d=ImageDraw.Draw(im)
    if crop_rectangle is not None:
        d.rectangle(tuple(crop_rectangle),outline=(255,210,0),width=2)
    if q_xy is not None:
        x,y=map(float,q_xy);d.line((x-5,y,x+5,y),fill=(255,220,0),width=2);d.line((x,y-5,x,y+5),fill=(255,220,0),width=2)
    return im


def font(size=18):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def fit(im, box, background=(246,248,250)):
    out=Image.new('RGB',box,background);a=im.convert('RGB').copy()
    factor=min(box[0]/a.width,box[1]/a.height)
    a=a.resize((max(1,round(a.width*factor)),max(1,round(a.height*factor))),Image.Resampling.BILINEAR)
    out.paste(a,((box[0]-a.width)//2,(box[1]-a.height)//2));return out


def donor_display(rgb, alpha, side=200):
    """Display-only tight native-alpha extent, never an altered training input."""
    ys,xs=np.nonzero(alpha)
    if len(xs)==0:
        return fit(Image.fromarray(rgb),(side,side)),None
    box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    x0,y0,x1,y1=box;a=rgb[y0:y1,x0:x1].copy();m=alpha[y0:y1,x0:x1]
    yy,xx=np.indices(m.shape);check=np.where(((xx//8+yy//8)%2)[...,None]==0,220,245).astype(np.uint8)
    check=np.repeat(check,3,axis=2);check[m]=a[m]
    return fit(Image.fromarray(check),(side,side)),box


def card(title, subtitle, raw_contexts, marked_contexts, raw_crops, marked_crops,
         donor_panel, donor_notes, footer):
    require=len(raw_contexts)==len(marked_contexts)==len(raw_crops)==len(marked_crops)==3
    if not require:raise ValueError('Exactly BEFORE/COVER/SHAM panels')
    w=1440;im=Image.new('RGB',(w,1450),'white');d=ImageDraw.Draw(im)
    d.text((18,14),title,font=font(24),fill=(15,30,45))
    for j,line in enumerate(textwrap.wrap(subtitle,150)[:3]):d.text((18,50+j*22),line,font=font(16),fill=(45,60,70))
    labels=['BEFORE','COVER','SHAM'];y=130
    for j,label in enumerate(labels):
        x=j*480+10;d.text((x,y-25),label,font=font(20),fill='black')
        im.paste(fit(raw_contexts[j],(460,220)),(x,y))
        im.paste(fit(marked_contexts[j],(460,220)),(x,y+225))
        im.paste(raw_crops[j].resize((280,280),Image.Resampling.NEAREST),(x,y+478))
        im.paste(marked_crops[j].resize((280,280),Image.Resampling.NEAREST),(x,y+763))
        d.text((x+288,y+500),'RAW 96',font=font(16),fill=(30,45,55))
        d.text((x+288,y+785),'q + support',font=font(16),fill=(30,45,55))
    # The fixed native donor is shown at bottom, with actual ownership mask alpha.
    im.paste(fit(donor_panel,(320,160)),(18,1260))
    for j,line in enumerate(textwrap.wrap(donor_notes,110)[:5]):d.text((355,1265+j*22),line,font=font(16),fill=(30,45,55))
    d.text((355,1400),footer,font=font(14),fill=(80,50,30))
    return im


def contact(cards, missing):
    if not cards:
        out=Image.new('RGB',(900,180),'white');ImageDraw.Draw(out).text((18,20),'NO SELECTED CASES: '+str(missing),font=font(18),fill='black');return out
    thumbw=720;thumbh=725;out=Image.new('RGB',(1440,70+thumbh*((len(cards)+1)//2)),(232,238,243));d=ImageDraw.Draw(out)
    d.text((15,12),'Fixed first3 accepted + first3 failed; missing groups: '+str(missing),font=font(18),fill='black')
    for j,a in enumerate(cards):out.paste(a.resize((thumbw,thumbh),Image.Resampling.LANCZOS),((j%2)*thumbw,70+(j//2)*thumbh))
    return out
