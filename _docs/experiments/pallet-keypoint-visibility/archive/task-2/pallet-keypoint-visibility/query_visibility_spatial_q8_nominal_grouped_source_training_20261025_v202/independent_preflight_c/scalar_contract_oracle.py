"""Independent invented scalar oracle; no source file/model reads."""
import math

def scalar_context(original_q8,original_available8,active_q,active_L,native):
    if len(original_q8)!=8 or len(original_available8)!=8 or native not in range(8):
        raise ValueError("exact native8 shape")
    if len(active_q)!=2 or not all(math.isfinite(float(x)) for x in active_q) or not math.isfinite(float(active_L)) or active_L<=0:
        raise ValueError("finite positive active crop grid")
    xy=[];mask=[]
    for j,(q,a) in enumerate(zip(original_q8,original_available8)):
        if type(a) is not bool or len(q)!=2:
            raise ValueError("original bool availability and xy2")
        if a and not all(math.isfinite(float(v)) for v in q):
            raise ValueError("available point nonfinite")
        keep=a and j!=native
        mask.append(float(keep))
        xy.extend([max(-8.,min(8.,(float(q[d])-float(active_q[d]))/float(active_L))) if keep else 0. for d in range(2)])
    return dict(xy16=xy,mask8=mask,native8=[float(i==native) for i in range(8)])

def nominal_source_eligible(family,state_index=None,state_name=None):
    if family=="SELF_SCENE15":
        if type(state_index)is not int or not (0<=state_index<15):
            raise ValueError("exact SELF state index")
        if (state_index==0)!=(state_name=="NOMINAL"):
            raise ValueError("nominal state name/index drift")
        return state_index==0
    if family=="BEFORE_ANCHOR4":
        return True
    raise ValueError("synthetic source family required")
