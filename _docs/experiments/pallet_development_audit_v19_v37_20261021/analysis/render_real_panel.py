"""Render existing stored poses on an authorized crop; inputs are not bundled."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
def render(gray,points,edges,metrics,title,output):
 """points use this identical crop's pixels. No pose/image warping or rephasing."""
 labels=['Image crop','Initial N0','Original OCPR','Geometric reference'];styles=[None,('#D55E00','--'),('#0072B2','-'),('#eeeeee',':')];methods=[None,'N0','ORIGINAL_OCPR','REFERENCE'];h,w=gray.shape;fig,axes=plt.subplots(1,4,figsize=(10.8,max(2.8,2.4*h/w+1.2)))
 for ax,label,style,method in zip(axes,labels,styles,methods):
  ax.imshow(gray,cmap='gray',vmin=0,vmax=255,interpolation='none');ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5);ax.set_aspect('equal');ax.axis('off')
  if method:
   q=points[method]
   for a,b in edges:ax.plot(q[[a,b],0],q[[a,b],1],color=style[0],ls=style[1],lw=1.0)
  if method in metrics:
   m=metrics[method];label+=f"\nT={m['translation_cm']:.2f} cm, R={m['rotation_deg']:.2f} deg"
  elif method=='REFERENCE':label+='\nReconstructed; not physical GT'
  ax.set_title(label,fontsize=8)
 fig.suptitle(title,fontsize=10);fig.text(.5,.02,'Historical real development example; original OCPR seed 20261018. Independent visibility/physical-owner labels unavailable.',ha='center',fontsize=7);fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(output,dpi=160);plt.close(fig)
 # Fresh lossless encoding retains pixels and strips inherited EXIF/text metadata.
 image=np.array(Image.open(output).convert('RGB'));Image.fromarray(image).save(output);check=Image.open(output);assert not check.getexif()and not check.info and np.array_equal(np.array(check),image)
