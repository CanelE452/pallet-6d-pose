"""Point-specific attention to incident-role Hough support and endpoint cells."""
import torch,math
from torch import nn
import torch.nn.functional as F
from model32 import VisibleGeometry,objective
from plpose_local.geometry import EDGES

class Correspondence(VisibleGeometry):
    def __init__(self,seed):
        super().__init__(seed)
        torch.manual_seed(33000+seed)
        self.role_embed=nn.Embedding(12,8)
        self.query_encode=nn.Sequential(nn.Linear(288+2+2+8+2,64),nn.SiLU(),nn.Linear(64,32))
        self.line_token=nn.Sequential(nn.Linear(32+8+8,64),nn.SiLU(),nn.Linear(64,32))
        self.end_token=nn.Sequential(nn.Linear(32+8+3,64),nn.SiLU(),nn.Linear(64,32))
        self.line_relation=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,1))
        self.end_relation=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,1))
        incident=torch.zeros(8,12,dtype=torch.bool)
        for r,(a,b) in enumerate(EDGES):incident[a,r]=True;incident[b,r]=True
        self.register_buffer('incident',incident)
        y,x=torch.meshgrid(torch.linspace(-1,1,24),torch.linspace(-1,1,24),indexing='ij');self.register_buffer('xy24',torch.stack((x.flatten(),y.flatten()),-1))
        y,x=torch.meshgrid(torch.linspace(-1,1,48),torch.linspace(-1,1,48),indexing='ij');self.register_buffer('xy48',torch.stack((x,y),0))

    def evidence(self,z,maps,bb,hw,blind=False):
        B=len(bb);wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);diag=wh.norm(dim=-1);half=wh/2;center=(bb[:,:2]+bb[:,2:])/2
        hough=self.hough(maps[:,:12]);score,index=hough.flatten(-2).topk(4,-1)
        # Four strongest bins per role, possibly neighbouring modes; not NMS.
        index=index.flatten(1);score=score.flatten(1);role=torch.arange(12,device=z.device).repeat_interleave(4)
        theta=(index//48)*math.pi/24;rho=(index%48)*2*math.sqrt(2)/47-math.sqrt(2)
        normal=torch.stack((theta.cos(),theta.sin()),-1)/half[:,None];length=normal.norm(dim=-1);normal=normal/length[...,None];offset=rho/length/diag[:,None]
        xy=self.xy24[None]*half[:,None]/diag[:,None,None];pixel=xy*diag[:,None,None]+center[:,None]
        observed=((pixel>=0)&(pixel<hw.flip(-1)[:,None])).all(-1)
        weight=F.interpolate(maps[:,:12],size=(24,24),mode='bilinear',align_corners=True).flatten(-2)[:,role]
        distance=torch.einsum('bnd,bkd->bnk',normal,xy)-offset[...,None]
        ridge=torch.exp(-.5*(distance/.025).square());support=weight*ridge*observed[:,None];mass=support.sum(-1);weights=support/mass[...,None].clamp_min(1e-6)
        loc=torch.einsum('bnk,bkd->bnd',weights,xy);tangent=torch.stack((-normal[...,1],normal[...,0]),-1)
        along=torch.einsum('bnkd,bnd->bnk',xy[:,None]-loc[:,:,None],tangent);spread=(weights*along.square()).sum(-1).clamp_min(1e-8).sqrt()
        visual=torch.einsum('bnk,bck->bnc',weights,F.interpolate(z,size=(24,24),mode='bilinear',align_corners=True).flatten(-2))
        geom=torch.cat((normal,offset[...,None],loc,spread[...,None],score[...,None],(mass/576)[...,None]),-1)
        if blind:geom=torch.cat((torch.zeros_like(geom[...,:6]),geom[...,6:]),-1)
        line=self.line_token(torch.cat((visual,self.role_embed(role)[None].expand(B,-1,-1),geom),-1))
        ep=maps[:,12:];xy48=self.xy48[None]*half[:,:,None,None]/diag[:,None,None,None]
        eweight=F.avg_pool2d(ep,6).flatten(-2);numer=F.avg_pool2d((ep[:,:,None]*xy48[:,None]).flatten(1,2),6).reshape(B,12,2,64)
        eloc=(numer/eweight[:,:,None].clamp_min(1e-6)).permute(0,1,3,2).flatten(1,2);escore=eweight.flatten(1)
        ev=F.avg_pool2d(z,6).flatten(-2).transpose(1,2)[:,None].expand(-1,12,-1,-1).flatten(1,2);erole=torch.arange(12,device=z.device).repeat_interleave(64)
        end=self.end_token(torch.cat((ev,self.role_embed(erole)[None].expand(B,-1,-1),torch.zeros_like(eloc) if blind else eloc,escore[...,None]),-1))
        return dict(hough=hough,line=line,end=end,normal=normal,offset=offset,loc=loc,spread=spread,line_score=score,line_role=role,end_loc=eloc,end_score=escore,end_role=erole)

    def attend(self,q,anchor,e,which,blind):
        token=e[which];role=e[which+'_role'];score=e[which+'_score'];B,Q=q.shape[:2]
        if which=='line':
            displacement=anchor[:,:,None]-e['loc'][:,None];tangent=torch.stack((-e['normal'][...,1],e['normal'][...,0]),-1)
            normal_distance=(anchor[:,:,None]*e['normal'][:,None]).sum(-1)-e['offset'][:,None]
            along=(displacement*tangent[:,None]).sum(-1)
            relation=torch.cat((normal_distance[...,None],along[...,None],e['spread'][:,None,:,None].expand(-1,Q,-1,-1),displacement),-1)
        else:
            d=anchor[:,:,None]-e['end_loc'][:,None];relation=torch.cat((d,d.norm(dim=-1,keepdim=True),torch.zeros_like(d)),-1)
        if blind:relation=torch.zeros_like(relation)
        logits=torch.einsum('bqd,bnd->bqn',q,token)/math.sqrt(32)+getattr(self,which+'_relation')(relation).squeeze(-1)+score.clamp_min(1e-5).log()[:,None]-math.log(len(role))
        allowed=self.incident[:,role].repeat_interleave(2,0);logits=logits.masked_fill(~allowed[None],-1e4)
        attention=torch.cat((logits,torch.zeros(B,Q,1,device=q.device)),dim=-1).softmax(-1)
        return torch.einsum('bqn,bnd->bqd',attention[...,:-1],token),attention

    def forward(self,O,S,arm='query',intervention=None,donor_maps=None):
        if arm in ('pooled','point_aux'):
            return super().forward(O,S,'visible_hough' if arm=='pooled' else 'point')
        B=len(O['box']);bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);diag=wh.norm(dim=-1);center=(bb[:,:2]+bb[:,2:])/2
        z=F.interpolate(self.project(O['features'])*O['content'],size=(48,48),mode='bilinear',align_corners=True);pix=self.pixel(F.interpolate(O['edge'],size=(48,48),mode='bilinear',align_corners=True));z=self.trunk(torch.cat((z,pix),1));logits=self.roles(z);maps=logits.sigmoid()
        if donor_maps is not None:maps=donor_maps
        def patch(points):
            axis=points.new_tensor([-.03,0,.03]);y,x=torch.meshgrid(axis,axis,indexing='ij');off=torch.stack((x.flatten(),y.flatten()),-1);positions=points[:,:,None]+diag[:,None,None,None]*off;grid=2*(positions-bb[:,None,None,:2])/wh[:,None,None]-1;valid=(grid.abs()<=1).all(-1)&((positions>=0)&(positions<O['image_hw'].flip(-1)[:,None,None])).all(-1)
            return (F.grid_sample(z,grid,align_corners=True,padding_mode='zeros').permute(0,2,3,1)*valid[...,None]).flatten(-2)
        base,strong=S['base'],S['strong'];pb,ps=patch(base),patch(strong);ident=self.ident(torch.arange(8,device=z.device))[None].expand(B,-1,-1);coords=torch.cat((2*(base-bb[:,None,:2])/wh[:,None]-1,2*(strong-bb[:,None,:2])/wh[:,None]-1,(strong-base)/diag[:,None,None]),-1)
        anchors=torch.stack((base,strong),2).flatten(1,2);an=(anchors-center[:,None])/diag[:,None,None];patches=torch.stack((pb,ps),2).flatten(1,2);parent=torch.eye(2,device=z.device)[None,None].expand(B,8,-1,-1).flatten(1,2)
        query=self.query_encode(torch.cat((patches,an,((strong-base)/diag[:,None,None]).repeat_interleave(2,1),ident.repeat_interleave(2,1),parent),-1));blind=arm=='no_geometry';e=self.evidence(z,maps,bb,O['image_hw'],blind)
        line,la=self.attend(query,an,e,'line',blind);end,ea=self.attend(query,an,e,'end',blind)
        if arm=='end_only' or intervention in ('line_off','both_off'):line=torch.zeros_like(line)
        if arm=='line_only' or intervention in ('end_off','both_off'):end=torch.zeros_like(end)
        tokens=torch.cat((line.reshape(B,8,64),end.reshape(B,8,64)),-1);gl=F.adaptive_avg_pool2d(z,(4,4)).flatten(1);context=torch.cat((pb,ps,gl[:,None].expand(-1,8,-1),tokens,ident,coords),-1)
        raw=self.head(context);mix=raw[...,0].tanh();delta=raw[...,1:].tanh();delta=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1);residual=.15*diag[:,None,None]*delta;points=base+mix[...,None]*(strong-base)+residual
        return dict(points=points,mix=mix,residual=residual,role_logits=logits,role_maps=maps,hough=e['hough'],base=base,strong=strong,diag=diag,line_attention=la,end_attention=ea,line_normal=e['normal'],line_offset=e['offset'],line_support=e['loc'],line_spread=e['spread'],endpoint_locations=e['end_loc'])
