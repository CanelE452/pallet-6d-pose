"""Inert factory: same418433 parameters for both arms, original GEMM preserved."""
def expand_base(torch,features):
 T,Q,C,P=features[:,:384],features[:,384:768],features[:,1152:1536],features[:,1536:1920]
 return torch.cat((features,torch.abs(T-C),T*C,torch.abs(Q-P),Q*P),1)

def make_head(torch,parent_state):
 class SpatialStencilHead(torch.nn.Module):
  def __init__(self):
   super().__init__();self.base_affine=torch.nn.Linear(3464,64);self.extra_affine=torch.nn.Linear(3072,64,bias=False);self.relu=torch.nn.ReLU();self.output=torch.nn.Linear(64,1)
   with torch.no_grad():
    self.base_affine.weight.copy_(parent_state['head.0.weight']);self.base_affine.bias.copy_(parent_state['head.0.bias']);self.output.weight.copy_(parent_state['head.2.weight']);self.output.bias.copy_(parent_state['head.2.bias']);self.extra_affine.weight.zero_()
   assert sum(p.numel()for p in self.parameters())==418433 and all(p.requires_grad for p in self.parameters())
   assert not self.extra_affine.weight.count_nonzero()
  def forward(self,base1928,extra3072):
   if base1928.dtype!=torch.float32 or extra3072.dtype!=torch.float32 or base1928.ndim!=2 or extra3072.shape!=(len(base1928),3072)or base1928.shape[1]!=1928:raise ValueError('Only FP32 base1928+stencil3072 image features')
   # Keep the old3464 GEMM and bias untouched; a zero extra affine gives exact0.
   hidden=self.base_affine(expand_base(torch,base1928))+self.extra_affine(extra3072)
   return self.output(self.relu(hidden)).squeeze(-1)
 return SpatialStencilHead()
