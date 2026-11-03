"""Restricted independent candidates with fixed or dynamic convex fusion; joint gradients."""
import math
import torch
from torch import nn
from .learned_correction import RestrictedCorrector,observed_context
from .fast_candidates import FastCandidates


class JointCorrection(nn.Module):
    def __init__(self,fixed_weights,dynamic=False):
        super().__init__();self.corrector=RestrictedCorrector(16);self.dynamic=dynamic
        self.register_buffer('fixed_weights',torch.tensor(fixed_weights,dtype=torch.float32)[None,:,None,None])
        if dynamic:
            self.gate=nn.Sequential(nn.Conv2d(19,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,3,1))
            nn.init.zeros_(self.gate[-1].weight)
            with torch.no_grad():self.gate[-1].bias.copy_(torch.tensor([.9,.05,.05]).log())

    def forward(self,features):
        rgb=features[:,:3];priors=features[:,3:6];context=features[:,9:11]
        z=self.corrector.trunk(rgb);v=self.corrector.head(torch.cat([z,rgb,context],1))
        gain=torch.exp(math.log(2)*torch.tanh(v[:,:3]));offset=self.corrector.shift*torch.tanh(v[:,3:])
        candidates=self.corrector.transform(priors,gain,offset)
        weights=torch.softmax(self.gate(torch.cat([z,features],1)).float(),1) if self.dynamic else self.fixed_weights.expand_as(candidates)
        h=(weights*candidates).sum(1,keepdim=True)
        return h,{'candidates':candidates,'weights':weights,'gain':gain,'offset':offset}


class InferencePipeline(nn.Module):
    """Forward accepts hazy display RGB only; calibration stays fixed on train."""
    def __init__(self,network,calibration,uses_priors=True,amp=True):
        super().__init__();self.network=network;self.uses_priors=uses_priors;self.amp=amp
        if uses_priors:self.extractor=FastCandidates(calibration)

    def forward(self,rgb):
        if rgb.ndim!=4 or rgb.shape[1]!=3:raise ValueError('Expected N,3,H,W RGB')
        with torch.autocast(rgb.device.type,enabled=False):
            if self.uses_priors:
                f,_=self.extractor(rgb.float());f=torch.cat([f,observed_context(rgb.float())],1)
            else:f=rgb.float()
        with torch.autocast(rgb.device.type,dtype=torch.float16,enabled=self.amp and rgb.device.type=='cuda'):
            h,_=self.network(f)
        return h.float()
