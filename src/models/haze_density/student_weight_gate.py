"""Research-only convex gates; original density student and weights stay intact.

StudentWeightGate reuses Encoder/MultiScale/Fusion and the two feature stages
of the original decoder architecture. It has no density head or output IN.
It is trained from scratch to avoid unverified provenance of old checkpoints.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .encoder import Encoder
from .multiscale import ParallelMultiScaleFeatureExtractor
from .fusion import FusionModule


class TemperatureGate(nn.Module):
    def __init__(self, temperature=2.):
        super().__init__();self.temperature=temperature
        self.gate=nn.Sequential(nn.Conv2d(9,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,3,1))
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():self.gate[-1].bias.copy_(temperature*torch.tensor([.5,.3,.2]).log())

    def logits(self, features):return self.gate(features)

    def forward(self,features,candidates=None):
        if candidates is None:candidates=features[:,3:6]
        w=torch.softmax(self.logits(features)/self.temperature,1)
        return (w*candidates).sum(1,keepdim=True),w


class StudentWeightGate(nn.Module):
    def __init__(self,base_channels=32,temperature=2.):
        super().__init__();self.temperature=temperature
        self.encoder=Encoder(base_channels=base_channels)
        self.multiscale=ParallelMultiScaleFeatureExtractor(channels=base_channels*2,dilation_rates=[2,3,4])
        self.fusion=FusionModule(in_channels=base_channels*2,out_channels=base_channels*2)
        # Same intermediate decoder operations as the old student. The final
        # single-channel Conv/InstanceNorm/Sigmoid is intentionally absent.
        self.up=nn.Sequential(nn.ConvTranspose2d(base_channels*2,base_channels,2,stride=2,bias=False),
            nn.InstanceNorm2d(base_channels),nn.ReLU(inplace=True),
            nn.Conv2d(base_channels,base_channels//2,3,padding=1,bias=False),
            nn.InstanceNorm2d(base_channels//2),nn.ReLU(inplace=True))
        # Direct calibrated/raw features preserve absolute cues across trunk IN.
        self.head=nn.Sequential(nn.Conv2d(base_channels//2+9,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,3,1))
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():self.head[-1].bias.copy_(temperature*torch.tensor([.5,.3,.2]).log())

    def logits(self,features):
        f=self.encoder(features[:,:3]);a,b,c,_=self.multiscale(f,return_separate=True)
        f=self.up(self.fusion(a,b,c))
        if f.shape[-2:]!=features.shape[-2:]:f=F.interpolate(f,size=features.shape[-2:],mode='bilinear',align_corners=False)
        return self.head(torch.cat([f,features],1))

    def forward(self,features,candidates=None):
        if candidates is None:candidates=features[:,3:6]
        w=torch.softmax(self.logits(features)/self.temperature,1)
        return (w*candidates).sum(1,keepdim=True),w
