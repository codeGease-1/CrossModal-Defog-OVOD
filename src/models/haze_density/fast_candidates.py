"""Same FP32 cue/calibration formulas, shared filter statistics and cached tables."""
import copy
import torch
from torch import nn
from .physical_prior import dark_channel
from .guided_filter import _mean_filter


class FastCandidates(nn.Module):
    def __init__(self, calibration):
        super().__init__();self.calibration=copy.deepcopy(calibration)
        self.register_buffer('rgb_weights',torch.tensor([.299,.587,.114])[None,:,None,None])
        self.register_buffer('low',torch.tensor([p[0] for p in calibration['limits']])[None,:,None,None])
        self.register_buffer('high',torch.tensor([p[1] for p in calibration['limits']])[None,:,None,None])
        tables=calibration['candidate_tables'];length=max(2,max(len(t['x']) for t in tables))
        xs=torch.full((3,length),float('inf'));ys=torch.empty(3,length);counts=[]
        for k,t in enumerate(tables):
            n=len(t['x']);assert n>0
            xs[k,:n]=torch.tensor(t['x']);ys[k,:n]=torch.tensor(t['y']);ys[k,n:]=t['y'][-1];counts.append(n)
        self.register_buffer('table_x',xs);self.register_buffer('table_y',ys)
        self.register_buffer('max_index',torch.tensor([max(1,n-1) for n in counts])[:,None])
        self.register_buffer('single',torch.tensor([n==1 for n in counts])[:,None])

    def forward(self,image):
        if image.ndim!=4 or image.shape[1]!=3 or not image.numel() or not image.is_floating_point():
            raise ValueError('Expected nonempty floating N,3,H,W RGB')
        if not bool(torch.isfinite(image).all() & (image>=0).all() & (image<=1).all()):
            raise ValueError('Expected finite display RGB in [0,1]')
        with torch.autocast(image.device.type,enabled=False):
            image=image.float();cfg=self.calibration['prior'];gray=(image*self.rgb_weights).sum(1,keepdim=True)
            contrast=(gray-_mean_filter(gray,cfg['window_size'])).abs()
            color=(image[:,:1]-image[:,1:2]).abs()+(image[:,:1]-image[:,2:3]).abs()
            raw=torch.cat([dark_channel(image,cfg['window_size']),-contrast,-color],1)
            cues=((raw-self.low)/(self.high-self.low).clamp_min(1e-8)).clamp(0,1)
            src=cues.pow(1.5);kernel=2*cfg['guided_radius']+1
            stats=_mean_filter(torch.cat([gray,gray*gray,src,gray*src],1),kernel)
            mean=stats[:,:1];var=(stats[:,1:2]-mean*mean).clamp_min(0)
            a=(stats[:,5:8]-mean*stats[:,2:5])/(var+max(cfg['guided_eps'],torch.finfo(var.dtype).tiny))
            b=stats[:,2:5]-a*mean
            ab=_mean_filter(torch.cat([a,b],1),kernel)
            scores=(ab[:,:3]*gray+ab[:,3:]).clamp(0,1)
            values=scores.permute(1,0,2,3).reshape(3,-1).contiguous()
            index=torch.minimum(torch.searchsorted(self.table_x,values).clamp_min(1),self.max_index)
            x0=self.table_x.gather(1,index-1);x1=self.table_x.gather(1,index)
            y0=self.table_y.gather(1,index-1);y1=self.table_y.gather(1,index)
            frac=((values-x0)/(x1-x0).clamp_min(1e-12)).clamp(0,1)
            mapped=torch.where(self.single,self.table_y[:,:1],y0+frac*(y1-y0))
            candidates=mapped.reshape(3,image.shape[0],*image.shape[-2:]).permute(1,0,2,3).contiguous()
            standard=(raw-self.low)/(self.high-self.low).clamp_min(1e-8)
            return torch.cat([image,candidates,standard],1),candidates
