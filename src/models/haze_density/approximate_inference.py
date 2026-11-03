"""Experimental approximate box means; distinct from exact captured inference."""
import copy
import torch
import torch.nn.functional as F
from .captured_inference import CapturedInference,CaptureCore,TrustedCandidates,validate_rgb
from .physical_prior import dark_channel

def separable_mean(x,kernel_size):
    ref=x[:,:,:1,:1];p=kernel_size//2
    v=F.avg_pool2d(x-ref,(1,kernel_size),stride=1,padding=(0,p),count_include_pad=False)
    return F.avg_pool2d(v,(kernel_size,1),stride=1,padding=(p,0),count_include_pad=False)+ref

class SeparableCandidates(TrustedCandidates):
    def forward(self,image):
        with torch.autocast(image.device.type,enabled=False):
            image=image.float();cfg=self.calibration['prior'];gray=(image*self.rgb_weights).sum(1,keepdim=True)
            contrast=(gray-separable_mean(gray,cfg['window_size'])).abs()
            color=(image[:,:1]-image[:,1:2]).abs()+(image[:,:1]-image[:,2:3]).abs()
            raw=torch.cat([dark_channel(image,cfg['window_size']),-contrast,-color],1)
            cues=((raw-self.low)/(self.high-self.low).clamp_min(1e-8)).clamp(0,1)
            src=cues.pow(1.5);kernel=2*cfg['guided_radius']+1
            stats=separable_mean(torch.cat([gray,gray*gray,src,gray*src],1),kernel)
            mean=stats[:,:1];var=(stats[:,1:2]-mean*mean).clamp_min(0)
            a=(stats[:,5:8]-mean*stats[:,2:5])/(var+max(cfg['guided_eps'],torch.finfo(var.dtype).tiny))
            b=stats[:,2:5]-a*mean
            ab=separable_mean(torch.cat([a,b],1),kernel)
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



class SeparableCore(torch.nn.Module):
    """Same arithmetic; hoist context RGB constants out of captured execution."""
    def __init__(self,pipeline,device):
        super().__init__();self.network=pipeline.network;self.uses_priors=pipeline.uses_priors;self.amp=pipeline.amp
        if self.uses_priors:self.extractor=pipeline.extractor
        self.register_buffer('context_weights',torch.tensor([.299,.587,.114],device=device)[None,:,None,None])
    def forward(self,rgb):
        with torch.autocast(rgb.device.type,enabled=False):
            if self.uses_priors:
                f,_=self.extractor(rgb.float())
                gray=(rgb.float()*self.context_weights).sum(1,keepdim=True)
                mean=separable_mean(gray,31)
                context=torch.cat([mean,separable_mean((gray-mean).abs(),31)],1)
                f=torch.cat([f,context],1)
            else:f=rgb.float()
        with torch.autocast(rgb.device.type,dtype=torch.float16,enabled=self.amp and rgb.device.type=='cuda'):
            h,_=self.network(f)
        return h.float()



class ApproxCapturedInference(CapturedInference):
    """Different floating-point accumulation; requires a quality audit."""
    def __init__(self,pipeline,example):
        validate_rgb(example)
        if example.device.type!='cuda':raise ValueError('CUDA capture requires CUDA')
        self.pipeline=SeparableCore(copy.deepcopy(pipeline).eval(),example.device).eval()
        if self.pipeline.uses_priors:
            old=self.pipeline.extractor
            ext=SeparableCandidates(old.calibration).to(example.device);ext.load_state_dict(old.state_dict());self.pipeline.extractor=ext
        self.static=example.detach().clone();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(),torch.cuda.stream(stream):
            for _ in range(4):self.pipeline(self.static)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(self.graph):self.output=self.pipeline(self.static)
