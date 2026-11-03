"""Inference-only CUDA graph replay; public input validation and owned outputs."""
import copy
import torch
from .fast_candidates import FastCandidates
from .physical_prior import dark_channel
from .guided_filter import _mean_filter

def validate_rgb(image):
    if image.ndim!=4 or image.shape[1]!=3 or not image.numel() or not image.is_floating_point():
        raise ValueError('Expected nonempty floating N,3,H,W RGB')
    if not bool(torch.isfinite(image).all() & (image>=0).all() & (image<=1).all()):
        raise ValueError('Expected finite RGB in [0,1]')

# TrustedCandidates below copies EXACT FP32 arithmetic from FastCandidates.
# Only the value check is moved to the public replay entry, outside capture.
class TrustedCandidates(FastCandidates):
    def forward(self,image):
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



class CaptureCore(torch.nn.Module):
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
                mean=_mean_filter(gray,31)
                context=torch.cat([mean,_mean_filter((gray-mean).abs(),31)],1)
                f=torch.cat([f,context],1)
            else:f=rgb.float()
        with torch.autocast(rgb.device.type,dtype=torch.float16,enabled=self.amp and rgb.device.type=='cuda'):
            h,_=self.network(f)
        return h.float()

class CapturedInference:
    """Fixed CUDA device/shape/dtype, no labels; not thread-safe, one caller."""
    def __init__(self,pipeline,example):
        validate_rgb(example)
        if example.device.type!='cuda':raise ValueError('CUDA capture requires a CUDA input')
        self.pipeline=copy.deepcopy(pipeline).eval()
        if self.pipeline.uses_priors:
            old=self.pipeline.extractor
            ext=TrustedCandidates(old.calibration).to(example.device)
            ext.load_state_dict(old.state_dict());self.pipeline.extractor=ext
        self.pipeline=CaptureCore(self.pipeline,example.device).eval()
        self.static=example.detach().clone()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(),torch.cuda.stream(stream):
            for _ in range(4):self.pipeline(self.static)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(self.graph):
            self.output=self.pipeline(self.static)
    def __call__(self,image):
        validate_rgb(image)
        if image.shape!=self.static.shape or image.dtype!=self.static.dtype or image.device!=self.static.device:
            raise ValueError('Capture shape/device/dtype mismatch; build a separate runner')
        with torch.no_grad():
            self.static.copy_(image);self.graph.replay()
            return self.output.clone()  # callers retain stable independent outputs
