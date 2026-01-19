import os, json
import torch

P = os.path.join(os.getcwd(), 'data', 'eeg_55_95_std.pth')

def quick_info(o):
    t = type(o)
    if torch.is_tensor(o):
        return {'type':'tensor','shape':tuple(o.size()),'dtype':str(o.dtype),'numel':int(o.numel())}
    if isinstance(o, dict):
        info={'type':'dict','len':len(o),'keys_sample':list(o.keys())[:200]}
        # inspect first few keys for tensor shapes
        kd={}
        for k in list(o.keys())[:50]:
            v=o[k]
            if torch.is_tensor(v):
                kd[str(k)]={'type':'tensor','shape':tuple(v.size()),'dtype':str(v.dtype),'numel':int(v.numel())}
            else:
                kd[str(k)]={'type':type(v).__name__}
        info['key_details_sample']=kd
        return info
    if isinstance(o, (list,tuple)):
        return {'type':'list' if isinstance(o,list) else 'tuple','len':len(o),'sample_types':[type(x).__name__ for x in o[:10]]}
    return {'type':t.__name__,'repr':repr(o)[:400]}

try:
    obj = torch.load(P, map_location='cpu')
except Exception as e:
    print(json.dumps({'error': str(e)}))
    raise

out = {'path': P, 'top_type': type(obj).__name__}

if isinstance(obj, (list,tuple)):
    out['len'] = len(obj)
    out['items'] = [quick_info(x) for x in obj[:50]]
elif isinstance(obj, dict):
    out.update(quick_info(obj))
else:
    out['info'] = quick_info(obj)

print(json.dumps(out, indent=2))
