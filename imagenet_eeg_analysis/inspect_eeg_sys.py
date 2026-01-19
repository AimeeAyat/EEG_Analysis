import os
import sys
import pickle
import json
try:
    import torch
except Exception as e:
    print(json.dumps({'error': f'torch import failed: {e}'}))
    raise

P = os.path.join(os.getcwd(), 'data', 'eeg_55_95_std.pth')
results = {'path': P, 'objects': []}

with open(P, 'rb') as f:
    i = 0
    while True:
        try:
            obj = pickle.load(f)
            i += 1
            entry = {'index': i, 'type': type(obj).__name__}
            try:
                if torch.is_tensor(obj):
                    entry.update({'tensor_shape': tuple(obj.size()), 'dtype': str(obj.dtype), 'numel': int(obj.numel())})
                elif isinstance(obj, dict):
                    entry['len'] = len(obj)
                    keys = list(obj.keys())[:200]
                    entry['keys_sample'] = [str(k) for k in keys]
                    # inspect keys for tensors
                    kd = {}
                    for k in keys:
                        try:
                            v = obj[k]
                            if torch.is_tensor(v):
                                kd[str(k)] = {'tensor_shape': tuple(v.size()), 'dtype': str(v.dtype), 'numel': int(v.numel())}
                            else:
                                kd[str(k)] = {'type': type(v).__name__}
                        except Exception as ex:
                            kd[str(k)] = {'error': str(ex)}
                    entry['key_details_sample'] = kd
                elif hasattr(obj, 'shape'):
                    try:
                        entry['shape'] = tuple(obj.shape)
                    except Exception:
                        pass
                elif isinstance(obj, (list, tuple)):
                    entry['len'] = len(obj)
                    entry['sample_types'] = [type(x).__name__ for x in obj[:10]]
                elif isinstance(obj, (int, float, str)):
                    entry['repr'] = repr(obj)[:200]
            except Exception as ex:
                entry['inspect_error'] = str(ex)
            results['objects'].append(entry)
        except EOFError:
            break
        except Exception as e:
            results.setdefault('errors', []).append({'index': i+1, 'error': str(e)})
            break

print(json.dumps(results, indent=2))
