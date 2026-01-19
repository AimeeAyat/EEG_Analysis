import json
import os
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))

def main():
    clip_path = os.path.join(ROOT, 'clip_feature.pth')
    desc_path = os.path.join(ROOT, 'description.json')
    print('Loading', clip_path)
    data = torch.load(clip_path, map_location='cpu')
    print('Loaded clip_feature.pth type:', type(data))
    # If dict-like, show keys and sample shapes
    if isinstance(data, dict):
        print('Keys in clip_feature.pth:', list(data.keys())[:10])
        for k in list(data.keys())[:5]:
            try:
                v = data[k]
                print(k, '->', type(v))
                if isinstance(v, dict):
                    print('  nested keys:', list(v.keys()))
                    for nk in list(v.keys())[:5]:
                        try:
                            sv = v[nk]
                            print('   ', nk, '->', getattr(sv, 'shape', type(sv)))
                        except Exception as e:
                            print('   ', nk, '-> err', e)
            except Exception as e:
                print('Could not inspect key', k, e)
    else:
        try:
            print('Object shape:', data.shape)
        except Exception:
            print('Object repr:', repr(data)[:200])

    # Try mapping a label from description.json to clip features
    if os.path.exists(desc_path):
        desc = json.load(open(desc_path, 'r', encoding='utf-8'))
        sample_key = next(iter(desc.keys()))
        uid = desc[sample_key].get('uid')
        print('\nSample image key from description.json:', sample_key, 'uid:', uid)
        if isinstance(data, dict) and uid in data:
            print('Found clip embedding for uid; shape:', data[uid].shape)
        else:
            print('UID not present in clip features or clip features not dict keyed by uid.')

if __name__ == '__main__':
    main()
