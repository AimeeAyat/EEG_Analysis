import os
import torch
import numpy as np
from PIL import Image

def generate_with_diffusers(embeddings, out_dir, model_name='stabilityai/stable-diffusion-2'):
    try:
        from diffusers import StableDiffusionPipeline
        from transformers import CLIPTextModel
    except Exception as e:
        raise RuntimeError('Please install diffusers and transformers to generate images: pip install diffusers transformers accelerate safetensors')

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')

    os.makedirs(out_dir, exist_ok=True)
    for i, emb in enumerate(embeddings):
        # embeddings should be a 1D numpy or torch array representing CLIP text embedding
        emb = torch.tensor(emb).unsqueeze(0).to(pipe.device)
        try:
            # attempt to use embeddings directly
            image = pipe(prompt_embeds=emb, guidance_scale=7.5, num_inference_steps=50).images[0]
        except Exception as e:
            # fallback: decode closest text by finding nearest caption in description.json (not implemented)
            print('Embedding->prompt failed:', e)
            continue
        image.save(os.path.join(out_dir, f'gen_{i}.png'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_file', required=True)
    parser.add_argument('--outdir', default='generated_images')
    parser.add_argument('--model', default='runwayml/stable-diffusion-v1-5')
    args = parser.parse_args()
    arr = np.load(args.emb_file)
    generate_with_diffusers(arr, args.outdir, model_name=args.model)
