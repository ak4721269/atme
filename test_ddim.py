import os
from PIL import Image
from torchvision import transforms
from diffusers import DDIMPipeline
from options.test_options import TestOptions
from util import html
from util.visualizer import save_images

import torch
from tqdm import tqdm

def load_image(image_path, size=512):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return transform(image).unsqueeze(0)

def decode_and_save(image_tensor, path):
    image_tensor = (image_tensor.squeeze().clamp(-1, 1) + 1.0) / 2.0
    image = transforms.ToPILImage()(image_tensor.cpu())
    image.save(path)

if __name__ == '__main__':
    opt = TestOptions().parse()

    # Load DDIM pipeline from diffusers
    model_id = "stabilityai/stable-diffusion-2-1"  # or your own fine-tuned model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = DDIMPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)

    # Set DDIM-specific settings
    pipeline.set_progress_bar_config(disable=False)
    ddim_steps = getattr(opt, 'ddim_steps', 50)
    guidance_scale = getattr(opt, 'guidance_scale', 7.5)

    # Prepare input and output
    input_dir = os.path.join(opt.dataroot, "testA")
    output_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.epoch}_ddim")
    os.makedirs(output_dir, exist_ok=True)

    # Create HTML page
    webpage = html.HTML(output_dir, f"DDIM Results: {opt.name}, Epoch {opt.epoch}")
    image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))])

    for i, path in enumerate(tqdm(image_paths)):
        image_name = os.path.basename(path)
        init_image = Image.open(path).convert("RGB").resize((512, 512))

        # Prompt can be dynamic or fixed (customize as needed)
        prompt = opt.prompt if hasattr(opt, 'prompt') else "translate this image to domain B"

        result = pipeline(prompt=prompt, image=init_image, strength=0.75, num_inference_steps=ddim_steps, guidance_scale=guidance_scale).images[0]
        save_path = os.path.join(output_dir, f"fake_B_{i:04d}.png")
        result.save(save_path)

        # Save to HTML
        visuals = {
            "input": init_image,
            "fake_B": result
        }
        save_images(webpage, visuals, [path], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()
    print(f"Inference complete. Results saved to: {output_dir}")
