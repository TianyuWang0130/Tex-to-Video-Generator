# Tex-to-Video-Generator
AnimateDiffPipeline

A comprehensive pipeline for text-to-video generation, fine-tuning, deployment, and evaluation using AnimateDiff-Lightning, built on top of HuggingFace's Diffusers library.

Features

Generate animations from text prompts using AnimateDiff

Fine-tune MotionAdapter with the YouCook2 dataset

Deploy a Gradio-based web app for interactive generation

Evaluate animation quality using LPIPS, SSIM, PSNR, FVD, and CLIP score

Installation

pip install --upgrade transformers accelerate diffusers imageio-ffmpeg safetensors gradio lpips piq

Basic Usage

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch

step = 4
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"

device = "cuda"
dtype = torch.float16

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

output = pipe(prompt="crack two eggs into a bowl", guidance_scale=1.0, num_inference_steps=step)
export_to_gif(output.frames[0], "animation.gif")

Fine-tuning

Fine-tune the MotionAdapter using the YouCook2 dataset.

Dataset

class YouCook2Dataset(Dataset):
    # Parses the YouCook2 annotations and loads frame-level data
    # Selects the first 8 frames per caption segment

Training Loop

for epoch in range(num_epochs):
    for batch in dataloader:
        frames, captions = batch
        # Preprocess, forward pass, compute SmoothL1 loss, backpropagate

Model is saved to:

/content/drive/.../motion_adapter_finetuned_4.6.pth

Gradio Web App

Launch a web interface to generate animations interactively.

def generate_animation(prompt):
    # Generates GIF using the fine-tuned pipeline

interface = gr.Interface(fn=generate_animation, inputs=gr.Textbox(), outputs=gr.Image(type="filepath"))
interface.launch(share=True)

Evaluation

Metrics Used:

LPIPS (Perceptual similarity)

SSIM (Structural Similarity Index)

PSNR (Peak Signal-to-Noise Ratio)

FVD (Fréchet Video Distance - simplified)

CLIP Score (Text-image alignment)

# Sample:
clip_score = compute_clip_score(prompt, Image.fromarray(generated_frame))
fvd_score = fvd_lightweight(real_frames, generated_frames)

Results can be stored in a pandas DataFrame and printed as a table.

Acknowledgements

ByteDance/AnimateDiff-Lightning

HuggingFace Diffusers

YouCook2 Dataset

License

This repository is intended for academic and research purposes only. Please refer to the respective licenses of the models and datasets used.

Contact

For questions or contributions, please open an issue or reach out via GitHub Discussions.
