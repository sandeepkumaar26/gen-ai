#!/usr/bin/env python3
"""
Ex. No: 8 - IMAGE GENERATION APPLICATION USING DIFFUSION MODELS

AIM
    To implement an image generation application using a pre-trained Diffusion
    Model (Stable Diffusion) that synthesises images from text prompts.

REQUIREMENTS
    pip install diffusers transformers torch accelerate
    A GPU-enabled environment is recommended (Google Colab with GPU runtime).
"""

import torch
from diffusers import StableDiffusionPipeline


def main():
    # Use the GPU when available; fall back to CPU (slower) otherwise.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    prompt = "A futuristic city skyline at sunset, digital art, highly detailed"

    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    image.save("generated_city.png")
    print("Image generated and saved as generated_city.png")


if __name__ == "__main__":
    main()
