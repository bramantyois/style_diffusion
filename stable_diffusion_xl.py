import torch
from diffusers import StableDiffusionXLPipeline


class Text2ImageModel:
    def __init__(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",     
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True)
        self.pipe.to("cuda" )
        self.pipe.enable_vae_slicing()
        
    def generate(
        self,
        prompt: str,
        guidance_scale=7.5,
        inference_steps=40,
        height=768,
        width=768,
        num_images=1,
        seed=None):
        """
        Generate images from a prompt using the Stable Diffusion XL model.
        :param prompt: The prompt to generate images from.
        :param guidance_scale: The scale of the guidance scale.
        :param inference_steps: The number of inference steps (denoising iteration) to run.
        :param height: The height of the generated image.
        :param width: The width of the generated image.
        :param num_images: The number of images to generate.

        :return: A list of PIL images.
        """
        # generate images
        generator = torch.Generator().manual_seed(seed)

        # run both experts
        images = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=inference_steps,
            generator=generator,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
        ).images

        return images
