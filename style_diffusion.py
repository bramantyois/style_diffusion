import os
from glob import glob

from tqdm.auto import tqdm

import torch
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, LMSDiscreteScheduler

from PIL import Image

YOUR_TOKEN = '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'

CACHE_DIR = os.environ.get('CACHE_DIR', '.cache')


class StyleDiffusion:
    def __init__(
            self,
            device='cuda',
            image_hw=[512, 512],
            noise_scheduler='lms',
            model='runwayml/stable-diffusion-v1-5'):

        self.device = device

        self.image_hw = image_hw

        self.vae = AutoencoderKL.from_pretrained(
            model,
            subfolder="vae",
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            use_auth_token=YOUR_TOKEN).to(self.device)

        self.text_encoder = CLIPTextModel.from_pretrained(
            model,
            subfolder="text_encoder",
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            use_auth_token=YOUR_TOKEN).to(self.device)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            cache_dir=CACHE_DIR,
            use_auth_token=YOUR_TOKEN)

        self.unet = UNet2DConditionModel.from_pretrained(
            model,
            subfolder="unet",
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            use_auth_token=YOUR_TOKEN).to(self.device)

        if noise_scheduler == 'lms':
            self.scheduler = LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000)
        elif noise_scheduler == 'ddpms':
            self.scheduler = DDPMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000)

        self.scheduler.config['steps_offset'] = 1

        self.added_artist_tokens = list()

    def update_tokens(self, learned_embedding_dir, extension='bin'):
        """append learned input embeddings into tokenizer

        :param learned_embeddings: list of file containing torch tensor embedding
        """
        files = glob(learned_embedding_dir + '/*.' + extension)

        for file in files:
            t_embed = torch.load(file)
            placeholder_token = list(t_embed)[0]
            input_embedding = t_embed[placeholder_token]

            placeholder_token = placeholder_token.replace(' ', '-')
            n_added_token = self.tokenizer.add_tokens(placeholder_token)

            if n_added_token == 0:
                print('failed to add {}'.format(placeholder_token))
                break
            else:
                print('adding {}'.format(placeholder_token))

            placeholder_token_id = self.tokenizer.convert_tokens_to_ids(placeholder_token)

            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            self.text_encoder.get_input_embeddings(
            ).weight.data[placeholder_token_id] = input_embedding

            self.added_artist_tokens.append(placeholder_token)

    def get_artist_vector(self, artist_token):
        """get artist vector learned from textual embedding

        :param artist_token: artist token, take form of <artist_name>
        :return: embedding vector
        """
        artist_id = self.tokenizer.convert_tokens_to_ids(artist_token)

        return self.text_encoder.get_input_embeddings().weight.data[artist_id].cpu().numpy()

    def tokenize_text(self, prompt):
        """tokenize text

        :param prompt: prompt strings
        :return: tokenized prompt in torch tensor
        """
        return self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt")

    def get_token_input_embeddings(self, token_ids):
        """get input token embedding. result should be added with position embedding before clip' transformer feedforward.

        :param tokens: token id
        :return: input embedding of the token
        """
        return self.text_encoder.get_input_embeddings()(token_ids)

    def get_position_embeddings(self):
        """get input position embedding. add this to token embedding befor feeding into clip's transformer

        :return: positional embedding of the tokenizer
        """
        idxs = self.text_encoder.text_model.embeddings.position_ids[:, :77]
        return self.text_encoder.text_model.embeddings.position_embedding(idxs)

    def swap_input_embedding(self, input_embeddings, pos, tensor_to_be_inserted, batch_idx=0):
        """swap an input embedding in sequence by a tensor

        :param input_embeddings: sequence of input embeddings
        :param pos: position of the tensor to be replaced
        :param tensor_to_be_inserted: tensor to be inserted into input_embeddings
        """
        input_embeddings[batch_idx, pos] = tensor_to_be_inserted
        return input_embeddings

    def encode_from_text_embedding(self, token_embeddings, position_embeddings):
        """compute text embedding from input token embeddings and position embeddings

        :param token_embeddings: input embedding
        :param position_embeddings: position embedding
        :return: encoded text
        """
        input_embeddings = (token_embeddings + position_embeddings).to(self.device)
        bsz, seq_len = input_embeddings.shape[:2]
        causal_attention_mask = self.text_encoder.text_model._build_causal_attention_mask(
            bsz, seq_len, input_embeddings.dtype).to(self.device)

        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None)

        output = encoder_outputs[0]

        output = self.text_encoder.text_model.final_layer_norm(output)

        return output

    def append_prompt_and_encode(self, prompts, tensors_to_be_inserted):
        """append prompt with a concept tensor

        :param prompts: prompt to be appended. can be a list of prompts
        :param tensor_to_be_inserted: concept tensor. can be a style matrix in the shape of (n, 768)
        """
        tokenized = self.tokenize_text(prompts).to(self.device)
        input_embeddings = self.get_token_input_embeddings(tokenized['input_ids'])
        
        mask, pos = torch.where(tokenized['input_ids'] == self.tokenizer.eos_token_id)

        mod_in_embeds = []
        for i,a in enumerate(mask.unique()):
            eos = pos[mask == a][0]
            mod_in_embeds.append(self.swap_input_embedding(input_embeddings[[i]], eos, tensors_to_be_inserted[i]))
        mod_in_embeds = torch.vstack(mod_in_embeds)
        pos_embeds = self.get_position_embeddings()

        return self.encode_from_text_embedding(mod_in_embeds, pos_embeds)

    def encode_text(self, prompt):
        """
        encode prompt
        """
        text_input = self.tokenize_text(prompt).to(self.device)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)[0]

        return text_embeddings

    def concat_unconditional_input(self, encoded_prompts):
        """concatenate random unconditional input
        """
        max_length = encoded_prompts.shape[1]

        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=max_length, return_tensors="pt").to(self.device)

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]

        # repeat along batch dim
        uncond_embeddings = uncond_embeddings.repeat(encoded_prompts.shape[0], 1, 1) 

        return torch.cat([uncond_embeddings, encoded_prompts])

    def generate_latent(self, batch_size=1):
        """
        generate random latents
        """
        latents = torch.randn((batch_size, self.unet.in_channels,
                              self.image_hw[0] // 8, self.image_hw[1] // 8))
        return latents

    def denoise_latents(self, latents, text_embeddings, inference_steps=50, guidance_scale=7.5):
        """
        denoise latents
        """
        self.scheduler.set_timesteps(inference_steps)

        code_uncond = True
        if isinstance(guidance_scale, bool):
            code_uncond = False

        timestep_tensor = self.scheduler.timesteps.to(self.device)

        latents = latents * self.scheduler.init_noise_sigma
        latents = latents.to(self.device)

        for i, t in tqdm(enumerate(timestep_tensor)):
            if code_uncond:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings).sample

            if code_uncond:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, i, latents).prev_sample
        return latents

    def decode_latents(self, latents, to_pil=True):
        """decode diffused latents

        :param latents: latens
        :param to_pil: return PIL instead of tensor, defaults to True
        :return: image
        """
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        if to_pil:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).round().astype('uint8')
            pil_images = []
            for i in range(image.shape[0]):
                pil_images.append(Image.fromarray(image[i]))
            return pil_images
        else:
            return image

    def generate(
            self,
            prompt,
            vector=None,
            guidance_scale=7.5,
            inference_steps=50,
            num_images=1,
            seed=-1,
            return_pil=True,
            return_embeddings=False,):
        """generate images given prompt and style vector(optionally)

        :param prompt: prompt to be fed. can be a list of prompts
        :param vector: style vector in np.ndarray or torch.tensor. can be an array. defaults to None
        :param guidance_scale: guidance free scale, defaults to 7.5
        :param inference_steps: diffusion steps, defaults to 50
        :param num_images: number of images to be generated, defaults to 1
        :param seed: seed, defaults to -1
        :param return_text_embeddings: return text embeddings, defaults to False
        :param return_pil: return PIL images, defaults to True
        :return: list of images and optionally a list of text embeddings
        """
        if seed >= 0:
            torch.manual_seed(seed)

        uncond_embed = False
        if isinstance(guidance_scale, float):
            uncond_embed = True
        
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector)
            if vector.ndim == 1:
                vector = [vector]
        elif isinstance(vector, list):
            vector = torch.tensor(vector)
            if vector.ndim == 1:
                vector = [vector]
        
        if isinstance(prompt, str):
            prompt = [prompt]

        images = []
        text_encodeds = []
        latents = []
        with torch.autocast(device_type=self.device):
            for _ in range(num_images):
                if vector is not None:
                    text_embedding = self.append_prompt_and_encode(prompt, vector)
                else:
                    text_embedding = self.encode_text(prompt)
                
                latent = self.generate_latent(batch_size=text_embedding.shape[0])
                
                text_encodeds.append(text_embedding.detach().cpu().numpy())
                
                if uncond_embed:
                    text_embedding = self.concat_unconditional_input(text_embedding)
                        
                denoised_latent = self.denoise_latents(
                    latents=latent,
                    text_embeddings=text_embedding,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale)
                
                latents.append(denoised_latent.detach().cpu().numpy())

                image = self.decode_latents(denoised_latent, to_pil=return_pil)
                
                if isinstance(image, list):
                    images += image
                else:
                    images.append(image)

        if return_embeddings:
            return images, {'text_encoded': text_encodeds, 'denoised_latent': latents}
        else:
            return images


if __name__ == '__main__':
    sd = StyleDiffusion()
    sd.update_tokens('data/embeddings')

    save_dir = '.temp'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    prompt = "a portrait of gollum in the style of"

    image = sd.generate(prompt)[0]
    image.save(os.path.join(save_dir, 'gollum.png'))

    # embeddings
    artist_1 = "<andy_warhol>"
    artist_2 = "<yayoi_kusama>"

    artist_1 = sd.tokenizer.convert_tokens_to_ids(artist_1)
    artist_2 = sd.tokenizer.convert_tokens_to_ids(artist_2)

    embed_1 = sd.text_encoder.get_input_embeddings().weight.data[artist_1]
    embed_2 = sd.text_encoder.get_input_embeddings().weight.data[artist_2]

    lin = np.linspace(0, 1, 7)
    interpolated = [embed_1 * (1-v) + embed_2 * v for v in lin]

    for i, embed in enumerate(interpolated):
        images, embeddings = sd.generate(prompt, vector=embed, return_text_embeddings=True)

        images[0].save(os.path.join(save_dir, f'diffused_custom_{i}.jpg'))
        np.save(os.path.join(save_dir, f'embeddings_custom_{i}.npy'), embeddings[0])
