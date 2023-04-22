import os
import torch
import PIL
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import argparse
from pathlib import Path

def save_grid(imgs, rows, cols, path):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
        
    grid.save(path)


def setup_pipe(embed_step, sample_step, guidance, pretrain_path, learned_embeds_path):
    
    tokenizer = CLIPTokenizer.from_pretrained(
      pretrain_path,
      subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
      pretrain_path, subfolder="text_encoder", torch_dtype=torch.float16)

  # load newly trained to CLIP
    def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
        num_placeholder_tokens = len(loaded_learned_embeds)
        for i in range(num_placeholder_tokens):

            # separate token and the embeds
            trained_token = list(loaded_learned_embeds.keys())[i]

            embeds = loaded_learned_embeds[trained_token]
            # cast to dtype of text_encoder
            dtype = text_encoder.get_input_embeddings().weight.dtype
            embeds.to(dtype)

            # add the token in tokenizer
            num_added_tokens = tokenizer.add_tokens(trained_token)
            # if num_added_tokens == 0:
            #   raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

            # resize the token embeddings
            text_encoder.resize_token_embeddings(len(tokenizer))

            # get the id for the token and assign the embeds
            token_id = tokenizer.convert_tokens_to_ids(trained_token)
            text_encoder.get_input_embeddings().weight.data[token_id] = embeds

        
    load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer)

    pipe = StableDiffusionPipeline.from_pretrained(
      pretrain_path,
      torch_dtype=torch.float16,
      text_encoder=text_encoder,
      tokenizer=tokenizer).to("cuda")
    
    return pipe
    

def check_folder_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def save_images_independent(all_images, path):
    check_folder_exist(path)
    output_folder = Path(path)
    output_folder.mkdir(exist_ok=True)
    for idx, img in enumerate(all_images):
        img_path = output_folder / f"image_{idx + 200}.jpeg"
        img.save(img_path, format='JPEG')
    
    
    
    
if __name__ == "__main__":
    from ti_base import generate_placeholder_token_string
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_image_name', type=str, required=True)
    parser.add_argument('--initialization', type=str, required=True)
    parser.add_argument('--init_type', type=str, required=True)  # one of {null, class, caption}
    # parser.add_argument('--eval_step', type=str, required=True)
    parser.add_argument('--num_rows', type=int, required=True)
    parser.add_argument('--num_cols', type=int, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    
    args = parser.parse_args()
    
    target_image_name = args.target_image_name
    init_string = args.initialization
    init_type = args.init_type
    num_rows = args.num_rows
    num_samples = args.num_cols
    # prompt = "A photo of \u003C{}>".format(token_map[init_token])
    pretrain_path = "sd-concept-output"
    sample_steps = 30
    guidance_scale = 7.5
    # steps = [i * 20 for i in range(10)] + [i * 100 + 200 for i in range(8)] + [i * 250 + 1000 for i in range(5)]
    steps = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240]
    # steps = [i * 500 + 2000 for i in range(1, 7)]
    
    
    # Get the num_placeholder_tokens
    base_path = f"{target_image_name}/{init_type}_{init_string}_init"
    loaded_learned_embeds = torch.load(
        os.path.join(base_path, "learned_embeds-step-0.bin"),
        map_location="cpu"
    )
    num_placeholder_tokens = len(loaded_learned_embeds)

    
    place_holder_string = generate_placeholder_token_string(num_placeholder_tokens)
    
    # breakpoint()
    prompts = ["A photo of {}".format(place_holder_string),
              "a good photo of a {}".format(place_holder_string),
                "the photo of a {}".format(place_holder_string)
              ]
    
    if args.prompt != '':
        prompts = [args.prompt.format(place_holder_string)]


        
    for step in steps:
        print("Inferencing step {} ...".format(step))
        embeds_path = base_path+'/learned_embeds-step-{}.bin'.format(step)
        pipe = setup_pipe(step, sample_steps, guidance_scale, pretrain_path, embeds_path)
        all_images = [] 
        for prompt in prompts:
            for _ in range(num_rows):
                all_images.extend(pipe([prompt] * num_samples, num_inference_steps=sample_steps, guidance_scale=guidance_scale).images)

        # save images
        save_path = base_path + '_results'
        check_folder_exist(save_path)
        # label = save_path+"/{}_step_{}.jpg".format(prompt, step)
        # save_grid(all_images, num_rows, num_samples, label)
        # save all generated images into a foler as well
        if args.prompt != '':
            save_images_independent(all_images, save_path+f"/step_{step}"+f"/prompt_{args.prompt}")
        else:
            save_images_independent(all_images, save_path+f"/step_{step}")
        
    

