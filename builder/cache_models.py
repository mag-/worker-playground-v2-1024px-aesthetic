import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline


def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }

    pipe = fetch_pretrained_model(StableDiffusionXLPipeline,
                                  "playgroundai/playground-v2-1024px-aesthetic", **common_args)

    return pipe


if __name__ == "__main__":
    get_diffusion_pipelines()
