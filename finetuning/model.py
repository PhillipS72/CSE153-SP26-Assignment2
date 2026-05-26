from transformers import MusicgenForConditionalGeneration

def load_model(cfg):
    if cfg.model_name == "musicgen":
        model = MusicgenForConditionalGeneration.from_pretrained(
            f"facebook/musicgen-{cfg.model_size}", device_map="auto")

    else:
        raise NotImplementedError

    return model
