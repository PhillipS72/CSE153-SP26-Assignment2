import random
import shutil
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from sampler import MusicGenSampler

def create_evaluation_sets(cfg):
    user_test_path = f"{cfg.evaluation.path}/user_test"

    references = []
    for i in range(cfg.evaluation.num_samples):
        folder = f"{user_test_path}/sample{i+1}"
        os.makedirs(folder, exist_ok=True)

        finetuned_source = f"{cfg.evaluation.path}/finetuned/{i+1:03d}.wav"
        pretrained_source = f"{cfg.evaluation.path}/pretrained/{i+1:03d}.wav"

        desc_source = f"{cfg.evaluation.path}/descriptions/{i+1:03d}.txt"
        desc_dest = f"{user_test_path}/sample{i+1}/prompt.txt"

        if random.random() < 0.5:
            finetuned_dest = f"{user_test_path}/sample{i+1}/melody1.wav"
            pretrained_dest = f"{user_test_path}/sample{i+1}/melody2.wav"
            references.append([i+1, "pretrained", "finetuned"])
        else:
            finetuned_dest = f"{user_test_path}/sample{i+1}/melody2.wav"
            pretrained_dest = f"{user_test_path}/sample{i+1}/melody1.wav"
            references.append([i+1, "finetuned", "pretrained"])

        shutil.copyfile(finetuned_source, finetuned_dest)
        shutil.copyfile(pretrained_source, pretrained_dest)
        shutil.copyfile(desc_source, desc_dest)

    references = pd.DataFrame(references, columns=["sample", "melody1", "melody2"])
    references.to_csv(f"{cfg.evaluation.path}/user_test_references.csv", index=False)

    forms = [[i+1, "", "", "", "", "", "", ""] for i in range(references.shape[0])]
    forms = pd.DataFrame(forms, columns=["sample", "melody1_jingle", "melody1_prompt", "melody1_quality", "melody2_jingle", "melody2_prompt", "melody2_quality", "better"])

    forms.to_csv(f"{user_test_path}/forms.csv", index=False)

    question = "First, please listen to 5 examples of train jingle audios in examples, which are used in actual train stations. Then, please evaluate how each melody melody sounds like a jingle in train station in 1 to 5 scale, and how each melody aligns with description, and how each melody sounds good in terms of quality in 1 to 5 scale. Finally decide which one is sounds better overall in the context of playing them in public transportation in forms.csv. Finally, choose the best melody from 40 samples in best_melody.txt. (example: 'sample 4, melody: 2')"

    with open(f"{user_test_path}/question.txt", "w") as f:
        f.write(question)

    with open(f"{user_test_path}/best_melody.txt", "w") as f:
        f.write("sample: [], melody: []")

    os.makedirs(f"{user_test_path}/examples", exist_ok=True)
    for i, example in enumerate(cfg.evaluation.examples):
        shutil.copyfile(example, f"{user_test_path}/example{i+1}.wav")

@hydra.main(version_base=None, config_path="configs", config_name="")
def main(cfg):
    if cfg.model.name == "musicgen":
        sampler = MusicGenSampler(cfg)
    else:
        raise NotImplementedError

    sampler.sample()
    create_evaluation_sets(cfg)

if __name__ == "__main__":
    main()