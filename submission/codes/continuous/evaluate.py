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

        conditional_source = f"{cfg.evaluation.path}/conditional/{i+1:03d}.wav"
        unconditional_source = f"{cfg.evaluation.path}/unconditional/{i+1:03d}.wav"
        pretrained_source = f"{cfg.evaluation.path}/pretrained/{i+1:03d}.wav"

        desc_source = f"{cfg.evaluation.path}/descriptions/{i+1:03d}.txt"
        desc_dest = f"{user_test_path}/sample{i+1}/prompt.txt"

        indices = list(range(3))
        random.shuffle(indices)

        conditional_dest = f"{user_test_path}/sample{i+1}/melody{indices[0]+1}.wav"
        unconditional_dest = f"{user_test_path}/sample{i+1}/melody{indices[1]+1}.wav"
        pretrained_dest = f"{user_test_path}/sample{i+1}/melody{indices[2]+1}.wav"

        reference = ["", "", ""]
        reference[indices[0]] = "conditional"
        reference[indices[1]] = "unconditional"
        reference[indices[2]] = "pretrained"
        references.append([i+1] + reference)

        shutil.copyfile(conditional_source, conditional_dest)
        shutil.copyfile(unconditional_source, unconditional_dest)
        shutil.copyfile(pretrained_source, pretrained_dest)
        shutil.copyfile(desc_source, desc_dest)

    references = pd.DataFrame(references, columns=["sample", "melody1", "melody2", "melody3"])
    references.to_csv(f"{cfg.evaluation.path}/user_test_references.csv", index=False)

    forms = [[i+1, "", "", "", "", "", "", ""] for i in range(references.shape[0])]

    columns = ["sample", "best",
        "melody1_jingle", "melody1_quality",
        "melody2_jingle", "melody2_quality",
        "melody3_jingle", "melody3_quality",
    ]
    forms = pd.DataFrame(forms, columns=columns)

    forms.to_csv(f"{user_test_path}/forms.csv", index=False)

    question = "First, please listen to 5 examples of train jingle audios in examples, which are used in actual train stations. Then, please evaluate how each melody melody sounds like a jingle in train station in 1 to 5 scale and how each melody sounds good in terms of quality in 1 to 5 scale. Finally decide which one is sounds better overall in the context of playing them in public transportation in forms.csv. Finally, choose the best melody from 60 samples in best_melody.txt. (example: 'sample 4, melody: 2')"

    with open(f"{user_test_path}/question.txt", "w") as f:
        f.write(question)

    with open(f"{user_test_path}/best_melody.txt", "w") as f:
        f.write("sample: [], melody: []")

    os.makedirs(f"{user_test_path}/examples", exist_ok=True)
    for i, example in enumerate(cfg.evaluation.examples):
        shutil.copyfile(example, f"{user_test_path}/examples/example{i+1}.wav")

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