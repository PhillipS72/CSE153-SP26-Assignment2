import os
import random

import tqdm
from google import genai

def generate_audio_descriptions(prompt, audio_path, desc_path):
    client = genai.Client()

    for source in os.listdir(audio_path):
        output_folder = f"{desc_path}/{source}"
        os.makedirs(output_folder, exist_ok=True)

        for name in tqdm.tqdm(os.listdir(f"{audio_path}/{source}")):
            output_name = name.replace(".wav", ".txt")
            output_file = f"{output_folder}/{output_name}"

            if os.path.exists(output_file):
                continue

            audio_file = client.files.upload(file=f"{audio_path}/{source}/{name}")
            response = client.models.generate_content(
                model="gemini-3.5-flash", contents=[prompt, audio_file])

            with open(output_file, "w") as f:
                f.write(response.text)

def generate_new_descriptions(prompt, desc_path, inference_path, n_shots, n_samples):
    descriptions = [f"{source}/{name}" for source in os.listdir(desc_path) for name in os.listdir(f"{desc_path}/{source}")]

    client = genai.Client()
    os.makedirs(inference_path, exist_ok=True)

    for i in range(n_samples):
        output_file = f"{inference_path}/{i+1:03d}.txt"
        if os.path.exists(output_file):
            continue

        samples = random.sample(descriptions, n_shots)

        examples = []
        for sample in samples:
            with open(f"{desc_path}/{sample}", "r") as f:
                example = f.read()

            examples.append(example)

        examples = "\n\n".join(examples)
        prompt = examples + "\n\n" + prompt

        response = client.models.generate_content(
            model="gemini-3.5-flash", contents=[prompt])

        with open(output_file, "w") as f:
            f.write(response.text)

if __name__ == "__main__":
    with open("prompt.txt", "r") as f:
        prompt = f.read()

    audio_path = "../data/audio/processed"
    desc_path = "../data/descriptions"

    generate_audio_descriptions(prompt, audio_path, desc_path)

    inference_path = "../data/inferences"

    n_shots = 3
    n_samples = 10
    prompt = f"Above {n_shots} texts are descriptions of short musical jingles. Based on these examples, generate a new description of a melody. Make sure to write a description which is distinct from examples but follows similar length and structure."

    generate_new_descriptions(prompt, desc_path, inference_path, n_shots, n_samples)