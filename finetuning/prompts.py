import os

import tqdm
from google import genai

def generate_audio_descriptions(prompt, audio_path, desc_path):
    client = genai.Client()

    for source in os.listdir(audio_path):
        output_folder = f"{desc_path}/{source}"
        os.makedirs(output_folder, exist_ok=True)

        for name in tqdm.tqdm(os.listdir(f"{audio_path}/{source}")):
            audio_file = client.files.upload(file=f"{audio_path}/{source}/{name}")

            output_name = name.replace(".wav", ".txt")
            output_file = f"{output_folder}/{output_name}"

            if os.path.exists(output_file):
                continue

            response = client.models.generate_content(
                model="gemini-3.5-flash", contents=[prompt, audio_file])

            with open(output_file, "w") as f:
                f.write(response.text)

if __name__ == "__main__":
    with open("prompt.txt", "r") as f:
        prompt = f.read()

    audio_path = "../data/audio/processed"
    desc_path = "../data/descriptions"

    generate_audio_descriptions(prompt, audio_path, desc_path)
