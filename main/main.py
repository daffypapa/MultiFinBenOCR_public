from lib.agent import Agent
from lib.tools import Tools
import pandas as pd
from tqdm import tqdm
import os
import time

def evaluate(model_name="gpt-4o", experiment_tag="zero-shot",language = "en", sample = None):
    tools = Tools()

    if language == "en":
        df = pd.read_parquet("hyr_ocr_process/output_parquet_hyr/EnglishOCR.parquet")
    elif language == "es":
        df = pd.read_parquet("hyr_ocr_process/spanish_output_parquet/spanish_batch_0000.parquet")
    else: 
        print("Not a valid choice of language, please try again.")
        return language
    
    if sample:
        df = df.head(sample)  # Run sample

    experiment_name = f"{model_name}_{experiment_tag}_financial"

    if language == "en":
        experiment_folder = os.path.join("hyr_results/predictions/", experiment_name)
    elif language == "es":
        experiment_folder = os.path.join("hyr_results/predictions_spanish/", experiment_name)

    os.makedirs(experiment_folder, exist_ok=True)

    agent = Agent(model_name)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {model_name}"):
        image_path = row["image_path"]
        ground_truth = row["matched_html"]
        output_file = os.path.join(experiment_folder, f"{model_name}_pred_{i}.txt")

        try:
            result = agent.draft(image_path, by_line=False)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            time.sleep(1.5)
        except Exception as e:
            print(f"⚠️ Error on index {i}: {e}")
            continue

def main():
    # Change this to "blip" or "llava" to control model
    evaluate(model_name="gpt-4o",language = "en", sample = 20)

if __name__ == '__main__':
    main()
