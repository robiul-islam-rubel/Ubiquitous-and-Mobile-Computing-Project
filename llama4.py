import os
from llama_api_client import LlamaAPIClient
import base64
from PIL import Image
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from pathlib import Path
import argparse
from pydantic import BaseModel
from typing import Literal
import json
import time


from dotenv import load_dotenv

# Load the environment variable form .env file
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Input dataset")
parser.add_argument( "--dataset", type=str, help="Input dataset")
parser.add_argument( "--output_folder", type=str, default="vicuna",help="Choose a model to run inference")
args = parser.parse_args()

# === Define Schema ===
class DrivingAssessment(BaseModel):
    traffic_sign_name: str
    explanation_of_traffic_sign:str
    traffic_sign_location: str
    confidence: float

# Read and encode an image to Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def llama4(prompt, image_urls=[]):
  image_urls_content = []
  model = "Llama-4-Maverick-17B-128E-Instruct-FP8" if args.output_folder =="llama4maverick" else "Llama-4-Scout-17B-16E-Instruct-FP8"
  for url in image_urls:
    image_urls_content.append(
        {"type": "image_url", "image_url": {"url": url}})

  content = [{"type": "text", "text": prompt}]
  content.extend(image_urls_content)

  client = LlamaAPIClient(
        api_key=os.environ.get("LLAMA_API_KEY"),
        base_url="https://api.llama.com/v1/",
    )

   
  response = client.chat.completions.create(
    model=model,
    messages=[
       {
        
        "role": "user",
        "content": content
    }
    ],
        response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "driving_assessment",
            "schema": DrivingAssessment.model_json_schema()
        }
    },

    
    temperature=0
  )
  
  # === Try parsing into JSON and validating ===
  raw_output = response.completion_message.content.text.strip()
  # return raw_output

  try:
      result = DrivingAssessment.model_validate_json(raw_output)
      return result.model_dump()   # return dict
  except Exception as e:
      print("[ERROR] Invalid JSON from model:", raw_output)
      raise e

if __name__=="__main__":
    # Base dataset and prompt directory
    DATASET_DIRECTORY = "./1_Datasets"
    PROMPT_DIRECTORY = "./2_GenerateDescriptions/prompts"
    
    # Prompt and input/output directory
    prompt_file = f"{PROMPT_DIRECTORY}/traffic_sign.txt"
    all_images = []

    INPUT_DIR = f"{DATASET_DIRECTORY}/{args.dataset}"
    print(INPUT_DIR)

    ext = "*.png"
    images = sorted(glob.glob(f"{INPUT_DIR}/{ext}"))
    all_images.extend(images)


    OUTPUT_DIR = f"{DATASET_DIRECTORY}/{args.output_folder}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

  
    print(f"Fuck: {all_images}")

    # Load the prompt
    with open(prompt_file, 'r') as file:
        prompt = file.read()

    
    print(f"Selected {len(all_images)}")
    # Process each image
    for img in tqdm(all_images[:1000]):
        b64 = encode_image_to_base64(img)
        img_name = Path(img).stem
        result = llama4(prompt,[f"data:image/jpeg;base64,{b64}"])
        output_json_file = f"{OUTPUT_DIR}/{img_name}.json"
        with open(f"{output_json_file}", 'w') as file:
          # Write the text to the file
          json.dump(result, file, indent=2)
        print(f"Image: {img}")
        print(result)
