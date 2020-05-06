import json
import requests
import pandas as pd
from pathlib import Path


def fetch_data(**params): # crawling
  url = "https://api.gnavi.co.jp/PhotoSearchAPI/v3/"
  response = requests.get(url, params=params)
  return response.json() # transforming into dictionary

def extract_data(response): # scraping
  """
    Here, we want to get 'Review comments' and 'Total score'.
    They're stored under 'response/{number}/photo/' as 'comment' and 'total_score', respectively.
    Set if branch to through empty comments.
  """
  for key in response["response"].keys():
    if not key.isdigit():
      continue
    d = response["response"][key]["photo"]
    if d.get("comment") and d.get("total_score"):
      comment = d["comment"]
      score = d["total_score"]
      data = {
        "comment": comment,
        "score": score
      }
      yield data

def save_as_json(save_file, record):
  with open(save_file, mode="a") as f:
    f.write(json.dumps(record) + "\n")

def initialize_files(paths:list) -> None:
  for path_ in paths:
    Path(path_).touch()

def main():
  # declare constant parameters
  raw_data  = "data/raw_data.json"
  save_file = "data/dataset.jsonl"
  key_dict  = pd.read_json("../../gnavi_keys.json", encoding="utf-8")

  # initialize files
  initialize_files([raw_data, save_file])
  
  # fetch and save data 
  params = {
    "keyid": key_dict["api_key"], 
    "area": "新宿",
    "hit_per_page": 50,
    "offset_page":1
    }
  response = fetch_data(**params)
  save_as_json(raw_data, response)
  
  # extract necessary informations and save
  records = extract_data(response)
  for record in records:
    save_as_json(save_file, record)

if __name__=="__main__":
  main()
