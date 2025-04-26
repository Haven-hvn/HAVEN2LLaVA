import time
import random
import logging
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from sqlalchemy import create_engine

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

PG_URI = "postgresql://username:password@host:port/database"  # <-- Change this
JSON_OUT = "llava_dataset.json"
IMAGE_FOLDER = "dataset_images"  # <-- Should match --image_folder
THREADS = 10

# --- Create image directory ---
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# --- Updated SQL Query with grouping ---
QUERY = """
SELECT
    vc.thumbnail AS thumbnail_cid,
    ARRAY_AGG(a.action_name) AS actions  # Group actions per image
FROM
    "VideoClip" vc
INNER JOIN
    "VideoClipAction" vca ON vc.clip_id = vca.clip_id
INNER JOIN
    "Action" a ON vca.action_id = a.action_id
WHERE
    vc.thumbnail IS NOT NULL AND
    vc.thumbnail <> ''
GROUP BY vc.thumbnail;
"""


 def fetch_ipfs_image_exponential_backoff(cid,
                                          max_retries=10,
                                          base_delay=2,
                                          max_delay=30,
                                          timeout=15):
     """
     Tries to download an image from IPFS via the configured gateway using exponential backoff.
     Returns image bytes or None if unavailable after all retries.
     """
     url = f"https://premium.w3ipfs.storage/ipfs/{cid}"
     attempt = 0
     while attempt < max_retries:
         attempt += 1
         try:
             resp = requests.get(url, timeout=timeout)
             sc = resp.status_code
             if sc == 200:
                 return resp.content
             elif sc == 404 or sc == 403:
                 return None
             elif sc == 429 or sc >= 500:
                 if attempt == max_retries:
                     return None
                 delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                 jitter = random.uniform(0, 1)
                 time.sleep(delay + jitter)
                 continue
             else:
                 return None
         except requests.exceptions.RequestException:
             if attempt == max_retries:
                 return None
             delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
             jitter = random.uniform(0, 1)
             time.sleep(delay + jitter)
             continue
         except Exception:
             return None
     return None

# --- New: Image saver with conflict resolution ---
def save_image(cid, image_data):
    filename = f"{cid}.jpg"
    filepath = os.path.join(IMAGE_FOLDER, filename)
    
    # Handle filename conflicts
    counter = 1
    while os.path.exists(filepath):
        filename = f"{cid}_{counter}.jpg"
        filepath = os.path.join(IMAGE_FOLDER, filename)
        counter += 1
    
    with open(filepath, "wb") as f:
        f.write(image_data)
    return filename

# --- Modified processing for LLaVA format ---
def process_group(cid, actions):
    image_data = fetch_ipfs_image_exponential_backoff(cid)
    if not image_data:
        return None
    
    try:
        filename = save_image(cid, image_data)
    except OSError as e:
        logging.error(f"Failed to save image {cid}: {str(e)}")
        return None

    # Build conversation turns
    conversations = []
    for action in actions:
        conversations.extend([
            {
                "from": "human",
                "value": f"<image>\nWhat action is happening in this scene?"
            },
            {
                "from": "gpt",
                "value": action
            }
        ])

    return {
        "id": cid,
        "image": filename,
        "conversations": conversations
    }

# --- Main execution flow ---
engine = create_engine(PG_URI)
df = pd.read_sql(QUERY, engine)

dataset = []
with ThreadPoolExecutor(max_workers=THREADS) as executor:
    future_to_cid = {
        executor.submit(process_group, row.thumbnail_cid, row.actions): row.thumbnail_cid
        for row in df.itertuples()
    }
    
    for future in as_completed(future_to_cid):
        cid = future_to_cid[future]
        try:
            result = future.result()
            if result:
                dataset.append(result)
                logging.info(f"Processed {cid} with {len(result['conversations'])//2} actions")
        except Exception as e:
            logging.error(f"Failed processing {cid}: {str(e)}")

# Write final JSON output
with open(JSON_OUT, 'w') as f:
    json.dump(dataset, f, indent=2)

logging.info(f"Dataset created: {JSON_OUT} with {len(dataset)} entries")
