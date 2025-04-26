import time
import random
import logging
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
    ARRAY_AGG(a.action_name ORDER BY vca.initial_confidence_score DESC) AS actions
FROM
    "VideoClip" vc
INNER JOIN
    "VideoClipAction" vca ON vc.clip_id = vca.clip_id
INNER JOIN
    "Action" a ON vca.action_id = a.action_id
WHERE
    vc.thumbnail IS NOT NULL
    AND vc.thumbnail <> ''
    AND vca.initial_confidence_score >= 0.7
GROUP BY vc.thumbnail
HAVING COUNT(*) >= 1;
"""


def fetch_ipfs_image_exponential_backoff(cid, max_retries=10, base_delay=2, max_delay=30, timeout=15):
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
    # TODO we don't have all the data get for multiple structured tag hierarchies
    # Structured Tag Hierarchy (soccer example)
    # TAG_HIERARCHY = {
    # 'primary': ['goal scored', 'penalty', 'red card'],
    # 'secondary': ['defending', 'kicking', 'passing'],
    # 'tertiary': ['running', 'jumping', 'celebrating']
    # }
    conversations = []
    if actions:
        # Use confidence-ordered actions from SQL
        main_action = actions[0]
        secondary_actions = actions[1:]
        
        # Primary action with confidence context
        conversations.extend([
            {
                "from": "human",
                "value": "<image>\nWhat's the most certain action in this scene?"
            },
            {
                "from": "gpt",
                "value": main_action
            }
        ])
        
        # Confidence-aware follow-up
        if secondary_actions:
            conversations.extend([
                {
                    "from": "human",
                    "value": "What other high-probability actions exist?"
                },
                {
                    "from": "gpt",
                    "value": ", ".join(secondary_actions[:3]) + " (lower confidence)"
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

# --- Initialize or load existing JSON ---
if os.path.exists(JSON_OUT):
    with open(JSON_OUT, 'r') as f:
        dataset = json.load(f)
    existing_cids = {item['id'] for item in dataset}
else:
    dataset = []
    existing_cids = set()

with ThreadPoolExecutor(max_workers=THREADS) as executor:
    future_to_cid = {
        executor.submit(process_group, row.thumbnail_cid, row.actions): row.thumbnail_cid
        for row in df.itertuples()
        if row.thumbnail_cid not in existing_cids
    }

    if not existing_cids:
        with open(JSON_OUT, 'w') as f:
            json.dump([], f, indent=2)

    # Modified progress bar handling
    with tqdm(total=len(future_to_cid), desc="Processing CIDs") as pbar:
        for future in as_completed(future_to_cid):
            cid = future_to_cid[future]
            try:
                result = future.result()
                if result:
                    dataset.append(result)
                    # Update JSON
                    with open(JSON_OUT + '.tmp', 'w') as f:
                        json.dump(dataset, f, indent=2)
                    os.replace(JSON_OUT + '.tmp', JSON_OUT)
                    # Update progress bar description with last processed CID
                    pbar.set_postfix_str(cid, refresh=False)
                pbar.update(1)
            except Exception as e:
                logging.error(f"Failed processing {cid}: {str(e)}")
                pbar.update(1)

# Write final JSON output
with open(JSON_OUT, 'w') as f:
    json.dump(dataset, f, indent=2)

logging.info(f"Dataset created: {JSON_OUT} with {len(dataset)} entries")
