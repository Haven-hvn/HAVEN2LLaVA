import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

PG_URI = "postgresql://username:password@host:port/database"  # <-- Change this
PARQUET_OUT = "output_file.parquet"
THREADS = 10
BATCH_SIZE = 200

# --- SQL Query ---
QUERY = """
SELECT
    vc.thumbnail AS thumbnail_cid,
    a.action_name
FROM
    "VideoClip" vc
INNER JOIN
    "VideoClipAction" vca ON vc.clip_id = vca.clip_id
INNER JOIN
    "Action" a ON vca.action_id = a.action_id
WHERE
    vc.thumbnail IS NOT NULL AND
    vc.thumbnail <> '';
"""

# --- Exponential Backoff Fetcher ---
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

# --- Metadata fetch (small footprint) ---
engine = create_engine(PG_URI)
full_df = pd.read_sql(QUERY, engine)

# --- PyArrow schema ---
arrow_schema = pa.schema([
    ('image', pa.binary()),
    ('query', pa.string()),
    ('labels', pa.list_(pa.string())),
    ('human_or_machine', pa.int64())
])

# --- Batch iterator ---
def df_batches(df, size):
    for start in range(0, len(df), size):
        yield df.iloc[start:start+size]

def parallel_image_fetch_records(batch_df):
    records = []
    # Use original DataFrame index to preserve order if desired
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        # Future maps back to row index
        future_to_rowidx = {
            executor.submit(fetch_ipfs_image_exponential_backoff, cid): (i, row['action_name'])
            for i, (cid, row) in enumerate(zip(batch_df['thumbnail_cid'], batch_df.itertuples()))
        }
        for future in as_completed(future_to_rowidx):
            i, action_name = future_to_rowidx[future]
            img_bytes = future.result()
            if img_bytes is not None:
                record = {
                    "image": img_bytes,
                    "query": "What action is happening in this scene?",
                    "labels": [action_name],
                    "human_or_machine": 1
                }
                records.append(record)
    return records

# --- Streaming write loop ---
with pq.ParquetWriter(PARQUET_OUT, arrow_schema) as writer:
    for batch_df in df_batches(full_df, BATCH_SIZE):
        recs = parallel_image_fetch_records(batch_df)
        if not recs:
            continue  # skip if no images fetched for this batch
        batch_table = pa.Table.from_pandas(
            pd.DataFrame(recs),
            schema=arrow_schema,
            preserve_index=False
        )
        writer.write_table(batch_table)
        logging.info(f"Wrote batch of {len(recs)} clips to parquet.")

logging.info(f"Done. Parquet file created: {PARQUET_OUT}")
