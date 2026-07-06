import os
import time
import logging
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

CSV_PATH = os.path.join(os.path.dirname(__file__), "idsp_data_final.csv")
INDEX_NAME = "aroha-db"
BATCH_SIZE = 100

pc = Pinecone(api_key=os.getenv("PC_API_Key"))

existing = [i["name"] for i in pc.list_indexes()]
if INDEX_NAME not in existing:
    log.info("Creating index %s", INDEX_NAME)
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

while not pc.describe_index(INDEX_NAME).status["ready"]:
    time.sleep(1)

index = pc.Index(INDEX_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv(CSV_PATH)
df["no_deaths"] = pd.to_numeric(df["no_deaths"], errors="coerce").fillna(0).astype(int)
df["no_of_cases"] = pd.to_numeric(df["no_of_cases"], errors="coerce").fillna(0).astype(int)

vectors = []
total = len(df)

for i, row in df.iterrows():
    document = (
        f"Location: State->{row['state']}, City->{row['district']}\n"
        f"Disease: {row['disease']}\n"
        f"Occuring Date: {row['start_of_outbreak']}\n"
        f"Reporting Date: {row['reporting_data']}\n"
        f"Number of Cases: {row['no_of_cases']}\n"
        f"Number of Deaths: {row['no_deaths']}\n"
        f"Status: {row['status']}\n"
        f"Action Taken To Mitigate the Incident:\n{row['action']}"
    )

    vectors.append({
        "id": str(i),
        "values": model.encode(document).tolist(),
        "metadata": {
            "text": document,
            "state": str(row["state"]) if pd.notna(row["state"]) else "Unknown",
            "district": str(row["district"]) if pd.notna(row["district"]) else "Unknown",
            "disease": str(row["disease"]) if pd.notna(row["disease"]) else "Unknown",
            "status": str(row["status"]) if pd.notna(row["status"]) else "Unknown",
            "cases": int(row["no_of_cases"]),
            "deaths": int(row["no_deaths"]),
            "start_date": str(row["start_of_outbreak"]) if pd.notna(row["start_of_outbreak"]) else "",
            "report_date": str(row["reporting_data"]) if pd.notna(row["reporting_data"]) else "",
        },
    })

    if len(vectors) >= BATCH_SIZE:
        index.upsert(vectors=vectors)
        log.info("Uploaded %d / %d records", i + 1, total)
        vectors = []

if vectors:
    index.upsert(vectors=vectors)
    log.info("Uploaded final batch — done.")
