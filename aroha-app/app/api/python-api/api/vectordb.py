from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chromadb import CloudClient
import pandas as pd
import os
import time

# Connect to Chroma Cloud
cdb_clt = CloudClient(
    api_key=os.getenv('CHROMA_API_KEY'),
    tenant=os.getenv('CHROMA_TENANT'),
    database='ADB'
)

# Read CSV
df = pd.read_csv(r'D:\KVKDEV\RAG\idsp_data_final.csv')

# Create/Get Collection
coll = cdb_clt.get_or_create_collection(
    name='AROHA_data',
    embedding_function=DefaultEmbeddingFunction()
)

documents = []
ids = []
metadatas = []

print("Preparing documents...")

for index in range(len(df)):
    row = df.iloc[index]

    template = f"""
Location: State->{row['state']}, City->{row['district']}
Disease: {row['disease']}
Occurring Date: {row['start_of_outbreak']}
Reporting Date: {row['reporting_data']}
Number of Cases: {row['no_of_cases']}
Number of Deaths: {row['no_deaths']}
Status: {row['status']}
Action Taken To Mitigate the Incident:
{row['action']}
"""

    documents.append(template)

    ids.append(str(index))

    metadatas.append({
        "id": index,
        "state": str(row['state']),
        "city": str(row['district']),
        "disease": str(row['disease']),
        "status": str(row['status']),
        "cases": str(row['no_of_cases']),
        "deaths": str(row['no_deaths']),
        "start_date": str(row['start_of_outbreak']),
        "report_date": str(row['reporting_data'])
    })

    if (index + 1) % 1000 == 0:
        print(f"Prepared {index + 1}/{len(df)} records")

print(f"\nFinished preparing {len(documents)} documents.")
print("Starting upload...\n")

# Upload in batches
BATCH_SIZE = 100
total_docs = len(documents)

for start in range(0, total_docs, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total_docs)

    print(f"Uploading batch {start // BATCH_SIZE + 1}")
    print(f"Records: {start} -> {end - 1}")

    batch_start = time.time()

    coll.add(
        ids=ids[start:end],
        documents=documents[start:end],
        metadatas=metadatas[start:end]
    )

    elapsed = time.time() - batch_start

    print(
        f"({elapsed:.2f} sec)"
    )
    print("-" * 50)

print("\n🎉 All documents uploaded successfully!")
print(f"Total records uploaded: {total_docs}")