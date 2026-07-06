from chromadb import CloudClient
import pandas as pd
import os
cdb_clt = CloudClient(
    api_key=os.getenv('CHROMA_API_KEY'),
    tenant=os.getenv('CHROMA_TENANT'),
    database='ADB'
)
df = pd.read_csv(r'C:\Users\KVKDEV\Desktop\KVK_CODESPACE\AROHA_PROJECT\project\aroha\app\api\python-api\api\idsp_data_final.csv')
def return_context(ip: str, collection_name: str = 'AROHA_data') -> str:
   coll=cdb_clt.get_collection(name=collection_name)
   results=coll.query(query_texts=[ip],n_results=15)
   if not results["documents"] or not results["documents"][0]:
        return "query_error"
   retrieved_chunks=results["documents"][0]
   context="\n\n".join(retrieved_chunks)
   return context