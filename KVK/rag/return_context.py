from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chromadb import PersistentClient
import os
import pandas as pd
df=pd.read_csv(rf'{os.getcwd()}\Aroha\KVK\data_folder\data.csv')
clt=PersistentClient(path=rf'{os.getcwd()}\Aroha\KVK\rag\db')
documents=[]
ids=[]
metadatas=[]
collection_name='AROHA_data'
embedding_function=DefaultEmbeddingFunction()
try:
    collection=clt.get_collection(name=collection_name)
except Exception as e:
    collection=clt.create_collection(name=collection_name,embedding_function=embedding_function)
counter=0
dict_var = {}
for index in range(len(df)):
    dict_var[index] = df.iloc[index].to_dict()
for i in dict_var:
    template = f'''
Location:State->{dict_var[i]['state']},City->{dict_var[i]['district']}
Disease:{dict_var[i]['disease']}
Occuring Date:{dict_var[i]['start_of_outbreak']}
Reporing Date:{dict_var[i]['date_of_reporting']}
Number of Cases:{dict_var[i]['no_of_cases']}
Number of Deaths:{dict_var[i]['deaths']}
Status:{dict_var[i]['status']}
Action Taken To Mitigate the Incident:
{dict_var[i]['action']}
'''
    documents.append(template)
    ids.append(str(counter))
    metadatas.append({
        "id": counter,
        "state": dict_var[i]['state'],
        "city": dict_var[i]['district'],
        "disease": dict_var[i]['disease'],
        "status": dict_var[i]['status'],
        "cases": dict_var[i]['no_of_cases'],
        "deaths": dict_var[i]['deaths'],
        "start_date": dict_var[i]['start_of_outbreak'],
        "report_date": dict_var[i]['date_of_reporting']
    })
    counter += 1
collection.add(ids=ids,documents=documents,metadatas=metadatas)
def return_context(query):
    results=collection.query(query_texts=[query],n_results=15)
    if not results["documents"] or not results["documents"][0]:
        return "query_error"
    retrieved_chunks=results["documents"][0]
    context="\n\n".join(retrieved_chunks)
    return context