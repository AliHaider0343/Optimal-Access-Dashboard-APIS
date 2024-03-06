from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

chroma_db = Chroma(persist_directory=f"./Optimal-Access-Vector-Store", embedding_function=OpenAIEmbeddings())

def deleteVectorsusingKnowledgeBaseID(knowledgeBaseID):
    documents = chroma_db.get()
    count=0
    for document_id, metadata in zip(documents['ids'], documents['metadatas']):
        if str(metadata['knowledge_Store_id']) == str(knowledgeBaseID):
            chroma_db.delete([document_id])
            count+=1
    return count>0
