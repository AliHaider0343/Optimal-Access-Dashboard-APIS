import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, desc,DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy.sql import func
from typing import Optional
from datetime import datetime
from typing import List,Dict
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

from CreateKnowledgeStore import create_chatbot
from DeleteKnowledgeStore import deleteVectorsusingKnowledgeBaseID
from ChatChain import Get_Conversation_chain
# Create SQLite database engine
SQLALCHEMY_DATABASE_URL = "sqlite:///./Database.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for SQLAlchemy models
Base = declarative_base()
# Define SQLAlchemy model for the entry
class knowledge_Store(Base):
    __tablename__ = "knowledge_Stores"
    knowledge_base_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    descriptive_name= Column(String)
    xml_url = Column(String)
    wordpress_base_url = Column(String)
    syncing_feature = Column(Integer, default=0)
    syncing_period = Column(Integer, default=0)
    syncing_state = Column(Integer, default=0)

class ChatBotsConfigurations(Base):
    __tablename__ = "chatBots_configurations"
    chatbot_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    descriptive_name= Column(String)
    temperature = Column(String)
    llm = Column(String)
    knowledgeBases=Column(String)


class syncingJob(Base):
    __tablename__ = "Sycing_Jobs"
    job_id = Column(Integer, primary_key=True, index=True)
    knowledge_base_id = Column(Integer)
    user_id = Column(Integer)
    xml_url = Column(String)
    wordpress_base_url = Column(String)
    syncing_period = Column(Integer, default=0)
    last_performed = Column(DateTime, default=func.now())

# Create tables in the database
Base.metadata.create_all(bind=engine)

# Define Pydantic model for request body
class knowledgeStoreCreate(BaseModel):
    user_id: int
    descriptive_name:str
    xml_url: str
    wordpress_base_url: str
    syncing_feature: int
    syncing_period: int
    syncing_state: int

class syncingFeatureStatus(BaseModel):
    knowledgeStoreId: int
    syncPeriod: int

class ChatBots(BaseModel):
    user_id :int
    descriptive_name:str
    temperature:str
    llm:str
    knowledgeBases: str

class EditChatBots(BaseModel):
    descriptive_name:str
    temperature:str
    llm:str
    knowledgeBases: str


class ChatRequest(BaseModel):
    chatbotId: int
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    answer: str
    reference_context: List[dict]

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.post("/Chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    db = SessionLocal()
    db_entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == request.chatbotId).first()
    if db_entry:
        try:
            answer,sources=Get_Conversation_chain(db_entry.knowledgeBases,db_entry.temperature,db_entry.llm,request.question,request.chat_history)
            return ChatResponse(answer=answer, reference_context=sources)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return ChatResponse(answer="Chatbot COnfiguration not Found under ID: " + str(request.chatbotId) , reference_context=[])
# Endpoint to create a new entry
@app.post("/KnowledgeStore/")
def create_knowledge_Store(entry: knowledgeStoreCreate):
    db = SessionLocal()
    db_entry = knowledge_Store(**entry.dict())
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    latest_knowlegde_id = db_entry.knowledge_base_id  # Get the knowledge base ID of the latest entry
    db.close()
    if create_chatbot(entry.user_id,str(latest_knowlegde_id),entry.xml_url,entry.wordpress_base_url):
        return {"message": "Knowledge Base Successfully Created with ID: " + str(latest_knowlegde_id)+" under user ID : "+str(entry.user_id)}
    last_entry = db.query(knowledge_Store).order_by(desc(knowledge_Store.knowledge_base_id)).first()
    if last_entry:
        db.delete(last_entry)
        db.commit()
        db.close()
    return {"message": "An Error occured while Creating the Vector Storage with ID: " + str(
        latest_knowlegde_id) + " under user ID : " + str(entry.user_id)}

# Endpoint to create a new entry
@app.post("/CreateChatbots/")
def createChatbot(entry: ChatBots):
    db = SessionLocal()
    db_entry = ChatBotsConfigurations(**entry.dict())
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return {"message": "Chatbot Configuration stored Successfully."}

# Endpoint to edit chatbot information based on ID
@app.put("/EditChatbot/{chatbot_id}")
def edit_chatbot(chatbot_id: int, edited_info: EditChatBots):
    db = SessionLocal()
    db_entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == chatbot_id).first()
    if db_entry is None:
        raise HTTPException(status_code=404, detail="Chatbot Configuration not Found")

    for key, value in edited_info.dict().items():
        setattr(db_entry, key, value)
    db.commit()
    db.refresh(db_entry)
    return {"message": "Chatbot information updated successfully"}

@app.post("/TurnSycningFeatureOn/")
def TurnSyncingFeatureOn(entry: syncingFeatureStatus):
    db = SessionLocal()
    knowldegeStore = db.query(knowledge_Store).filter(knowledge_Store.knowledge_base_id == entry.knowledgeStoreId).first()
    if knowldegeStore:
        if knowldegeStore.syncing_feature==1:
            return {"message": "Syncing Feature already Turned on for Vector Storage with ID: " + str(
                entry.knowledgeStoreId)}
        knowldegeStore.syncing_feature = 1
        knowldegeStore.syncing_period=entry.syncPeriod
        db.commit()
        db.close()
        return {"message": "Syncing Feature Successfully Turned on for Vector Storage with ID: " + str(entry.knowledgeStoreId)}
    else:
        db.close()
        raise HTTPException(status_code=404, detail="Knowledge base with ID: " + str(entry.knowledgeStoreId) + " not found")

@app.get("/TurnSycningFeatureOff/{knowledge_base_id}")
def TrunSycingFeatureOff(knowledge_base_id: int):
    db = SessionLocal()
    entry = db.query(knowledge_Store).filter(knowledge_Store.knowledge_base_id == knowledge_base_id).first()
    if entry:
        if entry.syncing_feature==0:
            return {"message": "Syncing Feature already Turned off for knowledge base with ID: " + str(knowledge_base_id)}
        entry.syncing_feature = 0
        entry.syncing_state=0
        entry.syncing_period=0
        job = db.query(syncingJob).filter(syncingJob.knowledge_base_id == knowledge_base_id).first()
        if job:
            db.delete(job)
        db.commit()
        db.close()
        return {"message": "Syncing Features Turned Off and Syncing Job Revoked for knowledge base with ID: " + str(knowledge_base_id)}
    else:
        db.close()
        raise HTTPException(status_code=404, detail="Knowledge base with ID: " + str(knowledge_base_id) + " not found")

@app.get("/GetChatbotsbyUserID/{user_id}")
def get_chatbots_by_user_ID(user_id: int):
    db = SessionLocal()
    entries = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id == user_id).all()
    return entries

@app.get("/GetChatbotEmbedableScript/{chatbot_id}")
def get_chatbots_Embdeding_Script(chatbot_id: int):
    db = SessionLocal()
    entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == chatbot_id).first()

    if entry:
        return f"""<script src="https://karan-api.000webhostapp.com/Chatbot.js"></script><script>setupChatbot({entry.chatbot_id},'{entry.descriptive_name}');</script>"""
    db.close()
    raise HTTPException(status_code=404, detail="Chatbot with ID: " + str(chatbot_id) + " not found")


@app.get("/GetknowledgeStoresbyUserID/{user_id}")
def get_knowledge_store_by_user_ID(user_id: int):
    db = SessionLocal()
    entries = db.query(knowledge_Store).filter(knowledge_Store.user_id == user_id).all()
    return entries

@app.get("/StartSyncing/{knowledge_base_id}")
def StartSyncing(knowledge_base_id: int):
    db = SessionLocal()
    entry = db.query(knowledge_Store).filter(knowledge_Store.knowledge_base_id == knowledge_base_id).first()
    if entry:
        if entry.syncing_state==1:
            return {"message": "Syncing already started for knowledge base with ID: " + str(knowledge_base_id)}
        entry.syncing_state = 1
        job=syncingJob(
                user_id=entry.user_id,
                knowledge_base_id= knowledge_base_id,
                syncing_period= entry.syncing_period,
                xml_url= entry.xml_url,
                wordpress_base_url=entry.wordpress_base_url,
                last_performed=datetime.now()
        )
        db.add(job)
        db.commit()
        db.close()
        return {"message": "Syncing started for knowledge base with ID: " + str(knowledge_base_id)}
    else:
        db.close()
        raise HTTPException(status_code=404, detail="Knowledge base with ID: " + str(knowledge_base_id) + " not found")

@app.get("/StopSyncing/{knowledge_base_id}")
def StopSyncing(knowledge_base_id: int):
    db = SessionLocal()
    entry = db.query(knowledge_Store).filter(knowledge_Store.knowledge_base_id == knowledge_base_id).first()
    if entry:
        if entry.syncing_state==0:
            return {"message": "Syncing already Stopped for knowledge base with ID: " + str(knowledge_base_id)}
        entry.syncing_state = 0
        job = db.query(syncingJob).filter(syncingJob.knowledge_base_id == knowledge_base_id).first()
        if job:
            db.delete(job)
        db.commit()
        db.close()
        return {"message": "Syncing stopped for knowledge base with ID: " + str(knowledge_base_id)}
    else:
        db.close()
        raise HTTPException(status_code=404, detail="Knowledge base with ID: " + str(knowledge_base_id) + " not found")

@app.delete("/deleteChatbot/{chatbot_id}")
def delete_Chatbot(chatbot_id: int):
    db = SessionLocal()
    entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == chatbot_id).first()
    if entry:
        db.delete(entry)
        db.commit()
        db.close()
        return {"message": "Chatbot Deleted Successfully with ID: " + str(chatbot_id)}
    else:
        raise HTTPException(status_code=404,
                                detail="Chatbot with ID: " + str(chatbot_id) + " not found in Databse.")

@app.delete("/deleteKnowledgeBase/{knowledge_base_id}")
def delete_knowledge_base(knowledge_base_id: int):
    db = SessionLocal()
    entry = db.query(knowledge_Store).filter(knowledge_Store.knowledge_base_id == knowledge_base_id).first()
    if entry:
        if deleteVectorsusingKnowledgeBaseID(knowledge_base_id):
            job = db.query(syncingJob).filter(syncingJob.knowledge_base_id == knowledge_base_id).first()
            if job:
                db.delete(job)
                db.commit()
            chatbots = db.query(ChatBotsConfigurations).all()
            for chatbot in chatbots:
                knowledge_bases = json.loads(chatbot.knowledgeBases)
                if knowledge_base_id in knowledge_bases:
                    if len(knowledge_bases) == 1:
                        db.delete(chatbot)
                        db.commit()
                    else:
                        knowledge_bases.remove(knowledge_base_id)
                        chatbot.knowledgeBases = json.dumps(knowledge_bases)
                        db.commit()
            db.delete(entry)
            db.commit()
            db.close()
            return {"message": "Knowledge Base Deleted Successfully with ID: " + str(knowledge_base_id)}
        else:
            raise HTTPException(status_code=404,
                                detail="Knowledge base with ID: " + str(knowledge_base_id) + " not found in Vector Documents.")
            db.close()
    else:
        db.close()
        raise HTTPException(status_code=404, detail="Knowledge base with ID: " + str(knowledge_base_id) + " not found")


