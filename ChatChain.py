import json

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from operator import itemgetter
from typing import Dict, List, Optional, Sequence
from langchain.schema.retriever import BaseRetriever
from pydantic import BaseModel
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema.runnable import (Runnable, RunnableBranch,
                                       RunnableLambda, RunnableMap)
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

embedding_function = OpenAIEmbeddings()
chroma_db = Chroma(persist_directory=f"./Optimal-Access-Vector-Store", embedding_function=OpenAIEmbeddings())

RESPONSE_TEMPLATE = """\
Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] and [${{Time Stamp}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "I donot have any information about it because it isn't provided in my context i do apologize for in convenience." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

MUST REMEMBER: Do not answer question on your own Must Refer to the Context If there is no relevant information within the context, just say "Sorry for Inconvenice, i dont have any Information about it in my Digital Brain." Don't try to make up an answer. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user. You are a helpful AI Assistant. Respond to the Greeting Messages Properly."""
REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

refrence_docuemnts_sources=[]
class ChatRequest(BaseModel):
    knowledgeBases: str
    temperature: str
    model: str
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None

def create_retriever_chain(llm: BaseLanguageModel, retriever: BaseRetriever) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()).with_config(
            run_name="CondenseQuestion", )
    conversation_chain = condense_question_chain | retriever

    return RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
            ),
            (
                    RunnableLambda(itemgetter("question")).with_config(
                        run_name="Itemgetter:question"
                    )
                    | retriever
            ).with_config(run_name="RetrievalChainWithNoHistory"),
        ).with_config(run_name="RouteDependingOnChatHistory")

def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    global refrence_docuemnts_sources
    if len(refrence_docuemnts_sources) > 0:
        refrence_docuemnts_sources = []
    for i, doc in enumerate(docs):
        refrence_docuemnts_sources.append({'Context-Information': doc.page_content,
                   'Source Link': doc.metadata['KuratedContent_sourceUrl'],
                    'Word Press Popup Link': str(doc.metadata['KuratedContent_WordpressPopupUrl'])
                                           })
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)

def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history

def create_chain(llm: BaseLanguageModel,retriever: BaseRetriever,) -> Runnable:
    retriever_chain = create_retriever_chain(
            llm,
            retriever,
        ).with_config(run_name="FindDocs")
    _context = RunnableMap(
            {
                "context": retriever_chain | format_docs,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
        ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RESPONSE_TEMPLATE),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
            run_name="GenerateResponse",
        )
    return (
                {
                    "question": RunnableLambda(itemgetter("question")).with_config(
                        run_name="Itemgetter:question"
                    ),
                    "chat_history": RunnableLambda(serialize_history).with_config(
                        run_name="SerializeHistory"
                    ),
                }
                | _context
                | response_synthesizer
        )

def Get_Conversation_chain(knowledgeBases,temperature,model,question,chat_history):
    knowledgeBase=json.loads(knowledgeBases)
    knowledgeBase=[str(item) for item in knowledgeBase ]
    retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7,"filter":{'knowledge_Store_id': {'$in': (knowledgeBase)}}})
    if model=='gemini-pro':
        llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    else:
        llm = ChatOpenAI(
        model=model,
        streaming=True,
        temperature=float(temperature),)
    answer_chain = create_chain(
        llm,
        retriever,
    )
    answer = answer_chain.invoke( {"question": question, "chat_history":chat_history})

    return answer,refrence_docuemnts_sources
