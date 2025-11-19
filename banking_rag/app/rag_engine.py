import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

# Setup Vector DB Path
DB_PATH = Path(__file__).parent.parent / "banking_db"

def process_banking_doc(file_path: str):
    """
    Ingests a Banking Policy PDF.
    Splits by paragraph to keep loan clauses intact.
    """
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # 2. Split (Banking docs have long lists, so we use bigger chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    # 3. Index
    # FIX: "embedding" is usually fine in from_documents, but being explicit helps
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(), 
        persist_directory=str(DB_PATH)
    )
    return len(splits)

def get_compliance_answer(query: str):
    """
    Retrieves loan rules and checks compliance.
    """
    # FIX: Changed 'embedding' to 'embedding_function' here!
    vectorstore = Chroma(
        persist_directory=str(DB_PATH), 
        embedding_function=OpenAIEmbeddings()
    )
    
    # Search for the top 5 most relevant policy clauses
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. The "Bank Auditor" System Prompt
    system_prompt = (
        "You are a Senior Credit Risk Auditor for a bank. "
        "Answer the user's question based ONLY on the retrieved policy documents below. "
        "Rules:\n"
        "1. If the policy does not explicitly mention the criteria, say 'Policy not found'.\n"
        "2. Do not make assumptions about interest rates or eligibility.\n"
        "3. Quote the exact policy section or page number in brackets [Page X].\n"
        "\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    
    response = chain.invoke({"input": query})
    
    # Extract unique sources
    sources = list(set([doc.metadata.get('page', 0) for doc in response['context']]))
    
    return {
        "auditor_response": response['answer'],
        "policy_pages_referenced": sources
    }