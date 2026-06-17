from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.vectorstore import get_vectorstore
from rag.model import get_llm

SYSTEM_PROMPT = (
    "You are a helpful educational assistant for students. "
    "Use the context below (extracted from study materials uploaded by faculty) to answer the student's question. "
    "If the answer is not present in the context, say \"I don't have enough information in the uploaded study materials to answer this.\"\n\n"
    "Context:\n{context}"
)

# prompt template: injects retrieved context and student's question before sending to LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

def chat(question: str, n_results: int = 5) -> dict:
    # step 1: find the top N most relevant chunks from Pinecone
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": n_results})
    docs = retriever.invoke(question)

    if not docs:
        return {
            "question": question,
            "answer": "No study materials have been indexed yet. Please ask faculty to upload materials.",
            "sources": [],
        }

    # step 2: combine all retrieved chunks into one context string
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # step 3: pipe context + question through prompt → LLM → plain text output
    chain = prompt | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # step 4: build source list from chunk metadata for display in the UI
    sources = [
        {
            "title": d.metadata.get("title", "Unknown"),
            "subject": d.metadata.get("subject", "Unknown"),
        }
        for d in docs
    ]

    return {"question": question, "answer": answer, "sources": sources}
