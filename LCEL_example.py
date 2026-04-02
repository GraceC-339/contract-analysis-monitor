from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

prompt = ChatPromptTemplate.from_template(
    "Answer based on context:\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(x["documents"])
    )
    | prompt
    | llm  # Your LLM instance
    | StrOutputParser()
)

result = chain.invoke({
    "documents": [Document(page_content="Sample context")],
    "question": "Your question?"
})

# This chains document formatting, prompting, LLM call, and parsing.