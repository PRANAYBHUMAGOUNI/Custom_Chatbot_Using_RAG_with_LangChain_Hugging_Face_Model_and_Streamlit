import os
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from dotenv import load_dotenv
import torch

# Load environment variables (if any)
load_dotenv()

# Step 1: Setup the Local LLM
def load_llm():
    torch.manual_seed(42)
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",  
        tokenizer="google/flan-t5-large",
        device=-1  
    )
    return llm

# Step 2: Set up the custom prompt template
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Updated Prompt Template to be generic
CUSTOM_PROMPT_TEMPLATE = """
You are a knowledgeable and detailed assistant who provides comprehensive answers based solely on the provided context.
If the context does not contain the answer, state that you do not have information on that topic within your current context.
Do not use any prior knowledge or make up answers.

Context:
{context}

Question:
{question}

Provide a detailed and informative answer:
"""

# Step 3: Load the vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Custom retrieval function with similarity threshold
def retrieve_relevant_documents(vectorstore, query, similarity_threshold=0.5):
    # Encode the query
    query_embedding = vectorstore.embedding_function.embed_query(query)
    # Perform similarity search
    docs_and_scores = vectorstore.similarity_search_with_score_by_vector(
        query_embedding, k=3
    )
    # Filter documents based on similarity threshold
    relevant_docs = []
    for doc, score in docs_and_scores:
        # For unit-normalized embeddings, distance ranges from 0 to 2
        similarity = 1 - (score / 2)
        # print(f"Document: {doc.metadata.get('source', 'Unknown')}, Distance: {score}, Similarity: {similarity}")
        if similarity >= similarity_threshold:
            relevant_docs.append(doc)
    return relevant_docs

# Step 4: Generate the answer using the LLM and retrieved documents
def generate_answer(question):
    # Retrieve relevant documents with similarity threshold
    relevant_docs = retrieve_relevant_documents(db, question, similarity_threshold=0.5)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Check if relevant_docs is empty
    if not relevant_docs:
        answer = "I'm sorry, but I do not have information on that topic within my current context."
    else:
        # Prepare the prompt
        prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context.strip(), question=question.strip())

        # Generate the answer using the LLM
        llm = load_llm()
        output = llm(
            prompt,
            max_new_tokens=1024,  
            do_sample=False,      
            clean_up_tokenization_spaces=True
        )
        answer = output[0]['generated_text']

    return answer, relevant_docs

# Now invoke with a single query
if __name__ == "__main__":
    user_query = input("Write your query here: ")
    result, source_documents = generate_answer(user_query)

    print("\nRESULT:")
    print(result)
    print("\nSOURCE DOCUMENTS:")
    if source_documents:
        for doc in source_documents:
            source = doc.metadata.get('source', 'Unknown')
            page_number = doc.metadata.get('page', 'Unknown')
            print(f"- {source}, Page {page_number}")
    else:

        print("No relevant documents found.")

