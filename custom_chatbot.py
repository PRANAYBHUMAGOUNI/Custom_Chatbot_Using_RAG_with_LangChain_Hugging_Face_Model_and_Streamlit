
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from dotenv import load_dotenv
import torch

# Load environment variables (if any)
load_dotenv()

# Set page config for Streamlit
st.set_page_config(
    page_title="Custom Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Apply custom CSS (using a workaround)
def apply_custom_css():
    st.markdown(
        """
        <style>
        /* Hide the hamburger menu */
        .css-1siy2j7 { display: none !important; }

        /* Hide the "Made with Streamlit" footer */
        .css-1lsmgbg.egzxvld1 { display: none !important; }

        /* Chat message styling */
        .stChatMessage {
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 16px;
        }
        .stChatMessage.user {
            background-color: #DCF8C6;
            align-self: flex-end;
        }
        .stChatMessage.assistant {
            background-color: #FFFFFF;
            align-self: flex-start;
        }

        /* Chat input styling */
        .stChatInput textarea {
            padding: 12px;
            border-radius: 8px;
        }

        /* Adjust the main content area */
        .main .block-container {
            max-width: 800px;
            padding-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Step 1: Set up the LLM
def load_llm():
    torch.manual_seed(42)
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        device=-1,  # Set to -1 for CPU or 0 for GPU
    )
    return llm

# Step 2: Set up the custom prompt template
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
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

# Step 3: Cache the vector store to avoid reloading
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        "vectorstore/db_faiss",
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return db

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
        if similarity >= similarity_threshold:
            relevant_docs.append(doc)
    return relevant_docs

# Step 4: Define the main function for the Streamlit app
def main():
    apply_custom_css()

    st.title("🤖 Custom Chatbot")

    # Sidebar content using Streamlit's methods
    with st.sidebar:
        st.header("✨ Welcome to Custom Chatbot!")
        st.write(
            "Custom Chatbot is your intelligent assistant, ready to answer your questions based on the documents provided."
        )
        st.markdown("---")
        st.subheader("🔍 How to Use")
        st.markdown(
            """
            - **Ask a Question**: Type your query in the input box below.
            - **Get an Answer**: The assistant will provide an informative response.
            - **View Sources**: Expand the "Source Documents" section to see where the information came from.
            """
        )        

    # Initialize session state to store messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(message['content'])
        else:
            with st.chat_message("assistant"):
                st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input("Type your message here..."):
        # Display user's message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Response placeholder
        response_placeholder = st.empty()

        # Add a spinner while generating the response
        with st.spinner("The assistant is typing..."):
            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store.")
                    return

                # Retrieve relevant documents with similarity threshold
                relevant_docs = retrieve_relevant_documents(vectorstore, prompt, similarity_threshold=0.5)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Check if relevant_docs is empty
                if not relevant_docs:
                    result = "I'm sorry, but I do not have information on that topic within my current context."
                else:
                    # Prepare the prompt
                    prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
                    full_prompt = prompt_template.format(context=context.strip(), question=prompt.strip())

                    # Generate the answer
                    llm = load_llm()
                    output = llm(
                        full_prompt,
                        max_new_tokens=1024,
                        do_sample=False,
                        clean_up_tokenization_spaces=True,
                    )
                    result = output[0]['generated_text']

                # Display assistant's response
                with st.chat_message("assistant"):
                    st.markdown(result)
                # Add assistant's response to session state
                st.session_state.messages.append({"role": "assistant", "content": result})

                # Optionally, display source documents
                with st.expander("Source Documents"):
                    if relevant_docs:
                        for doc in relevant_docs:
                            source = doc.metadata.get('source', 'Unknown')
                            page_number = doc.metadata.get('page', 'Unknown')
                            st.write(f"- **{source}**, Page {page_number}")
                    else:
                        st.write("No relevant documents found.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Run the app
if __name__ == "__main__":

    main()


