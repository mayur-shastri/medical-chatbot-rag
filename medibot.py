# import os
# import re
# import streamlit as st

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint

# DB_FAISS_PATH = "vectorstore/db_faiss"

# @st.cache_resource
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         task="text-generation",
#         temperature=0.5,
#         model_kwargs={"token": HF_TOKEN, "max_length": "512"}
#     )
#     return llm

# def highlight_terms(text, context):
#     """Highlight words from the context that appear in the chatbot response by making them bold."""
#     terms = set(re.findall(r'\b[A-Za-z-]+\b', context))  # Extract words from context
#     for term in sorted(terms, key=len, reverse=True):  # Sort to replace longer terms first
#         pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
#         text = pattern.sub(f'**{term}**', text)  # Make the term bold
#     return text

# def main():
#     st.title("Ask MediBot!")

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     prompt = st.chat_input("Pass your prompt here")

#     if prompt:
#         st.chat_message('user').markdown(prompt)
#         st.session_state.messages.append({'role': 'user', 'content': prompt})

#         CUSTOM_PROMPT_TEMPLATE = """
# Use the information provided in the context to answer the user's question as accurately and comprehensively as possible.
# If the context does not contain sufficient information, say that you don't know—do not generate an answer outside the given context.

# - Provide a well-structured and informative response.
# - Include explanations, possible causes, symptoms, treatments, or precautions, only if the provided context has any, and if relevant.
# - If the question involves a medical condition, include potential next steps a person might consider, such as consulting a specialist.

# Context: {context}
# Question: {question}

# Begin your answer concisely but provide enough detail for clarity.
# """

#         # HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
#         HUGGINGFACE_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#         HF_TOKEN = os.environ.get("HF_TOKEN")

#         try:
#             vectorstore = get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store")
#                 return

#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             response = qa_chain.invoke({'query': prompt})

#             if response:
#                 result = response["result"]
#                 source_documents = response["source_documents"]

#                 # Combine all retrieved document text for better matching
#                 context_text = " ".join([doc.page_content for doc in source_documents])

#                 # Apply highlighting to response
#                 highlighted_result = highlight_terms(result, context_text)

#                 result_to_show = f"**Answer:**\n{highlighted_result}\n\n"

#                 if source_documents:
#                     result_to_show += "**Source Documents:**\n\n"

#                     for i, doc in enumerate(source_documents):
#                         file_path = doc.metadata.get("source", "Unknown Source")
#                         page_number = doc.metadata.get("page", "N/A")

#                         pdf_filename = os.path.basename(file_path)
#                         pdf_url = f"http://localhost:8501/static/{pdf_filename}#page={page_number}"

#                         result_to_show += f"{i+1}. 📖 [Source Document (Page {page_number})]({pdf_url})\n"

#                 st.chat_message('assistant').markdown(result_to_show, unsafe_allow_html=True)
#                 st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()

import os
import re
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model,
                          allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=[
                            "context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm


def highlight_terms(text, context):
    """Highlight words from the context that appear in the chatbot response by making them bold."""
    terms = set(re.findall(r'\b[A-Za-z-]+\b', context)
                )  # Extract words from context
    # Sort to replace longer terms first
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
        text = pattern.sub(f'**{term}**', text)  # Make the term bold
    return text


def main():
    st.title("Ask House M.D - your medical diagnosis assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.user_info = {}

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Step 1: Ask for basic information if not already collected
    if "age" not in st.session_state.user_info:
        st.markdown("### Please provide the following information:")
        st.session_state.user_info['age'] = st.text_input("Age")
        if st.session_state.user_info['age']:
            st.session_state.messages.append(
                {'role': 'user', 'content': f"Age: {st.session_state.user_info['age']}"})

    elif "gender" not in st.session_state.user_info:
        st.session_state.user_info['gender'] = st.selectbox(
            "Gender", ["Male", "Female", "Other"])
        if st.session_state.user_info['gender']:
            st.session_state.messages.append(
                {'role': 'user', 'content': f"Gender: {st.session_state.user_info['gender']}"})

    elif "location" not in st.session_state.user_info:
        st.session_state.user_info['location'] = st.text_input(
            "Where do you live? (City, Country, or State)")
        if st.session_state.user_info['location']:
            st.session_state.messages.append(
                {'role': 'user', 'content': f"Location: {st.session_state.user_info['location']}"})

    elif "ethnicity" not in st.session_state.user_info:
        st.session_state.user_info['ethnicity'] = st.text_input("Ancestry")
        if st.session_state.user_info['ethnicity']:
            st.session_state.messages.append(
                {'role': 'user', 'content': f"Ethnicity: {st.session_state.user_info['ethnicity']}"})

    elif "medical_history" not in st.session_state.user_info:
        st.session_state.user_info['medical_history'] = st.text_area(
            "Do you have any known medical history?")
        if st.session_state.user_info['medical_history']:
            st.session_state.messages.append(
                {'role': 'user', 'content': f"Medical History: {st.session_state.user_info['medical_history']}"})

    elif "medical_query" not in st.session_state.user_info:
        # Once the basic info is collected, show the medical query input box
        prompt = st.text_input("Now, please ask your medical query")

        if prompt:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append(
                {'role': 'user', 'content': prompt})

            # Collect all user information
            user_info = f"""
            Age: {st.session_state.user_info.get('age')}
            Gender: {st.session_state.user_info.get('gender')}
            Location: {st.session_state.user_info.get('location')}
            Ethnicity: {st.session_state.user_info.get('ethnicity')}
            Medical History: {st.session_state.user_info.get('medical_history')}
            """

            CUSTOM_PROMPT_TEMPLATE = """
Use the information provided in the context to answer the user's question as accurately and comprehensively as possible.
If the context does not contain sufficient information, say that you don't know—do not generate an answer outside the given context.

- Provide a well-structured and informative response.
- Include explanations, possible causes, symptoms, treatments, or precautions, only if the provided context has any, and if relevant.
- If the question involves a medical condition, include potential next steps a person might consider, such as consulting a specialist.

Context: {context}  
Question: {question}

If the user question contains inappropriate manguage, respond with : I cannot assist with that" 

Begin your answer concisely but provide enough detail for clarity.

Example User Information :
Age: 62
Gender: Female
Location: India
Ancestry: Indian/Asian
Medical History: Had a heart attack at the age of 56, replaced heart valve.
"""         
    
            # CUSTOM_PROMPT_TEMPLATE += user_info

            # HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
            HUGGINGFACE_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            HF_TOKEN = os.environ.get("HF_TOKEN")

            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(
                        huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={
                        'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke(
                    {'query': prompt, 'user_info': user_info})

                if response:
                    result = response["result"]
                    source_documents = response["source_documents"]

                    # Combine all retrieved document text for better matching
                    context_text = " ".join(
                        [doc.page_content for doc in source_documents])

                    # Apply highlighting to response
                    highlighted_result = highlight_terms(result, context_text)

                    result_to_show = f"**Answer:**\n{highlighted_result}\n\n"

                    if source_documents:
                        result_to_show += "**Source Documents:**\n\n"

                        for i, doc in enumerate(source_documents):
                            file_path = doc.metadata.get(
                                "source", "Unknown Source")
                            page_number = doc.metadata.get("page", "N/A")

                            pdf_filename = os.path.basename(file_path)
                            pdf_url = f"http://localhost:8501/static/{pdf_filename}#page={page_number}"

                            result_to_show += f"{i+1}. 📖 [Source Document (Page {page_number})]({pdf_url})\n"

                    st.chat_message('assistant').markdown(
                        result_to_show, unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {'role': 'assistant', 'content': result_to_show})

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()