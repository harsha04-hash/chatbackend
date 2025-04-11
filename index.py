from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
import uuid
import json
import base64

app = Flask(__name__)
CORS(app)

os.environ["GOOGLE_API_KEY"] = "AIzaSyC9O_P4M9OFdofX5pl9Dzk0dSvBiD8dH9A"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
Give the pdf name and page numbers at the end which are related to question (format PDF: and PAGE NUMBERS:)
"""

def get_pdf_text(pdf_docs):
    text_data = []
    print(len(pdf_docs))
    for pdf in pdf_docs:
        # print(pdf) 
        pdf_name = pdf.filename
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text_data.append({
                "text": page.extract_text(),
                "page_number": page_num + 1,
                "pdf_name": pdf_name
            })

    # print(text_data[0])
    return text_data

def get_text_chunks(text_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=150)
    chunks = []
    for data in text_data:
        for chunk in text_splitter.split_text(data["text"]):
            chunks.append({
                "text": chunk,
                "metadata": {
                    "page_number": data["page_number"],
                    "pdf_name": data["pdf_name"]
                }
            })
    return chunks

def get_vector_store(chunks):
    # Define the embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    # Create a FAISS vector store
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    print(vector_store)
    # Serialize the FAISS index to bytes
    faiss_bytes = vector_store.serialize_to_bytes()
    faiss_base64 = base64.b64encode(faiss_bytes).decode('utf-8')

    # Create a JSON-compatible dictionary
    faiss_json = {
        "faiss_index": faiss_base64
    }
    # Convert dictionary to a JSON string
    faiss_json_str = json.dumps(faiss_json, indent=4)
    return faiss_json_str

def ollama_llm(question, context):
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.3)    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(prompt)
    response_text = response.content
    return response_text

def user_input(user_question,faiss_json_str):
    faiss_json = json.loads(faiss_json_str)
    faiss_base64 = faiss_json.get("faiss_index")
    # Decode the Base64 string back to bytes
    faiss_bytes = base64.b64decode(faiss_base64)
    # Deserialize the bytes back to a FAISS object
    # Provide the embeddings used for the original FAISS index creation
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store_deserialized = FAISS.deserialize_from_bytes(faiss_bytes, embeddings=embeddings,allow_dangerous_deserialization = True)

    # Return the deserialized vector store for further use
    print(vector_store_deserialized)
    print("question",user_question)
    docs = vector_store_deserialized.similarity_search_with_score(user_question)
    print(docs)
    context = " ".join([doc[0].page_content for doc in docs])
    response = ollama_llm(user_question, docs)

    return response, context


@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    pdf_files = request.files.getlist('pdf_files')
    if not pdf_files:
        return jsonify({"error": "No PDF files uploaded"}), 400

    raw_text = get_pdf_text(pdf_files)
    # print("Extracted raw text: ", raw_text)
    text_chunks = get_text_chunks(raw_text)
    # print("Generated text chunks: ", text_chunks)
    vector_store = get_vector_store(text_chunks)
    # print("Vector store: ", vector_store)
    # return jsonify({"message": "PDF files processed successfully"}), 200
    return jsonify({"message": "PDF files processed successfully", "embeddings": vector_store}), 200

    # print("Final embeddings: ", embeddings)

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    faiss_json_str = data.get('embeddings')
    session_id = data.get('session_id')
    # print("EMBEDDINGS",embeddings)
    if not question:
        return jsonify({"error": "No question provided"}), 400

    response, context = user_input(question,faiss_json_str)
    return jsonify({"response": response, "context": context, "session_id": session_id}), 200


@app.route('/new_chat', methods=['POST'])
def new_chat():
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id}), 200

if __name__ == '__main__':
    app.run(debug=False)
