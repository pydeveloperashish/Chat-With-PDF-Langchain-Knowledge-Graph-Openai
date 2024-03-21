from dotenv import load_dotenv, find_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

load_dotenv(find_dotenv())
llm = OpenAI(openai_api_key = os.getenv("OPENAI_API_KEY"))


def main():
    load_dotenv()
    st.set_page_config(page_title = "Ask your PDF")
    st.header("Ask your PDF")
    
    # Uploading the PDF file
    pdf = st.file_uploader("Upload Your PDF", type = "pdf")

    # Extracting the PDF file
    # PdfReader doesnt allow you to extract all the text from the PDF directly, but it allows you to
    # take the pages and then extract text from those pages.
    if pdf:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)
        
        # Embed the chunks
        embeddings = OpenAIEmbeddings(openai_api_key = os.getenv("OPENAI_API_KEY"))
        # Store in FAISS database.
        knowledgeBase = FAISS.from_texts(chunks, embeddings)

        # Show user input
        query = st.text_input("Ask a question about your PDF")
        if query:
            docs = knowledgeBase.similarity_search(query)
            
  
            # Now using Chain
            chain = load_qa_chain(llm)
            response = chain.run(input_documents = docs, question = query)
            st.write(response)
            
            
if __name__ == '__main__':
    main()