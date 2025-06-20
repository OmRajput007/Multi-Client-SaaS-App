import streamlit as st
import os
import stripe
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import tempfile

load_dotenv()
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

def process_user_pdf(uploaded_file, query, user_id, is_premium=False):
    max_size = 10 * 1024 * 1024 if is_premium else 2 * 1024 * 1024  # 10MB vs 2MB
    if uploaded_file.size > max_size:
        return "File too large. Upgrade to Premium for larger files."
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    chunk_size = 2000 if is_premium else 1000
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    save_path = f"dbs/{user_id}_db"
    vectorstore.save_local(save_path)
    
    qa_chain = RetrievalQA.from_chain_type(OpenAI(), retriever=vectorstore.as_retriever())
    answer = qa_chain.run(query)
    
    os.unlink(tmp_file_path)
    return answer

st.title("Multi Client SaaS Build")
user_id = st.text_input("Enter your email")

is_premium = st.session_state.get(f"{user_id}_premium", False)

if not is_premium and user_id:
    if st.button("Buy Premium - $9.99/month"):
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': 'price_1Rbcp3SALnciXEDISuaAR4kM',  # Your test mode price ID
                'quantity': 1,
            }],
            mode='subscription',
            success_url='https://yourdomain.com/success',
            cancel_url='https://yourdomain.com/cancel',
            metadata={'user_id': user_id}
        )
        st.markdown(f"[Complete Payment]({checkout_session.url})")

if is_premium:
    st.success("Premium User ✅")

uploaded_file = st.file_uploader("Upload your file")
query = st.text_input("Ask questions:")

if query and user_id:
    if uploaded_file:
        if not is_premium:
            st.warning("⚠️ Please upgrade to Premium to process files and get answers.")
        else:
            st.write("processing...")
            result = process_user_pdf(uploaded_file, query, user_id, is_premium)
            st.write(result)
    else:
        st.info("Please upload a PDF file")