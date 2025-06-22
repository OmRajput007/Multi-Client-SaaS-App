# Importing all the tools we need 

import streamlit as st  # This makes our website look pretty.
import os  # This helps us read secret passwrords from computer.
import stripe  # This handles money payments.
from dotenv import load_dotenv  # This reads our secret file
from langchain.document_loaders import PyPDFLoader  # This reads our PDF files
from langchain.text_splitter import CharacterTextSplitter  # This cuts texts into small pieces
from langchain.embeddings import OpenAIEmbeddings  # This turns words into numbers
from langchain.vectorstores import FAISS  # This stores our word numbers in a smart way
from langchain.chains import RetrievalQA  # This finds answers to questions
import tempfile # This creates temporary file which disappears after use
from langchain.llms import OpenAI  # Add this import

load_dotenv()
# Load our secret passwords from .env file

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
# Get the secret key for stripe payments

st.set_page_config(page_title = "Multi Client SaaS", page_icon="📄", layout = "centered")
st.title("Multi CLient SaaS Build")
# Make the website look nice with a title and an icon.

user_id = st.text_input("Enter your email")
# Ask user to enter their email


# This specific email gets free access
if user_id == "omsingh15om@gmail.com":
    is_premium = True
    st.session_state[f"{user_id}_premium"] = True  # Remember they are premium
else:
    is_premium = st.session_state.get(f"{user_id}_premium", False)


# Show different buttons based on if they are premium or not
if is_premium:
    st.success("Premium User Access Granted")
else:
    # Show demo button to get premium for testing
    if st.button("Get Demo Premium Access"):
        st.session_state[f"{user_id}_premium"] = True  # Make them premium
        st.rerun()  # Refresh the page to show changes


# Show payments button only if they are not premium and typed their email
if not is_premium and user_id:
    st.button("Buy Premium - $9.99/month")
    try :
        # Create a payment page with stripe
        checkout_session = stripe.checkout.Session.create(
                payment_method_types=['card'],  # Accept credit cards
                line_items=[{
                    'price': 'price_1RbcdDSALnciXEDIGCfqBQt2',  # The price ID from Stripe
                    'quantity': 1,  # Buy 1 subscription
                }],
                mode='subscription',  # Monthly subscription
                success_url='https://yourdomain.com/success',  # Where to go after payment
                cancel_url='https://yourdomain.com/cancel',  # Where to go if they cancel
                metadata={'user_id': user_id}  # Remember who is paying
            )
        # Show a link to complete payment
        st.markdown(f"[Complete Payment]({checkout_session.url})")
    except Exception as e:
        # If payment fails, then show error and demo button
        st.error("Payment temporarily unavailable. Contact admin for access")
        if st.button("Grant Premimum Access (Demo)"):
            st.session_state[f"{user_id}_premium"] = True
            st.rerun()


# File upload area
uploaded_file = st.file_uploader("Upload your PDF file", type = ["PDF"])

# Text box where users can ask questions
query = st.text_input("Ask questions : ")


# MAIN LOGIC = What happens when user has entered his email and now asks a questions
if query and user_id:
    if uploaded_file:
        if not is_premium:
            # If the user has entered the email and question but he is not premium
            # Then show this message :
            st.warning("⚠️ Please upgrade to Premium to process files and get answers.")
        else:
            # If they do have premium access, show that we are working on their request
            st.info("💬 Processing your file with GPT...")


            # Create a temporary file, and store the value of our pdf in it.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                tmpfile.write(uploaded_file.getvalue())  # Copying the uploaded file
                tmp_file_path = tmpfile.name  # Remember where we put it

            # Read the PDF file
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()  # Get all the text from the PDF

            # Cutting the text into small pieces
            text_splitter = CharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
            docs = text_splitter.split_documents(documents)

            # Turn all the text pieces into numbers that AI can understand.
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Save their data in their own folder
            save_path = f"dbs/{user_id}_db"
            vectorstore.save_local(save_path)

            # Create a smart question answering system
            qa_chain = RetrievalQA.from_chain_type(OpenAI(), retriever=vectorstore.as_retriever())
            answer = qa_chain.run(query) # Ask the AI the question

            # Show the answer to the user
            st.success("Answer:")
            st.write(answer)

            # Deleting the temporary file
            os.unlink(tmp_file_path)
    else:
        # If they didn't upload a file, remind them to do so
        st.info("📄 Please upload a PDF file to proceed.")
