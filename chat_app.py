from flask import Flask, render_template, jsonify, request, Blueprint,redirect,url_for,session
from Vivi_AI.helper import embeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from Vivi_AI.logger import logging
from dotenv import load_dotenv
from Vivi_AI.helper import load_local
from auth import auth_bp,oauth
import os

app = Flask(__name__)
chatbot_bp = Blueprint('chatbot', __name__)
load_dotenv()
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
persisted_vectorstore = load_local()
oauth.init_app(app)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_API_KEY']=LANGCHAIN_API_KEY
os.environ['OPENAI_API_KEY']=OPENAI_API_KEY



@chatbot_bp.route('/')
def index():
    app.logger.info('Index route called')
    # if 'google_token' in session or 'user' in session:
    #     app.logger.info('User authenticated, rendering index.html')
    return render_template("index.html")
    # app.logger.info('No authenticated user found, redirecting to login')
    # return redirect(url_for('auth.login'))
    

@chatbot_bp.route('/get', methods=['POST'])
def chat():
    try:
        msg = request.json['msg']
        if not msg:
            raise ValueError("Empty user query received.")
        
        logging.info(f"Message is {msg}")
       
        prompt_template = """
            Use the following pieces of information to answer the user's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Context: {context}
            Question: {question}
            Only return the helpful answer below and nothing else.
            Helpful answer:
        """
        result = rag_chain(query=msg, prompt_template=prompt_template)
        logging.info(f"LLM response is {result}")
        
        if not result:
            raise ValueError("Empty result received from rag_chain.")
        
        return jsonify({'response': result})
    except ValueError as ve:
        app.logger.warning(f"Value error in chat route: {str(ve)}")
        return jsonify({'response': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Error in chat route: {str(e)}")
        return jsonify({'response': "I'm sorry, but I encountered an error. Please try again."}), 500

def rag_chain(query, prompt_template):
    app.logger.debug(f"Entering rag_chain with query: {query}")
    logging.info(f"Query is {query}")
    try:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )
        app.logger.debug("PromptTemplate created")
       
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(api_key=OPENAI_API_KEY),
            chain_type='stuff',
            retriever=persisted_vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        app.logger.debug("RetrievalQA created")
       
        result = qa.run(query)
        logging.info(f"Result is {result}")
        app.logger.debug(f"Query result: {result}")
        return result
    except Exception as e:
        app.logger.error(f"Error in rag_chain: {str(e)}")
        raise e
    
@chatbot_bp.route('/log', methods=['POST'])
def log_client_message():
    data = request.json
    logging.info(f"Client log: {data['message']}")
    return jsonify({"status": "success"}), 200

app.register_blueprint(chatbot_bp, url_prefix='/chatbot')


