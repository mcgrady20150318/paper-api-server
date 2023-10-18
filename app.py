from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.output import LLMResult
from typing import Any
import threading
from langchain.vectorstores import Pinecone
import pinecone

app = Flask(__name__)
CORS(app)

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_BASE'] = os.getenv('OPENAI_API_BASE')

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)

embeddings = OpenAIEmbeddings()
index_name = 'qaoverpaper'
vector_index = Pinecone.from_existing_index(index_name, embeddings)

prompt_template = """你现在是一个人工智能学者，请根据以下内容。
    {context}，
    回答问题: {question}
    中文答案是:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.tokens = []
        # 记得结束后这里置true
        self.finish = False

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="")
        self.tokens.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.finish = 1

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        print(str(error))
        self.tokens.append(str(error))

    def generate_tokens(self):
        while not self.finish or self.tokens:
            if self.tokens:
                data = self.tokens.pop(0)
                yield data
            else:
                pass

def qa(query,id):
    handler = ChainStreamHandler()
    retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 1,"filter":{'source':'./%s/%s.pdf' % (id,id)}})
    chain_type_kwargs = {"prompt": PROMPT}
    qa_interface = RetrievalQA.from_chain_type(
        llm = OpenAI(max_tokens=1000,streaming=True,callback_manager=CallbackManager([handler])),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs
    )
    thread = threading.Thread(target=async_run, args=(qa_interface, query))
    thread.start()
    return handler.generate_tokens()

def async_run(qa_interface,query):
    qa_interface({"query": query}, return_only_outputs=True)

@app.route('/', methods=['GET'])
def _index():
    return 'hello world'

@app.route('/q', methods=['POST'])
def q():
    id = request.json.get('id')
    query = request.json.get('query')
    print(id,query)
    try:
        return Response(qa(query,id), mimetype='text/plain')
    except:
        return Response('error', mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000), host='0.0.0.0')
