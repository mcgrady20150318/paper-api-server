from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
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
from langchain.vectorstores.redis import Redis
import redis

app = Flask(__name__)
CORS(app)

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_BASE'] = os.getenv('OPENAI_API_BASE')

embeddings = OpenAIEmbeddings()

prompt_template = """你现在是一个人工智能学者，请根据以下内容。
    {context}，
    回答问题: {question}
    中文答案是:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
rdx = redis.from_url(os.getenv('REDIS_URL'))

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.tokens = []
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
    vector_index = Redis.from_existing_index(embeddings,index_name=id,redis_url=os.getenv('REDIS_URL'),schema="redis_schema.yaml")
    retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 1})
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

def sum(abstract):
    handler = ChainStreamHandler()
    llm = OpenAI(max_tokens=1000,streaming=True,callback_manager=CallbackManager([handler]))
    thread = threading.Thread(target=async_sum, args=(llm, abstract))
    thread.start()
    return handler.generate_tokens()

def async_sum(llm,abstract):
    context = '''给定论文摘要:''' + abstract + '''请用200字总结本文的研究并提出3个引导阅读的问题.'''
    llm(context)

def check_id_redis(id):
    id_list = rdx.lrange('cached_ids',0,-1)
    if id.encode('ascii') in id_list:
        return True
    else:
        return False

@app.route('/', methods=['GET'])
def _index():
    return 'hello qa'

@app.route('/r', methods=['POST'])
def r():
    id = request.json.get('id')
    print(id)
    try:
        if check_id_redis(id):
            return jsonify({"result":'true'})
        else:
            return jsonify({"result":'false'})
    except:
        return jsonify({"result":'error'})

@app.route('/s', methods=['POST'])
def s():
    abstract = request.json.get('abstract')
    print(abstract)
    try:
        return Response(sum(abstract), mimetype='text/plain')
    except:
        return Response('error', mimetype='text/plain')

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
