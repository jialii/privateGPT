#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, HuggingFaceEndpoint, HuggingFacePipeline
from nomic.gpt4all import GPT4AllGPU
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import transformers
import os, torch
import argparse
import time
import json
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()
nltk.download('punkt')

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS


def find_similar_questions(document, query):
    # Tokenize the document and query into sentences
    doc_sentences = nltk.sent_tokenize(document)
    query_sentences = nltk.sent_tokenize(query)
    
    # Combine document and query sentences
    sentences = doc_sentences + query_sentences
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Calculate TF-IDF matrix for sentences
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate cosine similarity between query sentences and document sentences
    similarity_matrix = cosine_similarity(tfidf_matrix[-len(query_sentences):], tfidf_matrix[:-len(query_sentences)])
    
    # Find the most similar question indices
    similar_indices = similarity_matrix.argmax(axis=1)
    
    # Get the similar questions
    similar_questions = [doc_sentences[index] for index in similar_indices]
    
    return similar_questions

def main():
    # Parse the command line arguments
    past_queries=json.load(open("source_documents/past_queries.json"))
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type: 
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "Falcon":
            llm = HuggingFaceEndpoint(
                huggingfacehub_api_token=os.environ.get("HUGGINGFACE_API_TOKEN"),
                endpoint_url= "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct" ,
                # huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                task="text-generation",
                model_kwargs = {
                    "min_length": 200,
                    "max_length":2000,
                    "temperature":0.5,
                    "max_new_tokens":200,
                    "num_return_sequences":1
            }
        )
        case "LocalFalcon":
            # model_name = "tiiuae/falcon-7b-instruct"
            # model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)

            # tokenizer = AutoTokenizer.from_pretrained(model_name)
            # tokenizer.save_pretrained('local_falcon')

            # model.save_pretrained('local_falcon')
            model2 = AutoModelForCausalLM.from_pretrained('/Users/jiali.xu/Desktop/falcon-7b-instruct',trust_remote_code=True)
            tokenizer2 = AutoTokenizer.from_pretrained('/Users/jiali.xu/Desktop/falcon-7b-instruct')
            model_kwargs = {
                    "min_length": 200,
                    "max_length":2000,
                    "temperature":0.5,
                    "max_new_tokens":200,
                    "num_return_sequences":1
            }
            pipeline = transformers.pipeline(
                "text-generation",
                model=model2,
                tokenizer=tokenizer2,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                max_new_tokens=500
            )

            llm = HuggingFacePipeline(pipeline=pipeline)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    # while True:
    # query = input("\nEnter a query: ")
    #     with open('source_documents/queries.txt') as f:
    #         for line in f:
    #             query = line
    #             if query == "exit":
    #                 break
    #             if query.strip() == "":
    #                 continue

    #     # Get the answer from the chain
    # start = time.time()
    #             f = open("source_documents/all_queries.txt", "r")
    #             similar_questions = find_similar_questions(f.read(), query)
    #             # check = input('Is this the question you are looking for? ' + similar_questions[0] + '(y/N)')
    #             # if check == 'y':
    #             if similar_questions[0] == query:
    #                 answer = past_queries[similar_questions[0]+'\n']
                            
    #             else:     
    while True:    
        query = input("\nEnter a query: ") 
        if query == "exit":
            break
        if query.strip() == "":
            continue
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        past_queries[query]=answer
        with open("source_documents/past_queries.json", 'w') as json_file:
            json.dump(past_queries, json_file, 
                        indent=4,  
                        separators=(',',': '))
        end = time.time()

    # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
