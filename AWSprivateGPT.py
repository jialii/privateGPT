#!/usr/bin/env python3
# from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, HuggingFaceEndpoint, HuggingFacePipeline
from nomic.gpt4all import GPT4AllGPU
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
    AutoModel
)
import transformers
import os, torch
import argparse
import time
import json
import nltk
import logging
from auto_gptq import AutoGPTQForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# load_dotenv()
nltk.download('punkt')

embeddings_model_name = 'all-MiniLM-L6-v2'
persist_directory = 'db'

model_type = 'Falcon'
model_path = 'models/ggml-gpt4all-j-v1.3-groovy.bin'
model_n_ctx = 1000
model_n_batch = 8
target_source_chunks = 4


from constants import CHROMA_SETTINGS


def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """

    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        # The code supports all huggingface models that ends with GPTQ and have some variation
        # of .no-act.order or .safetensors in their HF repo.
        logging.info("Using AutoGPTQForCausalLM for quantized models")

        if ".safetensors" in model_basename:
            # Remove the ".safetensors" ending if present
            model_basename = model_basename.replace(".safetensors", "")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        logging.info("Tokenizer loaded")

        model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=False,
            quantize_config=None,
        )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm

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
    # past_queries=json.load(open("source_documents/past_queries.json"))
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
                huggingfacehub_api_token=os.environ.get(""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""),
                endpoint_url= "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct" ,
                # huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                task="text-generation",
                model_kwargs = {
                    "min_length": 200,
                    "max_length":2000,
                    "temperature":0.5,
                    "max_new_tokens":200,
                    "num_return_sequences":1
                },            
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

        case "TheBloke":
            model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
            model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
            llm = load_model("cpu", model_id=model_id, model_basename=model_basename)
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
        # past_queries[query]=answer
        # with open("source_documents/past_queries.json", 'w') as json_file:
        #     json.dump(past_queries, json_file, 
        #                 indent=4,  
        #                 separators=(',',': '))
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
