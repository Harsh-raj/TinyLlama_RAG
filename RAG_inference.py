import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

document = SimpleDirectoryReader("sft/dataset").load_data()

from llama_index.core import PromptTemplate

system_prompt = "You are a QA bot. Given the questions answer it correctly."

query_wrapper_prompt = PromptTemplate("<|user|>:{query_str}\n<|assistant|>:")

llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature":0.0, "do_sample":False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="cuda",
    model_kwargs={"torch_dtype":torch.bfloat16},
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(document, service_context = service_context)

query_engine = index.as_query_engine()


# Defining a custom stopping criteria class for the model's text generation.
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]  # IDs of tokens where the generation should stop.
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:  # Checking if the last generated token is a stop token.
                return True
        return False

# Function to generate model predictions.
def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    # Formatting the input for the model.
    messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                        for item in history_transformer_format])

    model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.5,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # Starting the generation in a separate thread.
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if '</s>' in partial_message:  # Breaking the loop if the stop token is generated.
            break
        yield partial_message


def predict(input, history):
    response = query_engine.query(input)
    return str(response)

gr.ChatInterface(predict).launch(share=True)
# # Loading the tokenizer and model from Hugging Face's model hub.
# tokenizer = AutoTokenizer.from_pretrained("output/1T_FT_lr1e-5_ep5_top1_2024-03-04/checkpoint-575")
# model = AutoModelForCausalLM.from_pretrained("output/1T_FT_lr1e-5_ep5_top1_2024-03-04/checkpoint-575")

# # using CUDA for an optimal experience
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# # Defining a custom stopping criteria class for the model's text generation.
# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         stop_ids = [2]  # IDs of tokens where the generation should stop.
#         for stop_id in stop_ids:
#             if input_ids[0][-1] == stop_id:  # Checking if the last generated token is a stop token.
#                 return True
#         return False

# # Function to generate model predictions.
# def predict(message, history):
#     history_transformer_format = history + [[message, ""]]
#     stop = StopOnTokens()

#     # Formatting the input for the model.
#     messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
#                         for item in history_transformer_format])

#     model_inputs = tokenizer([messages], return_tensors="pt").to(device)
#     streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
#     generate_kwargs = dict(
#         model_inputs,
#         streamer=streamer,
#         max_new_tokens=1024,
#         do_sample=True,
#         top_p=0.95,
#         top_k=50,
#         temperature=0.5,
#         num_beams=1,
#         stopping_criteria=StoppingCriteriaList([stop])
#     )
#     t = Thread(target=model.generate, kwargs=generate_kwargs)
#     t.start()  # Starting the generation in a separate thread.
#     partial_message = ""
#     for new_token in streamer:
#         partial_message += new_token
#         if '</s>' in partial_message:  # Breaking the loop if the stop token is generated.
#             break
#         yield partial_message


# # Setting up the Gradio chat interface.
# gr.ChatInterface(predict,
#                  title="Tinyllama_chatBot",
#                  description="Ask Tiny llama any questions",
#                  examples=['How to cook a fish?', 'Who is the president of US now?']
#                  ).launch(share=True)  # Launching the web interface.