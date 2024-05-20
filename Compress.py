import torch
import numpy
import tensorflow as tf
import numpy as np
from transformers import TextDataset, DataCollatorForLanguageModeling
#from transformers import GPT2Tokenizer, GPT2LMHeadModel
#from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import re
from PyPDF2 import PdfReader
import os
import docx
import sys
import timeit
import statistics
stored_dict={}

def run_test(model,tokenizer, test, sequence):
  model=model
  tokenizer=tokenizer
  test_ids=tokenizer.encode(test)
  sequence_ids=tokenizer.encode(sequence)
  right_ids=test_ids[1000:]
  
  #padding_kwargs = {"pad_to_multiple_of": 8, "padding": True}
  xla_generate = tf.function(model, jit_compile=True)
  output_string=''
  for i in range(len(right_ids)):
      inputs=tokenizer(f'{sequence}',return_tensors="tf")
      with torch.no_grad():
        logits=xla_generate(**inputs).logits[:,-1,:]
        topk = tf.argsort(logits,axis=-1,direction='DESCENDING',stable=False).numpy().reshape(-1).tolist()
        right_id=right_ids[i]
      if right_id in topk:
          output_string+=str(topk.index(right_id))+'.'
      sequence=sequence+tokenizer.decode(right_id)
      sequence=sequence[-1000:]
  return output_string

def read_pdf(path):
  with open(path,"rb") as file:
    pdf_reader=PdfReader(file)
    text=""
    for i in range(len(pdf_reader.pages)):
      text+=pdf_reader.pages[i].extract_text()
    text=text.replace('\n',' ')
    return text
def extensive_test(filename,model_path):
  global store_dict
  print('LOADING MODEL AFRESH')
  model=TFAutoModelForCausalLM.from_pretrained(model_path)
  tokenizer=AutoTokenizer.from_pretrained(model_path)
  
  test=read_pdf(filename)
  sequence=test[:1000]
  start=timeit.default_timer()
  encoded=run_test(model,tokenizer,test,sequence)
  time_taken=timeit.default_timer()-start
  ratio=sys.getsizeof(encoded)/len(test[1000:])
  net_compression =gzip.compress(encoded,'utf-8')
  net_size=sys.getsizeof(net_compression)
  only_gzip=gzip.compress(sequence,'utf-8')
  gzip_size=sys.getsizeof(only_gzip)
  print('\n','.'*20,'STATISTICS','.'*20)
  print('Length of input',len(test[1000:]))
  print('Transformer only',ratio)
  print('only gzip',gzip_size)
  print('net size',net_size)
  print('time taken',time_taken)
  print('.'*62)
 
#EXECUTION
extensive_test('/home/ee22b149/25Cardiology/African-Americans-have-worse-outcomes-after-transcatheter-and-_2024_Journal-.pdf',"gpt2")
