import torch
import numpy
import tensorflow as tf
import numpy as np
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import re
from PyPDF2 import PdfReader
import os
import docx
import sys
import timeit
import statistics
import gzip
import brotli

#Hyperparameters
input_length=10000
context_size=100

def run_test(model,tokenizer, test, sequence):
  global context_size
  model=model
  tokenizer=tokenizer
  model.config.pad_token_id = model.config.eos_token_id
  test_ids=tokenizer.encode(test)                            #Tokenizing the input text
  sequence_ids=tokenizer.encode(sequence)                    
  length=len(sequence_ids)
  right_ids=test_ids[length:]                                #This list will be used for rank prediction based on comparison
  xla_generate = tf.function(model, jit_compile=True)        #COnverting the model into JIT compiled XLA graph
  output_string=''
  print('STARTING////////')
  for i in range(len(right_ids)):
   inputs=tokenizer(f'{sequence}',return_tensors="tf")
   logits=xla_generate(**inputs).logits[:,-1,:]              #Generating the probability distribution
   topk = tf.argsort(logits,axis=-1,direction='DESCENDING',stable=False).numpy().reshape(-1).tolist() 
   right_id=right_ids[i]
   rank=topk.index(right_id)                                 #Rank Prediction
   output_string+=str(rank)+'.'
   inputs=tokenizer(f'{sequence}',return_tensors="tf")
   sequence+=tokenizer.decode(right_id)                      #Context window moved forward
   sequence=sequence[-context_size:]
  print(output_string)
  #print(output_string.split('.').count('0'), len(right_ids))
  return output_string

def read_pdf(path):                                          #This function can be used when compressing the PDF rather than a text file 
  with open(path,"rb") as file:
    pdf_reader=PdfReader(file)
    text=""
    for i in range(len(pdf_reader.pages)):
      text+=pdf_reader.pages[i].extract_text()
    text=text.replace('\n',' ')
    return text
    
def extensive_test(filename,model_path):
  global context_size
  global input_length
  print('LOADING MODEL AFRESH')
  strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
  with tf.device('/gpu:0'):
    model = TFAutoModelForCausalLM.from_pretrained(model_path)
      tokenizer=AutoTokenizer.from_pretrained(model_path)
  #test=read_pdf(filename)
  with open(filename,'r') as f:
   test=f.read()
  length=len(test)
  test=test[:input_length]
  sequence=test[:context_size]
  start=timeit.default_timer()
  encoded=run_test(model,tokenizer,test,sequence)
  time_taken=timeit.default_timer()-start
  ratio=sys.getsizeof(encoded)*8/len(test[100:])
  net_compression =gzip.compress(encoded.encode('utf-8'))                  #GZIP BASED RANK COMPRESSION
  net_size=sys.getsizeof(net_compression)
  brotli_compression=brotli.compress(encoded.encode('utf-8'), quality=11)  #BROTLI BASED RANK COMPRESSION
  brotli_size=sys.getsizeof(brotli_compression)
  only_gzip=gzip.compress(test[context_size:].encode('utf-8'))
  gzip_size=sys.getsizeof(only_gzip)
  print('\n','.'*20,'STATISTICS','.'*20)
  print('Length of input',len(test[context_size:]))
  print('Transformer only',ratio)
  print('only gzip',gzip_size)
  print('net size',net_size)
  print('brotli size', brotli_size)
  print('time taken',time_taken)
  print('.'*62)
  del model
  del tokenizer

#Example execution code for a text file
extensive_test('test.txt','gpt2')    


#Comments
#For PDF files the read PDF function above can be used to extract the text to be compressed
