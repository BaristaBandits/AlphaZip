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

# Hyperparameters
input_length = 10000
context_size = 100


def run_test(model, tokenizer, test, context):  # Test: input text, sequence:context
    global context_size
    model.config.pad_token_id = model.config.eos_token_id

    test_ids = tokenizer.encode(test)  # Tokenizing the input text
    context_ids = tokenizer.encode(context)
    length = len(context_ids)

    right_ids = test_ids[
        length:
    ]  # This list will be used for rank prediction based on comparison

    xla_generate = tf.function(
        model, jit_compile=True
    )  # Converting the model into JIT compiled XLA graph
    output_string = ""
    print("STARTING////////")
    for i in range(len(right_ids)):
        inputs = tokenizer(f"{context}", return_tensors="tf")
        logits = xla_generate(**inputs).logits[
            :, -1, :
        ]  # Generating the probability distribution

        topk = (
            tf.argsort(logits, axis=-1, direction="DESCENDING", stable=False)
            .numpy()
            .reshape(-1)
            .tolist()
        )
        right_token = right_ids[i]
        rank = topk.index(right_token)  # Rank Prediction
        output_string += str(rank) + "."

        inputs = tokenizer(
            f"{context}", return_tensors="tf"
        )  # Context window moved forward
        context += tokenizer.decode(right_token)
        context = context[-context_size:]

    # print(output_string)                                         #uncomment to view the actual ranks
    return output_string


def read_pdf(
    path_PDF, path_text_file
):  # This function can be used when compressing the PDF rather than a text file
    with open(path_PDF, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for i in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[i].extract_text()
        text = text.replace("\n", " ")
        with open(path_text_file, "w") as f:
            f.write(text)


def extensive_test(filename, model_path):
    global context_size
    global input_length
    print("LOADING MODEL AFRESH")
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
    with tf.device("/gpu:0"):
        model = TFAutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    # test=read_pdf(filename)                                    #uncomment to extract text from PDF files and compress it

    with open(filename, "r") as f:
        test = f.read(
            input_length
        )  # Constraining the input to first n characters (where n=input_length)

    context = test[:context_size]  # This is the context

    # Neural Rank prediction
    start = timeit.default_timer()
    encoded = run_test(model, tokenizer, test, context)
    time_taken = timeit.default_timer() - start

    # Rank compression
    gzip_compression = gzip.compress(
        encoded.encode("utf-8")
    )  # GZIP BASED RANK COMPRESSION
    gzip_size = sys.getsizeof(gzip_compression)
    brotli_compression = brotli.compress(
        encoded.encode("utf-8"), quality=11
    )  # BROTLI BASED RANK COMPRESSION
    brotli_size = sys.getsizeof(brotli_compression)

    # Comparing with the baselines
    only_gzip = gzip.compress(test[context_size:].encode("utf-8"))
    only_gzip_size = sys.getsizeof(only_gzip)
    only_brotli = brotli.compress(test[context_size:].encode("utf-8"), quality=11)
    only_brotli_size = sys.getsizeof(only_brotli)

    separator = "." * 50
    title = "." * 20 + " STATISTICS " + "." * 20

    print(f"\n{title}")
    print(f"{'Length of input:':<35}{len(test[context_size:]):>10}")
    print(f"{'Gzip Neural Compression Size:':<35}{gzip_size:>10}")
    print(f"{'Brotli Neural Compression Size:':<35}{brotli_size:>10}")
    print(f"{'Gzip Size:':<35}{only_gzip_size:>10}")
    print(f"{'Brotli Size:':<35}{only_brotli_size:>10}")
    print(f"{'Time taken:':<35}{time_taken:>10}")
    print(separator)

    del model  # To free the disk space on the device
    del tokenizer


# Example execution code for a text file
extensive_test("test.txt", "gpt2")


# Comments
# For PDF files the read PDF function above can be used to extract the text to be compressed as shown below
read_PDF("test.pdf", "test.txt")
extensive_test("test.txt", "gpt2")
