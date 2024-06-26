# AlphaZip: Neural-Networks-Aided-Lossless-Text-Compression
Leveraging the power of Large Language Models in compressing text in a lossless fashion using a rank based prediction followed by compression.
Refer to the Wiki to know how you can compress text files in real-time in your personal Computer!

Welcome to the Neural-Networks-Aided-Lossless-Text-Compression wiki!

Requirements:

NVIDIA GeForce RTX 4080
Python 3.10.12
Pytorch 2.2.2
Tensorflow 2.11.0 (FOR XLA)
Procedure:

Install the requirements ensuring that there are no dependency conflicts
Compress any text file you have by using the compress.py code
Change the path variable to the required file's path
For PDFs utilise the read_PDF function to first extract the text from PDF to a text file and then proceed to compress the text file as illustrated in the code
For using the adaptive huffman method, you can copy paste the function or import the function as a user defined library
Information about Hyperparameters:

input_length: Refers to the number of ASCII characters from the input text that we would like to compress
context_size: Refers to the number of characters that is used as a context for the transformer block to predict the next next token
Domain Specific Compression:

**Fine-tuning: **To fine tune the model you can use the fine_tuning.py code and insert a input file as 'input.txt' . You can test the compression performance from any checkpoint by using the compress.py code. Modify the code as follows.

extensive_test('<file_to_be_compressed_path>.txt', '<current_directory_path>/fine_tuning_weights/checkpoint-XXXX')

**Knowldge Distillation: **To knowledge distil gpt2 you can use the fine_tuning.py code and insert a input file as 'input.txt' . You can test the compression performance from any checkpoint by using the compress.py code. Modify the code as follows.

extensive_test('<file_to_be_compressed_path>.txt', '<current_directory_path>/Knowledgdistillation_weights/checkpoint-XXXX')
