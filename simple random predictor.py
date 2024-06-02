#with key as the original word and value as another dictionary with possiblities against probabilities
"""Objectives:
             1. Understanding the probabilities/weights
             2. Assign Score on Accuracy of the model"""
import random
import numpy as np
import copy

def encoder(data_list):
    predict_dict={}
    predict_dict[data_list[0]]=[1,{}]
    for i in range(1,len(data_list)):
        word=data_list[i]
        if word not in predict_dict :                                    #Everytime a new word occurs we store it as a key in the dictionary
            predict_dict[word]=[1,{}]
        else:
            predict_dict[word][0]+=1
        if(i!=0):
         update_probability(data_list,predict_dict,i)
    return predict_dict


def update_probability(data_list, predict_dict,i):
    word=data_list[i]
    previous_word=data_list[i-1]
    denom_new=predict_dict[previous_word][0]
    denom_old=denom_new-1
    for i in predict_dict[previous_word][1]:
         if(i!=word):
                predict_dict[previous_word][1][i]=np.float16((denom_old*predict_dict[previous_word][1][i])/denom_new)
         else:
                predict_dict[previous_word][1][i]=np.float16((denom_old*predict_dict[previous_word][1][i]+1)/denom_new)

    else:
        if(word not in predict_dict[previous_word][1]):
            predict_dict[previous_word][1][word]=np.float16(1/denom_new)

def generate_topp(text,predict_dict):
    while(len(text)<150):
      word=text.split()[-1]
      probables=[]
      a=copy.deepcopy(predict_dict[word][1])
      for i in range(10):
        if(a=={}):
          break
        max_probability=max(a.values())
        for i in a:
            if(a[i]==max_probability):
                probables.append(i)
                break
        del a[i]
      n=random.randint(0,len(probables)-1)
      text+=probables[n]+' '
    return text

data_list=inputtxt.split()
predict_dict=encoder(data_list)
while(True):
 input_word=(input('enter a word').lower())
 text=''
 text+=input_word+' '
 print(generate_topp(text,predict_dict))


def generate_topk(text,predict_dict):
    threshold = 0.4
    while(len(text)<150):
      word=text.split()[-1]
      probables=[]
      t_count=0
      a=copy.deepcopy(predict_dict[word][1])
      while(t_count<=threshold):
        if(a=={}):
          break
        max_probability=max(a.values())
        t_count+=max_probability
        for i in a:
            if(a[i]==max_probability):
                probables.append(i)
                break
        del a[i]
      n=random.randint(0,len(probables)-1)
      text+=probables[n]+' '
    return text

while(True):
 input_word=(input('enter a word').lower())
 text=''
 text+=input_word+' '
 print(generate_topk(text,predict_dict))

def generate_topkp(text,predict_dict):
    threshold = 0.4
    while(len(text)<150):
      word=text.split()[-1]
      probables=[]
      t_count=0
      a=copy.deepcopy(predict_dict[word][1])
      for i in range(6):
        while(t_count<=threshold):
          if(a=={}):
            break
          max_probability=max(a.values())
          t_count+=max_probability
          for i in a:
              if(a[i]==max_probability):
                  probables.append(i)
                  break
          del a[i]
      n=random.randint(0,len(probables)-1)
      text+=probables[n]+' '
    return text

while(True):
 input_word=(input('enter a word').lower())
 text=''
 text+=input_word+' '
 print(generate_topkp(text,predict_dict))
