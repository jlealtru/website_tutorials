#!/usr/bin/env python
# coding: utf-8

# # Text classification with the *Longformer*
# 
# In a previous [post](https://jesusleal.io/2020/10/20/RoBERTA-Text-Classification/) I explored how to use Hugging Face Transformers Training class to easily create a text classification pipeline. The code was pretty straighforward to implement and I was able to obtain results that put the basic model at a very competitive stance. In that post I also briefly discussed one of the main drawbacks of the first iteration of Transformers and BERT based architectures; the sequence lenght is limited to 512 characters. The main limitation of attention based models is the fact that self-attention mechanism scales quadratically with the input sequence length O(n^2). To combat this a second generation of attention based models have been proposed to address this bottleneck. New models such as the [***Reformer***](https://arxiv.org/pdf/2001.04451.pdf) by Google (see this [post](https://huggingface.co/blog/reformer) from Hugging Face for a detailed discussion) implement different attention mechanisms such as Local Self Attention, Locality sensitive hashing (LSH) Self-Attention, Chunked Feed Forward Layers, etc. This model can process sequences of half a million tokens with as little as 8GB of RAM. However one big drawback of the model for downstream applications is the fact that the authors did not released pre trained weights of their model. 
# 
# Another very promising model, and the subject of this post, is the [***Longformer***](https://arxiv.org/pdf/2004.05150.pdf) by researchers from Allen AI Institure. The Longformer allows the processing sequences of thousand characters without facing the memory bottleneck of BERT like architectures and achieved SOTA at the time of publication in severa; . up to 4026 how code I was able In this post I will continue exploring the Training class but this time to use the Longformer a transformer architecture designed to overcome some of limitations of transformer based models as it pertains to sequence length.
# 
# 
# 
# * Explain what the main innovations are. Mention memory scalability issues.
# * mention architecture and use of unique kernel
# * mentio state of the art and see if this works with imdb to achieve new state of the art
# * talk about other stuff
# 

# In[1]:


import pandas as pd
import datasets
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
import os


# In[2]:


config = LongformerConfig()

config


# The datasets library handles the hassle of downloading and processing nlp datasets which is quite convenient to save time in processing and use it for modelling. First we need to instantiate the class by calling the method `load_dataset`. In case the dataset is not loaded, the library downloads it and saves it in the datasets default folder. 
# 
# This example provided by HuggingFace uses an older version of datasets (still called nlp) and demonstrates how to user the [trainer class with BERT](https://colab.research.google.com/drive/1-JIJlao4dI-Ilww_NnTc0rxtp-ymgDgM?usp=sharing#scrollTo=5DEWNilys9Ty). Todays tutorial will follow several of the concepts described there.
# 
# The dataset class has multiple useful methods to easily load, process and apply transformations to the dataset. We can even load the data and split it into train and test feeding a list to the split argument. 

# In[3]:


train_data, test_data = datasets.load_dataset('imdb', split =['train', 'test'], 
                                             cache_dir='/media/jlealtru/data_files/github/website_tutorials/data')


# The resulting objects contains an [arrow dataset](https://arrow.apache.org/docs/python/dataset.html)a format optimized to work with  all the attributes of the original dataset, including the original text, label, types, number of rows, etc. 

# In[ ]:


train_data
#dir(train_data)


# We can operate straigh into the dataset and tokenize the text using another one of the Hugging Face libraries [Tokenizers](https://github.com/huggingface/tokenizers). That library provides Rust optimized code to process the data and return all the necessary inputs for the model such as masks, token ids, etc. We simply load the corresponding model by specifying the name of the model and the tokenizer; if we want to use a finetuned model or a model trained from scratch simply change the name of the model to the location of the pretrained model.
# 
# We can apply the tokenizer to the train and test subsets using the FastTokenizerFromPretrained class from the Transformers library. To do that we simply define a function that makes a call to the tokenizer class. We can specify if we want to add `padding`, if we want to truncate sentences that are longer than the maximum lenght established, etc. The method returns a `batch_encode` class that holds all the necessary inputs for the model such as `tokens`, `attention_masks`, etc.  We then can use the `map` function and apply the tokenizer function to all the elements of all the splits in dataset.

# In[ ]:


# load model and tokenizer and define length of the text sequence
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                           gradient_checkpointing=True,
                                                           attention_window = 512)
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length = 1024)


# In[ ]:


#tokenizer.encode('Mama is not a ft  sentence from training data', return_tensors='pt')
model.config


# In[ ]:


# define a function that will tokenize the model, and will return the relevant inputs for the model
def tokenization(batched_text):
    return tokenizer(batched_text['text'], padding = 'max_length', truncation=True)

#attention_mask[:, [0]] = 1

train_data = train_data.map(tokenization, batched = True, batch_size = len(train_data))
test_data = test_data.map(tokenization, batched = True, batch_size = len(test_data))


# Once the tokenization process is finished we can use the set the column names and types.

# In[ ]:


train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])


# The trainer helper class is designed to facilitate the finetuning of models using the Transformers library. The `Trainer`  class depends on another class called `TrainingArguments` that contains all the attributes to customize the training. `TrainingArguments` contains useful parameter such as output directory to save the state of the model, number of epochs to fine tune a model, use of mixed precision tensors (available with the [Apex](https://github.com/NVIDIA/apex) library), warmup steps, etc. Using the same class we can also ask the model to evaluate the model at the end of each training epoch rather than after a determined amount of steps. To make sure we evaluate at the end of the training epoch we set `evaluation_strategy = 'Epoch'`. For this case we also set the option `load_best_model_at_end` to true, this will guarantee that we will load the best model for evaluation
# (according to the metrics defined) at the end of training.
# 
# The `Trainer` class provides also allows to implement more sophisticated optmizers and learning rates which can be fed in the `optimizer` option. For this tutorial I use the default gradient descent optimization algorithm provided by the library *AdamW*. AdamW is an optimization based on the original Adam(Adaptive Moment Estimation) that incorporates a regularization term designed to work well with adaptive optimizers; a pretty good discussion of Adam, AdamW and the importance of regularization can be found [here](https://towardsdatascience.com/why-adamw-matters-736223f31b5d). The class also uses a default scheduler to modify the learning rate as the training of the model progresses. The default scheduler on the trainer class is [`get_linear_schedule_with_warmup`](https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup) an scheduler that decreases the learning rate linearly until it reaches zero. As mentioned before we can also modify the default values to use a different scheduler. For the learning rate I chose the default of 5e-5 as I wanted to be conservative since this an already pretrained model. Further [Sun et al](https://arxiv.org/pdf/1905.05583.pdf) found that a learning rate of 5e-5 works well for text classification.  I did not modify any of the other parameters of AdamW. 
# 
# `Trainer` also makes accumulating gradient steps pretty straightforward. This is relevant when we need to train models on smaller GPU's. For this tutorial I will be using a [GeForce GTX 1080](https://www.nvidia.com/en-sg/geforce/products/10series/geforce-gtx-1080/) that has 8GB of RAM. Given the size of the models (in this case 125 million parameters) and the limitation of the memory.
# 
# We can also define if we want to log the training into wanddb. Wandb, short for [Weights and Biasis](https://www.wandb.com/), is a service that allows you visualize the performance of your model and parameters ina very nice dashboad. In this tutorial I assumed you have wandb installed and configured to log the information of weights and parameters. A detailed tutorial of wandb can be found [here](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-Hugging-Face-Model-Performance--VmlldzoxMDE2MTU). We define the name of the run with `run_name` in the `TrainingArguments` class to easily keep track of the model.
# 
# Finally we can also specify the metrics to evaluate the performance of the model on the test set with the `compute_metrics` argument in the `Trainer` class. In this example I selected accuracy, f1 score, precision and recall as suggested in the tutorial by Hugging Face and wrapped them in a functiont hat returns the values for these metrics. This set of metrics provide a very good idea on the performance of the model.

# In[ ]:


# define accuracy metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # argmax(pred.predictions, axis=1)
    #pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# In[ ]:


# define the training arguments
training_args = TrainingArguments(
    output_dir = '/media/jlealtru/data_files/github/website_tutorials/results',
    num_train_epochs=3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 32,    
    per_device_eval_batch_size= 2,
    evaluation_strategy = "epoch",
    disable_tqdm = False, 
    load_best_model_at_end=True,
    warmup_steps=1,
    weight_decay=0.01,
    logging_steps = 8,
    fp16 = True,
    logging_dir='/media/jlealtru/data_files/github/website_tutorials/logs',
    dataloader_num_workers = 4,
    run_name = 'longformer-classification-trash'
)


# In[ ]:


# instantiate the trainer class and check for available devices
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# In[ ]:





# In[ ]:


# train the model
trainer.train()


# After the training has been completed we can evaluate the performance of the model and make sure we are loading the right model.

# In[ ]:


trainer.evaluate()


# The best iteration of our model achieved an accuracy 0.9565, which would put us on [third place](http://nlpprogress.com/english/sentiment_analysis.html) in the leaderboard of sentiment analysis classification with IMDB.
# 
# ![](images/eval_roberta.svg)
# 
# Thats it for this tutorial, hopefully you will find this helpful.
