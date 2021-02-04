#!/usr/bin/env python
# coding: utf-8

# # Multi-label classification with the RoBERTa
# 
# In a previous post I explored the functionality of the Longformer for text classification. In this post I will explore the performance of the Longformer in a setting of multilabel classification problem.
# 
# For this dataset we need to download it manually from Kaggle and load it like we usually do with the datasets library. 'jigsaw_toxicity_pred', data_dir='/path/to/extracted/data/'

# In[1]:


import torch
import transformers
import pandas as pd
import numpy as np
from torch.nn import BCEWithLogitsLoss
from transformers import RobertaTokenizerFast, RobertaModel, RobertaConfig, Trainer, TrainingArguments,EvalPrediction
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaClassificationHead
from torch.utils.data import Dataset, DataLoader
import wandb
import random


# In[2]:


# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


# In[3]:


#wandb.login()


# We are going to instantiate a raw LongFormer Model and add a classifier head on top. 
# 
# talk about pos_weight

# In[4]:


# read the dataframe
insults = pd.read_csv('../data/jigsaw/train.csv')
#insults = insults.iloc[0:64]
insults['labels'] = insults[insults.columns[2:]].values.tolist()
insults = insults[['id','comment_text', 'labels']].reset_index(drop=True)


# In[5]:


'''
from sklearn.model_selection import train_test_split
insults_train, insults_test = train_test_split(insults,
                                               random_state = 55,
                                               test_size = 0.35)
insults_test.head()
insults_test.columns
'''
train_size = 0.8
train_dataset=insults.sample(frac=train_size,random_state=200)
test_dataset=insults.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


# In[6]:


#insults_test = pd.read_csv('../data/jigsaw/test.csv')
#insults_test_ids = pd.read_csv('../data/jigsaw/test_labels.csv')
#insults_test['labels']
train_dataset


# In[7]:


# instantiate a class that will handle the data
class Data_Processing(object):
    def __init__(self, tokenizer, id_column, text_column, label_column):
        
        # define the text column from the dataframe
        self.text_column = text_column.tolist()
    
        # define the label column and transform it to list
        self.label_column = label_column
        
        # define the id column and transform it to list
        self.id_column = id_column.tolist()
        
    
# iter method to get each element at the time and tokenize it using bert        
    def __getitem__(self, index):
        comment_text = str(self.text_column[index])
        comment_text = " ".join(comment_text.split())
        
        inputs = tokenizer.encode_plus(comment_text,
                                       add_special_tokens = True,
                                       max_length= 512,
                                       padding = 'max_length',
                                       return_attention_mask = True,
                                       truncation = True,
                                       return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        labels_ = torch.tensor(self.label_column[index], dtype=torch.float)
        id_ = self.id_column[index]
        return {'input_ids':input_ids[0], 'attention_mask':attention_mask[0], 
                'labels':labels_, 'id_':id_}
  
    def __len__(self):
        return len(self.text_column) 


# In[8]:


batch_size = 4
# create a class to process the traininga and test data
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base',
                                          padding = 'max_length',
                                          truncation=True, 
                                          max_length = 512)
training_data = Data_Processing(tokenizer, 
                                train_dataset['id'], 
                                train_dataset['comment_text'], 
                                train_dataset['labels'])

test_data =  Data_Processing(tokenizer, 
                             test_dataset['id'], 
                             test_dataset['comment_text'], 
                             test_dataset['labels'])

# use the dataloaders class to load the data
dataloaders_dict = {'train': DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2),
                    'val': DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
                   }

dataset_sizes = {'train':len(training_data),
                 'val':len(test_data)
                }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[9]:


# check we are getting the right output
a = next(iter(dataloaders_dict['val']))
a
#len(dataloaders_dict['train'])


# In[10]:


# instantiate a Longformer for multilabel classification class

class RobertaForMultiLabelSequenceClassification(RobertaPreTrainedModel):
    """
    We instantiate a class of LongFormer adapted for a multilabel classification task. 
    This instance takes the pooled output of the LongFormer based model and passes it through a
    classification head. We replace the traditional Cross Entropy loss with a BCE loss that generate probabilities
    for all the labels that we feed into the model.
    """

    def __init__(self, config, pos_weight=None):
        super(RobertaForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()
        
    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, 
                token_type_ids=None, position_ids=None, inputs_embeds=None, 
                labels=None):
        
        # create global attention on sequence, and a global attention token on the `s` token
        # the equivalent of the CLS token on BERT models
        # pass arguments to longformer model
        outputs = self.roberta(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids)
        
        # if specified the model can return a dict where each key corresponds to the output of a
        # LongformerPooler output class. In this case we take the last hidden state of the sequence
        # which will have the shape (batch_size, sequence_length, hidden_size). 
        sequence_output = outputs['last_hidden_state']
        
        
        # pass the hidden states through the classifier to obtain thee logits
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), 
                            labels.view(-1, self.num_labels))
            #outputs = (loss,) + outputs
        
        
        return loss, logits


# In[11]:


model = RobertaForMultiLabelSequenceClassification.from_pretrained('roberta-base',
                                                                   gradient_checkpointing=False,
                                                                   num_labels = 6,
                                                                   cache_dir='/media/data_files/github/website_tutorials/data',
                                                                   return_dict=True)
model


# In[12]:


from sklearn.metrics import f1_score
#acc = accuracy_score(labels, preds)
    #acc = accuracy_score(labels, preds)
    
def multi_label_metric(
    predictions, 
    references, 
    ):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_true = references
    y_pred[np.where(probs > 0.5)] = 1
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    metrics = {'f1':f1_micro_average}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metric(
        predictions=preds, 
        references=p.label_ids
    )
    return result


# In[13]:


# define the training arguments
training_args = TrainingArguments(
    output_dir = '/media/data_files/github/website_tutorials/results',
    num_train_epochs = 3,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 16,    
    per_device_eval_batch_size= 32,
    evaluation_strategy = "epoch",
    disable_tqdm = False, 
    load_best_model_at_end=True,
    warmup_steps = 1000,
    weight_decay=0.01,
    logging_steps = 4,
    fp16 = False,
    logging_dir='/media/data_files/github/website_tutorials/logs',
    dataloader_num_workers = 0,
    run_name = 'roberta_multilabel_trainer_jigsaw'
)


# In[14]:


# instantiate the trainer class and check for available devices
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=test_data,
    compute_metrics = compute_metrics,
    #data_collator = Data_Processing(),

)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# In[15]:


trainer.train()


# In[16]:


trainer.evaluate()


# Mention there several observations longer than 1000

# In[17]:


'''
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                           gradient_checkpointing=False,
                                                           attention_window = 512,
                                                           cache_dir='/media/data_files/github/website_tutorials/data')
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length = 2048)
'''


# In[18]:


'''from torch.nn import BCEWithLogitsLoss, Dropout, Linear
from transformers import LongformerTokenizerFast, LongformerModel, LongformerConfig
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel,LongformerClassificationHead


# instantiate the multi-label classification class

class LongFormerMultilabelClass(LongformerPreTrainedModel):
    def __init__(self, config, pos_weight = None):
        super(LongFormerMultilabelClass, self).__init__(config)
        self.num_labels = config.num_labels
        self.LongformerModel = LongformerModel(config)
        self.dropout = Dropout(0.3)
        self.classifier = Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        

    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, position_ids = None,
                head_mask = None, inputs_embeds=None, labels = None):
        
        outputs = self.LongformerModel(input_ids, attention_mask=attention_mask, 
                                       token_type_ids=token_type_ids, position_ids=position_ids,
                                       head_mask=head_mask,
                                       inputs_embeds=inputs_embeds)
        
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

            outputs = (loss,) + outputs

        return outputs 
'''

