import torch
import os
import sys

from dataloader import *
from indices import *
from dataset import *
from biaffine_parser import *




DEVICE     = 1
BATCH_SIZE = 8 #32 leads to memory problems
NB_EPOCHS  = 10 #40
#LR = 0.001
#LR = 0.005
#LR = 0.0001
#LR = 0.00005
LR = 0.0005    
NB_EPOCHS_FRAME_ONLY=0 #5
LEX_DROPOUT=0.33


split_info_file = './FrenchFN-corpus-1_3/sequoiaftb_split_info.txt'
gold_conll_file = './FrenchFN-corpus-1_3/sequoia+ftb.asfalda_sem_annotated.1_3.conll'
pretrained_w_emb = './vecs100-linear-frwiki'


from transformers import AutoModel, AutoTokenizer, AutoConfig

bert_tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_cased")
bert_config = AutoConfig.from_pretrained("flaubert/flaubert_base_cased")
bert_model = AutoModel.from_pretrained("flaubert/flaubert_base_cased",return_dict=True)


sentences = load_frame_graphs(gold_conll_file, split_info_file,val_proportion=None,other_sense=False)
indices = Indices(sentences['train'], w_embeddings_file=pretrained_w_emb, bert_tokenizer=bert_tokenizer)

data = {}
for part in sentences.keys():
    data[part] = DepGraphDataSet(part, sentences[part], indices, DEVICE)


biaffineparser = BiAffineParser(indices, DEVICE, 
                              stacked='simple',
                              w_emb_size=100,
                              l_emb_size=100,
                              p_emb_size=100,
                              lstm_h_size=300, #600
                              mlp_frame_h_size=500,
                              mlp_arc_o_size=300, 
                              mlp_lab_o_size=500,
                              use_pretrained_w_emb=True, 
                              use_bias=False,
                              freeze_bert=True,
                              dyn_weighting=False,
                              bert_model=bert_model)


train_data = data['train']
val_data = data['dev']

outdir="."  
config_name = 'compacte.ftb.train.lr' + '_Adam' + str(LR) + '_flaub-b-c' + '_ldpo' + str(LEX_DROPOUT)

biaffineparser.train_model(train_data, val_data, outdir, config_name, 
                        NB_EPOCHS, BATCH_SIZE, LR, 
                        LEX_DROPOUT,
                        nb_epochs_frame_only=NB_EPOCHS_FRAME_ONLY, 
                        frame_training=True, role_training=True)