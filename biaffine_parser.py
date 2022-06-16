"""## The parser"""

import torch as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import numpy as np
from random import shuffle
import sys

from modules import *
from dataloader import *
from copy import deepcopy
import pandas as pd
import os



def fscore(nb_correct, nb_gold, nb_pred):
    if (nb_correct * nb_gold * nb_pred == 0):
        return 0
    r = nb_correct / nb_gold
    p = nb_correct / nb_pred
    return 100*2*r*p/(p+r)




class BiAffineParser(nn.Module):
    """
dozat and manning 2018 hyperparameters: (table 2 p4)
lstm_layers = 3
p_emb_size = l_emb_size = char = 100
f_emb_size = 100
w_emb_size = 125
lstm_layers = 3
lstm_h_size = 600

mlp_arc_o_size = 600
mlp_lab_o_size = 600

dozat conll 2017

lexical dropout (word and pos dropout) = 1/3
we use same- mask dropout (Gal and Ghahramani, 2015) in the LSTM, ReLU layers, and classifiers, dropping in- put and recurrent connections with 33% probability
lstm_layers = 3
lstm_h_size = 200

mlp_arc_o_size = 400
mlp_lab_o_size = 400

"""
    # TODO replicate from dozat conll 2017:
    # "we drop word and tag embeddings inde- pendently with 33% probability
    # When only one is dropped, we scale the other by a factor of two"
    # TODO: replicate also:
    #   We used 100-dimensional pretrained GloVe em- beddings
    #   but linearly transformed them to be 125-dimensional
    def __init__(self, indices, device, 
                 stacked='simple',
                 w_emb_size=10, #125
                 l_emb_size=10 , 
                 p_emb_size=None, # 100,
                 use_pretrained_w_emb=False,
                 lstm_dropout=0.33, 
                 lstm_h_size=20, # 600
                 lstm_num_layers=3, 
                 mlp_frame_h_size=400, #TEST
                 mlp_frame_dropout=0.25,
                 mlp_arc_o_size=25, # 600
                 mlp_arc_dropout=0.25, 
                 mlp_lab_o_size=10, # 600
                 mlp_lab_dropout=0.25,
                 use_bias=False,
                 dyn_weighting=False,
                 freeze_bert = True,
                 bert_model=None):  # caution: should coincide with indices.bert_tokenizer
        super(BiAffineParser, self).__init__()

        self.indices = indices
        self.device = device
        self.use_pretrained_w_emb = use_pretrained_w_emb
        self.stack = stacked #option for forward of heads

        self.lexical_emb_size = w_emb_size
        w_vocab_size = indices.get_vocab_size('w')

        self.num_labels = indices.get_vocab_size('label')
        self.num_frames = indices.get_vocab_size('frame')
        self.w_emb_size = w_emb_size
        self.p_emb_size = p_emb_size
        self.l_emb_size = l_emb_size
        self.lstm_h_size = lstm_h_size
        self.mlp_frame_h_size = mlp_frame_h_size
        self.mlp_frame_dropout = mlp_frame_dropout
        self.mlp_arc_o_size = mlp_arc_o_size
        self.mlp_arc_dropout = mlp_arc_dropout
        self.mlp_lab_o_size = mlp_lab_o_size
        self.mlp_lab_dropout = mlp_lab_dropout
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout

        self.use_bias = use_bias # whether to add bias in all biaffine transformations
        self.freeze_bert = freeze_bert

        self.dyn_weighting = dyn_weighting
        if self.dyn_weighting :
            #self.log_vars = nn.Parameter(torch.ones((3))).to(self.device)
            self.log_vars = nn.Parameter(torch.ones((3))).to(self.device)
    
        # -------------------------
        # word form embedding layer
        if not use_pretrained_w_emb:
            self.w_embs = nn.Embedding(w_vocab_size, w_emb_size).to(self.device)
        else:
            matrix = indices.w_emb_matrix
            
            #if w_emb_size != matrix.shape[1]:
            #    sys.stderr.write("Error, pretrained embeddings are of size %d whereas %d is expected"
            #                     % (matrix.shape[1], w_emb_size))
            if w_vocab_size != matrix.shape[0]:
                sys.stderr.write("Error, pretrained embeddings have a %d vocab size while indices have %d"% (matrix.shape[0], w_vocab_size))
            self.w_embs = nn.Embedding.from_pretrained(matrix, freeze = False).to(self.device)
            # linear transformation of the pre-trained embeddings (dozat 2018)
            self.w_emb_linear_reduction = nn.Linear(matrix.shape[1],w_emb_size).to(self.device)
            
        print("w_embs done")
        # -------------------------
        # pos tag embedding layer
        if p_emb_size:
            p_vocab_size = indices.get_vocab_size('p')
            # concatenation of embeddings hence +
            self.lexical_emb_size += p_emb_size
            self.p_embs = nn.Embedding(p_vocab_size, p_emb_size).to(self.device)
        else:
            self.p_embs = None

        # -------------------------
        # lemma embedding layer, needed for frame
        l_vocab_size = indices.get_vocab_size('l')
        self.lexical_emb_size += l_emb_size
        self.l_embs = nn.Embedding(l_vocab_size, l_emb_size).to(self.device)


        # -------------------------
        # bert embedding layer
        if bert_model is not None:
            self.bert_layer = bert_model.to(self.device)
            if self.freeze_bert:
              for p in self.bert_layer.parameters():
                  p.requires_grad = False 

            self.lexical_emb_size += self.bert_layer.config.emb_dim
        else:
            self.bert_layer = None
            
        # -------------------------
        # recurrent LSTM bidirectional layers
        #   TODO: same mask dropout across time-steps ("locked dropout")
        self.lstm = nn.LSTM(input_size = self.lexical_emb_size, 
                            hidden_size = lstm_h_size, 
                            num_layers = lstm_num_layers, 
                            batch_first = True,
                            bidirectional = True,
                            dropout = lstm_dropout).to(self.device)

        # -------------------------
        # specialized MLP applied to biLSTM output
        #   rem: here hidden sizes = output sizes
        #   for readability:
        s = 2 * lstm_h_size
        a = mlp_arc_o_size
        l = mlp_lab_o_size
        f = self.mlp_frame_h_size
        
        #here input size for the mlp head depending of options
        #option use hiddenframe for stacked prop
        if self.stack == 'simple':
            h = f 
        #option use hiddenframe for stacked prop with concateation
        elif self.stack == 'cat':
            h= f+s
        #option for baselin pipeline:
        elif self.stack == 'pipeline':
            h = s + f
        #no stack prop
        else:
            h = s

        #---------------------
        #frame part
        #first frame identification
        self.frame_mlp = MLP_out_hidden(s+l_emb_size, mlp_frame_h_size, self.num_frames, dropout=0).to(device)  
        #one more option with pipeline
        # frame embedding layer
        if self.stack == 'pipeline':
            f_vocab_size = indices.get_vocab_size('frame')
            self.f_embs = nn.Embedding(f_vocab_size, f).to(self.device)
        else:
            self.f_embs = None


        #MLPs        
        self.arc_d_mlp = MLP(s, a, a, dropout=mlp_arc_dropout).to(device)  
        self.arc_h_mlp = MLP(h, a, a, dropout=mlp_arc_dropout).to(device)  

        self.lab_d_mlp = MLP(s, l, l, dropout=mlp_lab_dropout).to(device)  
        self.lab_h_mlp = MLP(h, l, l, dropout=mlp_lab_dropout).to(device)

        # ---------------------------
        # BiAffine scores
        # biaffine matrix size is num_label x d x d, with d the output size of the MLPs
        self.biaffine_arc = BiAffine(device, a, use_bias=self.use_bias)
        self.biaffine_lab = BiAffine(device, l, num_scores_per_arc=self.num_labels, use_bias=self.use_bias, diag=True)


    def forward(self, w_id_seqs, l_id_seqs, p_id_seqs, bert_tid_seqs, bert_ftid_rkss, b_pad_masks, b_fram_mat, lengths=None):
        """
        Inputs:
         - id sequences for word forms, lemmas and parts-of-speech for a batch of sentences
             = 3 tensors of shape [ batch_size , max_word_seq_length ]
         - bert_tid_seqs : sequences of *bert token ids (=subword ids) 
                           shape [ batch_size, max_token_seq_len ]
         - bert_ftid_rkss : ranks of first subword of each word [batch_size, max_WORD_seq_len +1] (-1+2=1 (no root, but 2 special bert tokens)
         - b_pad_masks : 0 or 1 tensor of shape batch_size , max_word_seq_len , max_word_seq_len 
                         cell [b,i,j] equals 1 iff both i and j are not padded positions in batch instance b
        If lengths is provided : (tensor) list of real lengths of sequences in batch
                                 (for packing in lstm)
        """
        w_embs = self.w_embs(w_id_seqs)
        if self.use_pretrained_w_emb:
            w_embs = self.w_emb_linear_reduction(w_embs)
            
        if self.p_embs:
            p_embs = self.p_embs(p_id_seqs)
            w_embs = torch.cat((w_embs, p_embs), dim=-1)

        l_embs = self.l_embs(l_id_seqs)
        
        if self.l_embs:
            w_embs = torch.cat((w_embs, l_embs), dim=-1)
        
        if bert_tid_seqs is not None:
            bert_emb_size = self.bert_layer.config.emb_dim
            bert_embs = self.bert_layer(bert_tid_seqs).last_hidden_state
            # select among the subword bert embedddings only the embeddings of the first subword of words
            #   - modify bert_ftid_rkss to serve as indices for gather:
            #     - unsqueeze to add the bert_emb dimension
            #     - repeat the token ranks index along the bert_emb dimension (expand better for memory)
            #     - gather : from bert_embs[batch_sample, all tid ranks, bert_emb_dim]
            #                to bert_embs[batch_sample, only relevant tid ranks, bert_emb_dim]
            #bert_embs = torch.gather(bert_embs, 1, bert_ftid_rkss.unsqueeze(2).repeat(1,1,bert_emb_size))
            bert_embs = torch.gather(bert_embs, 1, bert_ftid_rkss.unsqueeze(2).expand(-1,-1,bert_emb_size))
            w_embs = torch.cat((w_embs, bert_embs), dim=-1)
            
        # h0, c0 vectors are 0 vectors by default (shape batch_size, num_layers*2, lstm_h_size)

        # pack_padded_sequence to save computations
        #     (compute real length of sequences in batch)
        #     see https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
        #     NB:batch must be sorted in sequence length descending order
        if lengths is not None:
            lengths=lengths.cpu()
            w_embs = pack_padded_sequence(w_embs, lengths, batch_first=True)
        lstm_hidden_seq, _ = self.lstm(w_embs)
        if lengths is not None:
            lstm_hidden_seq, _ = pad_packed_sequence(lstm_hidden_seq, batch_first=True)
        
        #Pour chaque phrase du batch, recupère les positions du prédicats --> recupere hidden du lstm aux pos p
        # matrice Hh sans dim du batch --> (nb_pred, )
        #lstm_hidden_seq:(batch_size,seq_length,lstm_hidden_size)

        # MLP frame
        out_frame, hidden_frame = self.frame_mlp(torch.cat((lstm_hidden_seq, l_embs), dim=-1))#on prédit le frame pour tous les tokens plus que nécessaire

        #MLPs heads with opotions of stack
        if self.stack == 'simple':
            arc_h = self.arc_h_mlp(hidden_frame)
            lab_h = self.lab_h_mlp(hidden_frame)

        elif self.stack == 'cat':
            arc_h = self.arc_h_mlp(torch.cat((lstm_hidden_seq, hidden_frame), dim=-1))
            lab_h = self.lab_h_mlp(torch.cat((lstm_hidden_seq, hidden_frame), dim=-1))

        elif self.stack == 'pipeline':
            if self.training:
                f_emb = self.f_embs(b_fram_mat)
            else:
                # out_frame is [batch, seqlength, frame]
                #mask no frame
                f_emb = self.f_embs( (b_fram_mat > 0).int() * torch.argmax(out_frame, dim= 2 ) )
            arc_h = self.arc_h_mlp(torch.cat((lstm_hidden_seq, f_emb), dim=-1))
            lab_h = self.lab_h_mlp(torch.cat((lstm_hidden_seq, f_emb), dim=-1))
        
        else:
            arc_h = self.arc_h_mlp(lstm_hidden_seq)
            lab_h = self.lab_h_mlp(lstm_hidden_seq)

        #MLPs Dep
        arc_d = self.arc_d_mlp(lstm_hidden_seq)
        lab_d = self.lab_d_mlp(lstm_hidden_seq)


        # Biaffine scores
        S_arc = self.biaffine_arc(arc_h, arc_d) # S(k, i, j) = score of sample k, head word i, dep word j
        S_lab = self.biaffine_lab(lab_h, lab_d) # S(k, l, i, j) = score of sample k, label l, head word i, dep word j
        
        # padded cells get -inf : does not work
        #S_arc[b_pad_masks == 0] = -math.inf
        
        return S_arc, S_lab, out_frame
    
    def batch_forward_and_loss(self, batch):
        """
        - batch of sentences (output of make_batches)
        

        NB: in batch graph mode
          - in arc_adja (resp. lab_adja), 0 cells are either
             - 0 cells for non gold or padded
             - 1 (resp. label id) for gold arcs
          - pad_masks : 0 cells if head or dep is padded token and 1 otherwise
        """
        if self.graph_mode:
            lengths, pad_masks, pred_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, arc_adja, lab_adja, fram_mat = batch
            
            # forward 
            S_arc, S_lab , S_frame = self(forms, lemmas, tags, bert_tokens, bert_ftid_rkss, pad_masks, fram_mat, lengths=lengths)

            if self.role_training and self.nb_epochs_frame_only == 0:
                # --- Arc loss -------------------------
                # pred_masks allows to ignore cells (h,d) where h is not evoking a frame, or d is padded
                arc_loss = self.bce_loss_fn_arc(S_arc, arc_adja, pred_masks) #search only arc with the pred gov: pred_mask: b,h,d
                

                # --- Label loss -------------------------
                # label scores : rearrange into a batch in which each sample is 
                #                - one head and dep token pair
                #                - label scores for such arc (NB: non-gold arcs will be masked)
                # S_lab is [batch, label, head, dep]
                s_labels = S_lab.transpose(2,1).transpose(3,2)             # b , h , d, l
                s_labels = torch.flatten(s_labels, start_dim=0, end_dim=2) # b * h * d, l
                
                # same for gold labels
                g_labels = torch.flatten(lab_adja) # [b, h, d] ==> [b * h * d]

                # loss with ignore_index == 0 (=> ignore padded arcs and non-gold arcs, which don't have gold labels anyway)
                # cf. Dozat et al. 2018 "back-propagating error to the labeler only through edges with a non-null gold label"
                lab_loss = self.ce_loss_fn_label(s_labels, g_labels) 

            else:
                lab_loss =  torch.tensor([0]).to(self.device)
                arc_loss = torch.tensor([0]).to(self.device)
            
            if self.frame_training:
                # --- Frame loss -------------------------
                # S_frame is [batch, seqlength, frame]
                s_frames = torch.flatten(S_frame, start_dim=0, end_dim=1) # b * n , f
                # same for gold labels
                g_frames = torch.flatten(fram_mat) # [b, h] ==> [b * h]
                frame_loss = self.ce_loss_fn_frame(s_frames, g_frames) 
            else :
                frame_loss =  torch.tensor([0]).to(self.device)

            
            # --- Evaluation --------------------------
            with torch.no_grad():

                if self.frame_training:
                    pred_frame = torch.argmax(S_frame, dim= 2 ) #dim frame
                    nb_correct_f = torch.sum((pred_frame == fram_mat).float() * (fram_mat > 0).int()).item() #  
                    nb_gold_frame = torch.sum((fram_mat > 0).int()).item()
                else :
                    nb_correct_f = 0
                    nb_gold_frame = 0
                
                if self.role_training and self.nb_epochs_frame_only == 0:
                    #unlabeled
                    pred_arcs = (S_arc > 0).int() * pred_masks  # b, h, d
                    nb_correct_u = torch.sum((pred_arcs * arc_adja).int()).item()                   
                    nb_gold = torch.sum(arc_adja).item()
                    nb_pred = torch.sum(pred_arcs).item()
                    
                    # labeled
                    pred_labels = torch.argmax(S_lab, dim=1) # for all arcs (not only the predicted arcs)
                    # count correct labels for the predicted arcs only
                    nb_correct_u_and_l = torch.sum((pred_labels == lab_adja).float() * pred_arcs).item()

                else:
                    nb_correct_u = 0
                    nb_correct_u_and_l = 0
                    nb_gold = 0
                    nb_pred = 0


        if self.dyn_weighting:
            precision0 = torch.exp(-self.log_vars[0])
            loss0 = precision0*frame_loss + self.log_vars[0]/2

            precision1 = torch.exp(-self.log_vars[1])
            loss1 = precision1*arc_loss + self.log_vars[1]/2

            precision2 = torch.exp(-self.log_vars[2])
            loss2 = precision2*lab_loss + self.log_vars[2]/2

            loss = loss0+loss1+loss2
        
        else:
            loss = lab_loss + arc_loss + frame_loss
        
        # returning the sub-losses too for trace purpose
        return loss, frame_loss.item(), arc_loss.item(), lab_loss.item(), nb_correct_f, nb_correct_u, nb_correct_u_and_l, nb_gold_frame, nb_gold, nb_pred    
    

    def train_model(self, train_data, val_data, log_stream, nb_epochs, batch_size, lr, lex_dropout, nb_epochs_frame_only=0, out_model_file=None, graph_mode=True, frame_training=True, role_training=False, pos_weight=None, config_name=None, score_csv=None):
        """
                
        For graph mode only:
        - pos_weight : weight used in binary cross-entropy loss for positive examples, i.e. for gold arcs
        """
        self.graph_mode = graph_mode
        self.lr = lr
        self.frame_training = frame_training
        self.role_training = role_training
        self.nb_epochs_frame_only = nb_epochs_frame_only # nb of epochs to train the frames only (no arc and dep labeling)
        self.lex_dropout = lex_dropout # proba of word / lemma / pos tag dropout
        self.batch_size = batch_size
        self.beta1 = 0.9
        self.beta2 = 0.9
        #optimizer = optim.SGD(biaffineparser.parameters(), lr=LR)
        #optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0., 0.95), eps=1e-09)
        optimizer = optim.Adam(self.parameters(), lr=lr, betas=(self.beta1, self.beta2), eps=1e-09)
        # from benoit
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
        
        # loss functions
        self.pos_weight=pos_weight
        # for graph mode arcs
        self.bce_loss_fn_arc = BCEWithLogitsLoss_with_mask(reduction='sum', pos_weight_scalar=pos_weight)
        # for label loss, the label for padded deps is PAD_ID=0 
        #   ignoring padded dep tokens (i.e. whose label id equals PAD_ID)
        self.ce_loss_fn_label = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_ID) 
        # for frame loss,the ignored tokens will be the padded ones, and the tokens that do not evoke any frame
        self.ce_loss_fn_frame = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_ID) 
        

        self.log_train_hyper(sys.stdout)
        self.log_train_hyper(log_stream)

        # losses and scores at each epoch (on train / validation)
        train_losses = []

        train_frame_losses = []
        train_arc_losses = []
        train_lab_losses =[]
        
        train_scores_fr     = []
        train_fscores_u = []
        train_fscores_l = []
        
        val_losses = []
        
        val_frame_losses =[]
        val_arc_losses = []
        val_lab_losses = []
        
        val_scores_fr   =[]
        val_fscores_u = []
        val_fscores_l = []
        
        best_epoch   = 0                 
        min_val_loss = np.inf


        # word / pos / lemma dropout of training data only
        # NB: re-drop at each epoch seems quite detrimental
        #train_data.lex_dropout(lex_dropout)


        for epoch in range(1,nb_epochs+1):
            #log_gc()
   
            i = 0
            train_loss = 0
            train_frame_loss = 0
            train_arc_loss = 0
            train_lab_loss = 0

            # arc evaluation on train
            train_nb_correct_f = 0
            train_nb_correct_u = 0
            train_nb_correct_l = 0
            train_nb_pred = 0
            train_nb_gold = 0
            train_nb_gold_frame = 0 
    

            # training mode (certain modules behave differently in train / eval mode)
            self.train()
            train_data.lex_dropout(lex_dropout)
            bid = 0
            for batch in train_data.make_batches(self.batch_size, shuffle_data=True, sort_dec_length=True, shuffle_batches=True):        
                self.zero_grad()
                bid += 1
                if bid % 2000 == 0:
                  print("BATCH SHAPE:", batch[2].shape, batch[5].shape)
                  print("MEMORY BEFORE BATCH FORWARD AND LOSS")
                  printm()

                loss, frame_loss, arc_loss, lab_loss, nb_correct_f, nb_correct_u, nb_correct_l, nb_gold_frame, nb_gold, nb_pred = self.batch_forward_and_loss(batch)
                train_loss += loss.item()
                if frame_loss:
                    train_frame_loss += frame_loss
                if arc_loss:
                    train_arc_loss += arc_loss
                if lab_loss:
                    train_lab_loss += lab_loss

                loss.backward()
                optimizer.step() 
                loss.detach()           

                #predictions, nb_correct, nb_gold, nb_pred = self.predict_from_scores_and_evaluate(S_arc, S_lab, batch)    
                train_nb_correct_f += nb_correct_f
                train_nb_correct_u += nb_correct_u
                train_nb_correct_l += nb_correct_l
                train_nb_gold_frame += nb_gold_frame
                train_nb_gold += nb_gold
                train_nb_pred += nb_pred

            if self.nb_epochs_frame_only :
                 self.nb_epochs_frame_only -=1

            print(train_nb_correct_u, train_nb_correct_l, train_nb_gold, train_nb_pred)
            train_scores_fr.append(train_nb_correct_f/train_nb_gold_frame if train_nb_gold_frame else 0)
            train_fscores_u.append( fscore(train_nb_correct_u, train_nb_gold, train_nb_pred) )            
            train_fscores_l.append( fscore(train_nb_correct_l, train_nb_gold, train_nb_pred) )            
            train_losses.append(train_loss)
            train_arc_losses.append(train_arc_loss)
            train_lab_losses.append(train_lab_loss)
            train_frame_losses.append(train_frame_loss)

            self.log_perf(log_stream, epoch, 'Train', train_loss, train_frame_loss, train_arc_loss, train_lab_loss, 
                          train_scores_fr[-1], train_fscores_u[-1], train_fscores_l[-1])

            if val_data:
                self.eval()
                # arc evaluation on validation
                val_nb_correct_f = 0
                val_nb_gold_frame = 0
                val_nb_correct_u = 0
                val_nb_correct_l = 0
                val_nb_pred = 0
                val_nb_gold = 0

                # calcul de la perte sur le validation set
                with torch.no_grad():
                    val_loss = 0
                    val_frame_loss = 0
                    val_arc_loss = 0
                    val_lab_loss = 0
                    for batch in val_data.make_batches(self.batch_size, sort_dec_length=True):
                        loss, frame_loss, arc_loss, lab_loss, nb_correct_f, nb_correct_u, nb_correct_l, nb_gold_frame, nb_gold, nb_pred = self.batch_forward_and_loss(batch)
                        val_loss += loss.item()
                        if frame_loss:
                            val_frame_loss += frame_loss
                        if arc_loss:
                            val_arc_loss += arc_loss
                        if lab_loss:
                            val_lab_loss += lab_loss
                        #predictions, nb_correct, nb_gold, nb_pred = self.predict_from_scores_and_evaluate(S_arc, S_lab, batch)    
                        
                        val_nb_correct_f += nb_correct_f
                        val_nb_gold_frame += nb_gold_frame
                        val_nb_correct_u += nb_correct_u
                        val_nb_correct_l += nb_correct_l
                        val_nb_gold += nb_gold
                        val_nb_pred += nb_pred
                        
                    print(val_nb_correct_u, val_nb_correct_l, val_nb_gold, val_nb_pred)
                    val_scores_fr.append(val_nb_correct_f/val_nb_gold_frame if val_nb_gold_frame else 0)
                    val_fscores_u.append( fscore(val_nb_correct_u, val_nb_gold, val_nb_pred) )            
                    val_fscores_l.append( fscore(val_nb_correct_l, val_nb_gold, val_nb_pred) )            
                    val_losses.append(val_loss)
                    val_frame_losses.append(val_frame_loss)
                    val_arc_losses.append(val_arc_loss)
                    val_lab_losses.append(val_lab_loss)

                self.log_perf(log_stream, epoch, 'Valid', val_loss, val_frame_loss, val_arc_loss, val_lab_loss, val_scores_fr[-1], val_fscores_u[-1], val_fscores_l[-1])

                #save the parameters if needed
                if min_val_loss  > val_loss and self.nb_epochs_frame_only==0:
                    min_val_loss = val_loss
                    best_epoch   = epoch
                    for stream in [sys.stdout, log_stream]:
                        stream.write(" saving model, current nb epochs = %d\n" % epoch)
                    if out_model_file:
                        torch.save(self.state_dict(), out_model_file)

            scheduler.step()
        for stream in [sys.stdout, log_stream]:
          stream.write("train losses: %s\n" % ' / '.join([ "%.2f" % x for x in train_losses]))
          stream.write("val   losses: %s\n" % ' / '.join([ "%.2f" % x for x in val_losses]))
          stream.write("train frame scores: %s\n" % ' / '.join([ "%.4f" % x for x in train_scores_fr]))
          stream.write("val   frame scores: %s\n" % ' / '.join([ "%.4f" % x for x in val_scores_fr]))
          stream.write("train unlab fscores: %s\n" % ' / '.join([ "%.4f" % x for x in train_fscores_u]))
          stream.write("val   unlab fscores: %s\n" % ' / '.join([ "%.4f" % x for x in val_fscores_u]))
          stream.write("train   lab fscores: %s\n" % ' / '.join([ "%.4f" % x for x in train_fscores_l]))
          stream.write("val     lab fscores: %s\n" % ' / '.join([ "%.4f" % x for x in val_fscores_l]))



        #print('drop token w emb', self.indices.iw2emb[2])

        log_heading_res, log_values_res = self.build_log_res(best_epoch, min_val_loss, val_scores_fr[best_epoch-1], val_fscores_u[best_epoch-1], val_fscores_l[best_epoch-1])
        if config_name:
            log_heading_res.append("config_name")
            log_values_res.append(config_name)

        log_stream.write('\t'.join(['RESULT'] + log_heading_res ) +'\n')
        log_stream.write('\t'.join(['RESULT'] + log_values_res ) +'\n')

        if score_csv:
            df = pd.DataFrame([log_values_res], columns=log_heading_res)
            df.to_csv(score_csv, mode='a', index=False, header=not os.path.exists(score_csv))


    def build_log_res(self, best_epoch, min_val_loss, val_score_fr, val_fscore_u, val_fscore_l):
        # Fscore for tasks a, l, ah, lh (ah = n best-scored arcs, n computed with nbheads task (h))
        self.log_heading_suff = '\t'.join([ 'RESULT', 'corpus', 'Af', 'Fu', 'Fl'] )

        featnames = ['w_emb_size', 'use_pretrained_w_emb', 
                    'l_emb_size', 'p_emb_size', 'freeze_bert', 
                    'lstm_h_size', 'lstm_dropout', 'mlp_arc_o_size','mlp_arc_dropout', 
                    'mlp_frame_h_size','batch_size', 'beta1','beta2','lr', 'lex_dropout', 
                    'mlp_lab_o_size', 'mlp_lab_dropout', 'pos_weight','dyn_weighting', 'stack' ]
        featvals = [ str(self.__dict__[f]) for f in featnames ]


        log_heading_res = ['best_epoch', 'min_val_loss', 'val_score_fr', 'val_fscore_u', 'val_fscore_l'] + featnames 
        log_values_res  = list(map(lambda x:"%5.2f" %x, [best_epoch, min_val_loss, val_score_fr, val_fscore_u, val_fscore_l])) + featvals 

        return log_heading_res, log_values_res
       

    def log_perf(self, outstream, epoch, ctype, l, frame_l, arc_l, lab_l, acc_fr, f_u, f_l):
        for stream in [sys.stdout, outstream]:
          stream.write("%s     Loss   for epoch %d: %.2f\n" % (ctype, epoch, l))
          stream.write("%s fra Loss   for epoch %d: %.2f\n" % (ctype, epoch, frame_l))
          stream.write("%s arc Loss   for epoch %d: %.2f\n" % (ctype, epoch, arc_l))
          stream.write("%s lab Loss   for epoch %d: %.2f\n" % (ctype, epoch, lab_l))
          stream.write("%s Frame ACC after epoch %d : %.2f\n" % (ctype, epoch, acc_fr))
          stream.write("%s U Fscore  after epoch %d : %.2f\n" % (ctype, epoch, f_u))
          stream.write("%s L Fscore  after epoch %d : %.2f\n\n" % (ctype, epoch, f_l))

    def log_train_hyper(self, outstream):
        for h in ['w_emb_size', 'l_emb_size', 'p_emb_size', 'lstm_h_size','mlp_arc_o_size','mlp_arc_dropout','beta1','beta2','lr','use_pretrained_w_emb']:
          outstream.write("%s : %s\n" %(h, str(self.__dict__[h])))
        outstream.write("\n")
        for h in ['batch_size', 'beta1','beta2','lr', 'nb_epochs_frame_only', 'lex_dropout', 'use_pretrained_w_emb','graph_mode']:
          outstream.write("%s : %s\n" %(h, str(self.__dict__[h])))
        outstream.write("\n")



    def predict_and_evaluate(self, test_data, parsed_file=None, csv_file=None, config_name=None):
        """ predict on test data and evaluate 
        if out_file is set, prediction will be dumped in readable format in out_file
        if data_file is set, the score from evaluation will be store in data_file.
        """

        if parsed_file:
            out_stream = open(parsed_file, 'w')
        else:
            out_stream = None

        test_nb_correct_f = 0
        test_nb_correct_u = 0
        test_nb_correct_l = 0
        test_nb_gold_frame = 0
        test_nb_gold = 0
        test_nb_pred = 0

        self.eval()

        with torch.no_grad():
            for batch in test_data.make_batches(self.batch_size, shuffle_data=False, sort_dec_length=True, shuffle_batches=False):

                nb_correct_f, nb_correct_u, nb_correct_l, nb_gold_frame, nb_gold, nb_pred = self.batch_predict_and_evaluate(batch, out_stream)

                test_nb_correct_f += nb_correct_f
                test_nb_correct_u += nb_correct_u
                test_nb_correct_l += nb_correct_l
                test_nb_gold_frame += nb_gold_frame
                test_nb_gold += nb_gold
                test_nb_pred += nb_pred

        scores_names  = ['test_score_fra', 'test_fscore_u', 'test_fscore_l']
        scores_values = list(map(lambda x:"%5.2f" %x, [test_nb_correct_f/test_nb_gold_frame, fscore(test_nb_correct_u, test_nb_gold, test_nb_pred), fscore(test_nb_correct_l, test_nb_gold, test_nb_pred)]))
        #for stream in [sys.stdout, log_stream]:
        sys.stdout.write(list(zip(scores_names, scores_values)))

        if csv_file:

            featnames = ['w_emb_size', 'use_pretrained_w_emb', 
                    'l_emb_size', 'p_emb_size', 'freeze_bert', 
                    'lstm_h_size', 'lstm_dropout', 'mlp_arc_o_size','mlp_arc_dropout', 
                    'mlp_frame_h_size','batch_size', 'beta1','beta2','lr', 'lex_dropout', 
                    'mlp_lab_o_size', 'mlp_lab_dropout', 'pos_weight', 'dyn_weighting', 'stack' ]
            featvals = [ str(self.__dict__[f]) for f in featnames ]

            if config_name:
                featnames.append("config_name")
                featvals.append(config_name)

            scores_names  = scores_names + featnames 
            scores_values = scores_values + featvals
            df = pd.DataFrame([scores_values], columns=scores_names)
            df.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))

        #self.log_test_perf(log_stream, 'test', test_nb_correct_f/test_nb_gold_frame, fscore(test_nb_correct_u, test_nb_gold, test_nb_pred), fscore(test_nb_correct_l, test_nb_gold, test_nb_pred))




    def batch_predict_and_evaluate(self, batch, out_stream):

        lengths, pad_masks, pred_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, arc_adja, lab_adja, fram_mat = batch

        #forward
        S_arc, S_lab , S_frame = self(forms, lemmas, tags, bert_tokens, bert_ftid_rkss, pad_masks, fram_mat, lengths=lengths)
        linear_pad_mask = pad_masks[:,0,:] # from [b, m, m] to [b, m]

        #predictions and scores
        if self.frame_training:
            pred_frame = torch.argmax(S_frame, dim= 2 ) #dim frame
            nb_correct_f = torch.sum((pred_frame == fram_mat).float() * (fram_mat > 0).int()).item() #  
            nb_gold_frame = torch.sum((fram_mat > 0).int()).item()
        else :
            nb_correct_f = 0
            nb_gold_frame = 0

        if self.role_training:
            #unlabeled scores
            pred_arcs = (S_arc > 0).int() * pred_masks  # b, h, d
            nb_correct_u = torch.sum((pred_arcs * arc_adja).int()).item()
            nb_gold = torch.sum(arc_adja).item()
            nb_pred = torch.sum(pred_arcs).item()

            #labeled scores
            pred_labels = torch.argmax(S_lab, dim=1) # for all arcs (not only the predicted arcs)
            # count correct labels for the predicted arcs only
            nb_correct_u_and_l = torch.sum((pred_labels == lab_adja).float() * pred_arcs).item()
        else:
            nb_correct_u = 0
            nb_correct_u_and_l = 0
            nb_gold = 0
            nb_pred = 0

        if out_stream:

            # whether sentences in batch start with a dummy root token or not
            root_form_id = self.indices.s2i('w', ROOT_FORM)
            if forms[0,0] == root_form_id:
                start = 1
                add = 0
            else:
                start = 0
                add = 1

            (batch_size, n) = forms.size() 

            for b in range(batch_size):     # sent in batch
                for d in range(start, n):   # tok in sent (skiping root token)
                    if forms[b,d] == PAD_ID:
                        break
                    out = [str(d+add), self.indices.i2s('w', forms[b,d])]

                    if self.frame_training:
                        if fram_mat[b,d]:
                            out.append(self.indices.i2s('frame', fram_mat[b,d]) )
                            out.append(self.indices.i2s('frame', pred_frame[b,d]) )
                        else:
                            out.append('_')
                            out.append('_')

                    if self.role_training:
                        # gold head / label pairs for dependent d
                        gpairs = [ [h, self.indices.i2s('label', lab_adja[b,h,d])] for h in range(n) if lab_adja[b,h,d] != 0 ] # PAD_ID or no arc == 0
                        # predicted head / label pairs for dependent d, for predicted arcs only
                        ppairs = [ [h, self.indices.i2s('label', pred_labels[b,h,d])] for h in range(n) if pred_arcs[b,h,d] != 0 ]

                        for pairs in [gpairs] + [ ppairs]:
                            if len(pairs):
                                hs, ls = zip(*pairs)
                                out.append('|'.join( [ str(x+add) for x in hs ] ))
                                out.append('|'.join(  ls  ))
                            else:
                                out.append('_')
                                out.append('_')

                    out_stream.write('\t'.join(out) + '\n')
                out_stream.write('\n')

        return nb_correct_f, nb_correct_u, nb_correct_u_and_l, nb_gold_frame, nb_gold, nb_pred