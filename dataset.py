"""# DataSet superclass"""

from random import shuffle
import numpy as np
from dataloader import *
import torch
import copy



class DepDataSet:
    def __init__(self, corpus_type, sentences, indices, device):
        """
        Superclass for dataset of dependency graphs or trees
        
        Inputs:
        - corpus type string (train/dev/test/val)
        - list of sentences (each sentence = list of 5-tuples form, lemma, tag, head(s), label(s))
        - indices = instance of Indices
          NB: we suppose all known symbols are already in Indices
          any other encountered symbol will be UNK_ID
        - device (batch tensors will be created directly on this device)
    
        """
        self.corpus_type = corpus_type  # train / dev / test / val
        self.size = len(sentences)
        self.indices = indices
        self.device = device

        self.sentences = sentences      # list of list of tokens (one token = 5-tuple)
        self.isentences = None          # will be set in subclasses
        self.nodrop_isentences = None     # used in case of lexical dropout (see lex_dropout)

        self.bert_tid_seqs = None
        self.bert_ftid_rks = None
        if self.indices.bert_tokenizer is not None:
          (self.bert_tid_seqs, self.bert_ftid_rks) = self.indices.bert_encode(sentences)
        

    def shuffle(self):
      """
      Rearranges all the data in a new random order
      (sentences, isentences)

      NB: ** original order is lost **
      """
      new_order = list(range(self.size))
      shuffle(new_order)
      
      for member in ['sentences','isentences']:
        self.__dict__[member] = [self.__dict__[member][rk] for rk in new_order]
      
      if self.bert_tid_seqs is not None:
        for member in ['bert_tid_seqs','bert_ftid_rks']:
          self.__dict__[member] = [self.__dict__[member][rk] for rk in new_order]
      
      if self.nodrop_isentences is not None:
          self.nodrop_isentences = [ self.nodrop_isentences[rk] for rk in new_order ]
        
        
    def sort_dec_length(self):
        """ 
        Sort all the data  by decreasing length (sentences, isentences)

        NB: ** original order is lost **
        """
        # minus length in order for argsort to provide the decreasing order
        l = [ - len(x) for x in self.isentences ]
        #print("MAX LEN", -min(l))
        order = np.argsort(l)
        self.isentences = [ self.isentences[rk] for rk in order ]
        self.sentences  = [ self.sentences[rk] for rk in order ]
        if self.bert_tid_seqs is not None:
          self.bert_tid_seqs = [ self.bert_tid_seqs[rk] for rk in order ]
          self.bert_ftid_rks = [ self.bert_ftid_rks[rk] for rk in order ]

        if self.nodrop_isentences is not None:
          self.nodrop_isentences = [ self.nodrop_isentences[rk] for rk in order ]
          
        
    def build_matrix_pad_mask(self, padded_vec_batch):
        """  
        Input : batch of vectors of size n=max_seq_len, some of which are padded at the end (shape [batch_size, n])
        Output : matrix pad_mask for the whole batch (shape [batch_size, n, n])
                 cell [b,i,j] ==  1 if BOTH i and j are not padded positions in batch instance b
                              and 0 otherwise
        """
        b_vec_pad_mask = (padded_vec_batch != PAD_ID).float() # [b, seq_len]
        b = padded_vec_batch.shape[0]
        m = padded_vec_batch.shape[1]
        v1 = b_vec_pad_mask.repeat(1,m).view(b, m, m)
        v2 = v1.transpose(-1,-2)
        return v1 * v2   # * acts as logical and

    def make_batches(self, batch_size, shuffle_data=False, sort_dec_length=False, shuffle_batches=False):
        """
        Input : batch_size
        
        Returns an iterator over batches, each batch being output 
        by pad_and_build_tensors (specific to TreeDataSet or GraphDataSet)
        """
        if shuffle_data:
            self.shuffle()
        # shuffling before length sort impacts batches of same length sentences
        if sort_dec_length:
            self.sort_dec_length()
                    
        # ranks of first sentence of each batch
        batch_rks = list(range(0, self.size, batch_size))
        
        if shuffle_batches:
            shuffle(batch_rks)
            
        for i in batch_rks:
            b_isentences = self.isentences[i : i + batch_size]
            if self.bert_tid_seqs is not None:
              b_bert_tid_seqs = self.bert_tid_seqs[i : i + batch_size]
              b_bert_ftid_rks = self.bert_ftid_rks[i : i + batch_size]
              yield(self.pad_and_build_tensors(b_isentences, b_bert_tid_seqs, b_bert_ftid_rks))
            else:
              yield(self.pad_and_build_tensors(b_isentences, None, None))

    def pad_bert_batch(self, bert_tid_seqs, bert_ftid_rkss):
      """
      Input:        
        - bert_tid_seqs : list of list of bert-token ids 
          (including *bert special symbols bos and eos,
           <root> is not included to *bert vocab)
        - bert_ftid_rkss : ranks in bert_tid_seqs of the first token of each word
          (the first is always 0 => 
          the *bert embedding for the bos symbol will be associated to the <root> word)

      Output:
          - tensor for bert token ids sequences (padded)
          - tensor for ranks of first tokens (subwords) of each word

      """
      if bert_tid_seqs == None:  
          return (None, None)
      
      bert_ipad = self.indices.bert_tokenizer.pad_token_id

      mw = max([len(x) for x in bert_ftid_rkss]) # max length in words (including bert special toks, but not <root>)
      mt = max([len(x) for x in bert_tid_seqs])  # max length in tokens (subwords)
      # PB!
      # a padded word seq may not lead to a padded subword seq
      # example:
      # 3 words : European Medicines Agency
      # 2 words + 1 pad word : PROPRIETES PHARMACOLOGIQUES
      #  but :
      # ['<s>', 'European</w>', 'Medi', 'cines</w>', 'Agency</w>', '</s>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
      # ['<s>', 'PROPRI', 'ET', 'ES</w>', 'PH', 'AR', 'MA', 'CO', 'LOGI', 'QUES</w>', '</s>']
      # ==> # add 1 token length to ensure there is at least one padding subword
      mt = mt + 1 
      b_bert_tokens = []
      b_bert_ftid_rkss = []
      for i,bert_tid_seq in enumerate(bert_tid_seqs):
          bert_ftid_rks = bert_ftid_rkss[i]
          lt = len(bert_tid_seq)
          lw = len(bert_ftid_rks)
          b_bert_tokens.append( bert_tid_seq + (mt - lt)*[bert_ipad] )
          b_bert_ftid_rkss.append( bert_ftid_rks + (mw -lw) * [mt - 1]) # ftid of padded words can be the rank of the last tid (mt-1) 
      
      b_bert_tokens = torch.tensor(b_bert_tokens, device=self.device)
      b_bert_ftid_rkss = torch.tensor(b_bert_ftid_rkss, device=self.device)
      
      return (b_bert_tokens, b_bert_ftid_rkss)
      
    def lex_dropout(self, lex_dropout_rate):
        """
        independent dropout of word, lemma and pos
        """
        # nodrop_isentences is None iff the sentences have never undergone dropout yet
        if self.nodrop_isentences == None:
          self.nodrop_isentences = self.isentences
        # should be applicable several times  (application on nodrop_isentences)
        self.isentences = self.indices.lex_dropout_isentences(self.nodrop_isentences, lex_dropout_rate)




"""## GraphDataSet"""

class DepGraphDataSet(DepDataSet):
    """ Dependency graphs dataset
    one token = 5-tuple (form, lemma, tag, list of heads, list of labels)
    """
    def __init__(self, *args, **kwargs):
        super(DepGraphDataSet, self).__init__(*args, **kwargs)

        # sentences (list of token lists), with all strings converted to ids
        self.isentences = self.indices.convert_frame_symbols_to_indices(self.sentences)
        
        # total number of token pairs in dataset (max number of arcs)
        self.nb_tok_pairs = sum([ len(s)**2 for s in self.sentences ])
        # total number of arcs in dataset
        self.nb_arcs = sum([ len(tok[3]) for s in self.sentences for tok in s])
              
  
    def pad_and_build_tensors(self, isentences, bert_tid_seqs=None, bert_ftid_rkss=None):
        """
        Input : 
        - isentences : list of list of itoks (1 itok = a 5-tuple of corresponding indices)
        - bert_tid_seqs : list of list of bert-token ids
        - bert_ftid_rkss : ranks in bert_tid_seqs of the first token of each word

        Output : padded tensors for each type of symbol

        with b = len of isentences, m = max seq length in isentences
        - b_lengths : tensor of shape [b] : true lengths of isentences (before padding)
        - b_pad_mask : tensor of shape [b, m, m] : pad matrices for each batch sample
              cell [k,i,j] ==  1 if BOTH i and j are not padded positions in batch sample k
                           and 0 otherwise
        - 3 tensors of shape [b, m] for word form / lemma / pos id sequences

        - corresponding batch of arc adjacency matrices 
          [k, h, d] = 0 if arc h-->d is not gold OR either h or d is a padded token
                    = 1 otherwise
        - corresponding batch of label adjacency matrices (same meaning of zero cells)
          [k, i, j] = 0 if arc h-->d is not gold OR either h or d is a padded token
                    = label id of gold arc h-->d
        
        - if bert stuff is not None:
          - tensor for bert token ids sequences (padded)
          - tensor for ranks of first tokens (subwords) of each word

        """
        m = max([len(x) for x in isentences])

        b_forms = []
        b_lemmas = []
        b_tags = []
        
        # padding info (redundant)
        # real lengths of sequences in batch
        b_lengths = []
        b_pad_mask = []

        # dependencies info : batch_size * m * m   
        # cell A[b,h,d] : arc existence in batch instance b, with head h and dependent d
        # NB: zero cells hold for 
        #     - unexistent arc in gold graph 
        #     - OR padded cell (padded dep or head)
        b_arc_adja = np.zeros((len(isentences), m, m))          # float needed in BCELoss
        b_lab_adja = np.zeros((len(isentences), m, m), 'int64') # int needed in crossentropyloss
        b_fram_mat = np.zeros((len(isentences), m), 'int64')
        
        for (b,sent) in enumerate(isentences):
            (forms, lemmas, tags, headss, labelss, frames) = zip(*sent)
            l = len(forms)
            b_lengths.append(l)
            pad_mask_matrix = []
            # not very nice
            for (d,heads) in enumerate(headss):
                pad_mask_matrix.append( l*[1] + (m-l)*[0] )
                # better to loop over heads, cf. they are few
                for (i,h) in enumerate(heads): 
                    b_arc_adja[b,h,d] = 1
                    b_lab_adja[b,h,d] = labelss[d][i]
                if frames[d]:
                    b_fram_mat[b,d] = frames[d][0]

            if l < m:
                p = m*[0]
                pad_mask_matrix.extend( (m-l)* [p])
            b_pad_mask.append(pad_mask_matrix)
            
            b_forms.append( list(forms) + (m - l)*[PAD_ID] )
            b_lemmas.append( list(lemmas) + (m - l)*[PAD_ID] )
            b_tags.append( list(tags) + (m - l)*[PAD_ID] )
            
        b_forms  = torch.tensor(b_forms, device=self.device)
        b_lemmas = torch.tensor(b_lemmas, device=self.device)
        b_tags   = torch.tensor(b_tags, device=self.device)
        b_lengths = torch.tensor(b_lengths, device=self.device)
        # padding matrices for sents in batch
        b_pad_mask = torch.tensor(b_pad_mask, device=self.device) # (b, h, d)
        # dependencies adjacency matrixes for sents in batch
        b_arc_adja = torch.from_numpy(b_arc_adja).to(self.device)
        b_lab_adja = torch.from_numpy(b_lab_adja).to(self.device)
        #
        b_fram_mat = torch.from_numpy(b_fram_mat).to(self.device) # (b, m)
        
        #test pour target location frame
        #print(torch.nonzero(b_fram_mat))
        #print(torch.nonzero(b_fram_mat,as_tuple=True))
        # predicates & padding matrices for sents in batch
        # b_arc_adja : (b, h, d) => a : (b, h)
        # counting the number of dependents of each token
        a = torch.sum(b_arc_adja, dim = 2) # b, h
        #b_pred_mask = (a > 0).int()
        #avec loss arc_adja
        # (a > 0).int().unsqueeze(2) => boolean tensor (b, h, 1), with ones when the h evokes a frame, and 0 otherwise
        # (a > 0).int().unsqueeze(2) * b_pad_mask => boolean tensor (b, h, d), with ones when the h evokes a frame and d is not padded, and 0 otherwise
        b_pred_mask = (a > 0).int().unsqueeze(2) * b_pad_mask
        
        # bert stuff, if any
        b_bert_tokens, b_ftid_rkss = self.pad_bert_batch(bert_tid_seqs, bert_ftid_rkss)
        
        return (b_lengths, b_pad_mask, b_pred_mask, b_forms, b_lemmas, b_tags, b_bert_tokens, b_ftid_rkss, b_arc_adja, b_lab_adja, b_fram_mat)
