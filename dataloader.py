"""## Data loader (before conversion to indices)"""

# lecture des données
import re

ROOT_FORM = '<root>'
ROOT_LEMM = '<root>'
ROOT_TAG = '<root>'
ROOT_LABEL = '<root>'
ROOT_RK = 0

PAD_SYMB = '*PAD*'
UNK_SYMB = '*UNK*'
DROP_SYMB = '*DROP*'
# see Indices.index_new_vocab, by construction pad id will always be 0 and UNK will always be 1, drop 2
PAD_ID = 0
UNK_ID = 1
DROP_ID = 2
PAD_HEAD_RK = -1 # for tree mode only: id for head of padded dep token
 

def load_dep_trees(gold_conll_file, split_info_file, val_proportion=None):
    """
        Inputs: - conll(u) file for dependency trees
                   (first token of each sentence should contain a 'sentid' feature)
                - file with list of pairs (sentid , corpus type) (corpus types are train/dev/test)
                - val_proportion : if set to value > 0 (and <1)
                  the training file is split into train/validation,
                  (the validation part representing the provided proportion 
                  of the original training file)
        Returns 3 dictionaries (whose keys are corpus types (train/dev/test/val))
        - sentences dictionary
          - key = corpus type
          - value = list of sentences, 
                    each sentence is a list of 5-tuples [form, lemma, tag, gov, label]                                
    """
    # lecture du fichier donnant la répartition usuelle des phrases en train / dev / test
    s = open(split_info_file)
    lines = [ l[:-1].split('\t') for l in s.readlines() ]
    split_info_dic = { line[0]:line[1] for line in lines }

    # les phrases de dev / train / test
    sentences = {'dev':[], 'train':[], 'test':[]}

    max_sent_len = {'dev':0, 'train':0, 'test':0}

    stream = open(gold_conll_file)
    # debug: fake token root gets pad token as head (so its dep won't be counted in loss nor evaluation)
    #sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, ROOT_RK, ROOT_LABEL]]
    sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, PAD_HEAD_RK, PAD_SYMB]]
    sent_rk = 0
    for line in stream.readlines():
        if line.startswith('#'):
            continue
        line = line.strip()
        if not line:
            part = split_info_dic[sentid]
            sentences[part].append(sent)
            l = len(sent)
            if max_sent_len[part] < l: 
                max_sent_len[part] = l 

            #sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, ROOT_RK, ROOT_LABEL]]
            sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, PAD_HEAD_RK, PAD_SYMB]]
        else:
            cols = line.split('\t')
            # skip conllu multi-word tokens
            if '-' in cols[0]:
                continue
            form  = cols[1]
            lemma = cols[2]
            tag   = cols[4]
            gov   = int(cols[6])
            label = cols[7]
            if label == '':
                print("PROBLEM", line)
            # sentid attribute on first token
            if cols[0] == '1':
                m = re.search('sentid=([^\|=]+)', cols[5])
                if m:
                    sentid = m.group(1)
                else:
                    sentid = sent_rk
            sent.append([form, lemma, tag, gov, label])

    print("Longueur max des phrases:", max_sent_len)
    
    # decoupage du train en train + validation
    # (pour réglage du nombre d'époques)
    if val_proportion:
        (sentences['val'], sentences['train']) = split_list(sentences['train'], proportion=val_proportion)
    return sentences

def load_dep_graphs(gold_conll_file, split_info_file, canonical_gf=True, val_proportion=None):
    """
        Inputs: - conll(u) file with dependency graphs
                    (columns HEAD and GOV are pipe-separated values)
                   (first token of each sentence should contain a 'sentid' feature)
                - file with list of pairs (sentid , corpus type) (corpus types are train/dev/test)
                - canonical_gf: if set, canonical grammatical functions are used (final gf are discarded)
                - val_proportion : if set to value > 0 (and <1)
                  the training file is split into train/validation,
                  (the validation part representing the provided proportion 
                  of the original training file)
        Returns 3 dictionaries (whose keys are corpus types (train/dev/test/val))
        - sentences dictionary
          - key = corpus type
          - value = list of sentences, 
                    each sentence is a list of 5-tuples :
                    [form, lemma, tag, list of govs, list of labels]                                
    """
    # lecture du fichier donnant la répartition usuelle des phrases en train / dev / test
    s = open(split_info_file)
    lines = [ l[:-1].split('\t') for l in s.readlines() ]
    split_info_dic = { line[0]:line[1] for line in lines }

    # les phrases de dev / train / test
    sentences = {'dev':[], 'train':[], 'test':[]}

    max_sent_len = {'dev':0, 'train':0, 'test':0}

    stream = open(gold_conll_file)
    # debug: fake token root should get an empty list of governors (no reflexive link to itself!)
    #sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, [ROOT_RK], [ROOT_LABEL]]]
    sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, [], []]]
    sent_rk = 0
    for line in stream.readlines():
        if line.startswith('#'):
            continue
        line = line.strip()
        if not line:
            part = split_info_dic[sentid]
            sentences[part].append(sent)
            l = len(sent)
            if max_sent_len[part] < l: 
                max_sent_len[part] = l 

            #sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, [ROOT_RK], [ROOT_LABEL]]]
            sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, [], []]]
        else:
            cols = line.split('\t')
            # skip conllu multi-word tokens
            if '-' in cols[0]:
                continue
            form  = cols[1]
            lemma = cols[2]
            tag   = cols[4]
            govs   = cols[6]
            labels = cols[7]
            (govs, labels) = get_deep_govs(govs, labels, canonical_gf)
            if labels == '':
                print("PROBLEM", line)
            # sentid attribute on first token
            if cols[0] == '1':
                m = re.search('sentid=([^\|=]+)', cols[5])
                if m:
                    sentid = m.group(1)
                else:
                    sentid = sent_rk
            sent.append([form, lemma, tag, govs, labels])

    print("Longueur max des phrases:", max_sent_len)
    
    # decoupage du train en train + validation
    # (pour réglage du nombre d'époques)
    if val_proportion:
        (sentences['val'], sentences['train']) = split_list(sentences['train'], proportion=val_proportion)
    return sentences

def get_label(label, canonical_gf=True):
    if label.startswith("S:") or label.startswith("I:"):
        return ''
    if label.startswith('D:'):
        label = label[2:]
    if ':' in label:
        (label, cano) = label.split(':')
        if canonical_gf:
            return cano
    return label
    
def get_deep_govs(govs, labels, canonical_gf=True):
    """ works out the governors / labels in the deep_and_surf sequoia format
    - S: and I: arcs are discarded
    - get canonical function if canonical_gf is set, otherwise final functions
    
    Returns list of gov linear indices, list of corresponding labels
    
    Examples:
        input : "16|15", "S:obj.p|D:de_obj:de_obj" 
        output : [15], ["de_obj"]
       
        input : "3|6", "suj:suj|D:suj:obj" => [3,6], ['suj','obj']
    """
    govs = [int(x) for x in govs.split("|")]
    labels = [get_label(x, canonical_gf) for x in labels.split("|")]
    filtered = filter(lambda x: x[1], zip(govs, labels))
    f = list(zip(*filtered))
    if not(f):
        return [],[]
    return f[0],f[1]

def split_list(inlist, proportion=0.1, shuffle=False):
     """ partitions the input list of items (of any kind) into 2 lists, 
     the first one representing @proportion of the whole 
     
     If shuffle is not set, the partition takes one item every xxx items
     otherwise, the split is random"""
     n = len(inlist)
     size1 = int(n * proportion)
     if not(size1):
          size1 = 1
     print("SPLIT %d items into %d and %d" % (n, n-size1, size1))
     # if shuffle : simply shuffle and return slices
     if shuffle:
          # shuffle inlist (without changing the original external list
          # use of random.sample instead of random.shuffle
          inlist = sample(inlist, n)
          return (inlist[:size1], inlist[size1:])
     # otherwise, return validation set as one out of xxx items
     else:
          divisor = int(n / size1)
          l1 = []
          l2 = []
          for (i,x) in enumerate(inlist):
               if i % divisor or len(l1) >= size1:
                    l2.append(x)
               else:
                    l1.append(x)
          return (l1,l2)

def load_frame_graphs(gold_conll_file, split_info_file, canonical_gf=True, val_proportion=None, only_frame=True, other_sense=True):
    """
        Inputs: - conll(u) file with dependency graphs
                    (columns HEAD and GOV are pipe-separated values)
                   (first token of each sentence should contain a 'sentid' feature)
                - file with list of pairs (sentid , corpus type) (corpus types are train/dev/test)
                - canonical_gf: if set, canonical grammatical functions are used (final gf are discarded)
                - val_proportion : if set to value > 0 (and <1)
                  the training file is split into train/validation,
                  (the validation part representing the provided proportion 
                  of the original training file)
        Returns 3 dictionaries (whose keys are corpus types (train/dev/test/val))
        - sentences dictionary
          - key = corpus type
          - value = list of sentences, 
                    each sentence is a list of 5-tuples :
                    [form, lemma, tag, list of govs, list of labels]                                
    """
    # lecture du fichier donnant la répartition usuelle des phrases en train / dev / test
    s = open(split_info_file)
    lines = [ l[:-1].split('\t') for l in s.readlines() ]
    split_info_dic = { line[0]:line[1] for line in lines }

    # les phrases de dev / train / test
    sentences = {'dev':[], 'train':[], 'test':[]}

    max_sent_len = {'dev':0, 'train':0, 'test':0}

    stream = open(gold_conll_file)
    # debug: fake token root should get an empty list of governors (no reflexive link to itself!)
    #sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, [ROOT_RK], [ROOT_LABEL], [root_frame]]]
    sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, [], [], []]]
    sent_rk = 0
    #frames = 0
    f2w ={}
    
    for line in stream.readlines():
        if line.startswith('#'):
            continue
        line = line.strip()
        if not line:
            #pointe vers la vraie pos du frame dans la phrase
            is_frame = False
            for j, w in enumerate(sent):
                govs = w[3]
                true_govs = []
                if govs:
                    for k, num in enumerate(govs):
                        one_gov = num.split('.')[0]
                        #sent[j][3][k] = f2w[num.split('.')[0]] # bc possibly many frames, impossible
                        if one_gov in f2w:
                            true_govs.append(f2w[one_gov])
                    sent[j][3] = true_govs
                    is_frame = True

            if (only_frame and is_frame) or not only_frame:
                part = split_info_dic[sentid]
                sentences[part].append(sent)
                l = len(sent)
                if max_sent_len[part] < l: 
                    max_sent_len[part] = l 
            #sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, [ROOT_RK], [ROOT_LABEL], [root_frame]]]
            sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, [], [], []]]
            f2w ={}
        else:
            cols = line.split('\t')
            # skip conllu multi-word tokens
            if '-' in cols[0]:
                continue
            form  = cols[1]
            lemma = cols[2]
            tag   = cols[4]
            #role label search 
            roles = re.search('role=([^\|]+)', cols[5])
            if roles and 'synthead=y' in cols[5] :
                govs, labels = list(zip(*[role.split('#')[:2] for role in roles.group(1).split(',') if 'synthead=y' in role]))
            else:
                govs, labels = [], []

            #frame search
            frame = []
            """f = re.search('frame=([^\|]+,\d+#[^\|]+)', cols[5])
            if f:
                print(f.group(0))
                print(cols[5])
                frames += 1"""
            f = re.search('frame=([^\|=,]+)', cols[5])
            if f:
                key, o_frame = f.group(1).split('#')[:2]
                if 'Other_sense' not in o_frame or other_sense:
                    f2w[key] = int(cols[0])
                    frame = [o_frame]


            # sentid attribute on first token
            if cols[0] == '1':
                m = re.search('sentid=([^\|=]+)', cols[5])
                if m:
                    sentid = m.group(1)
                else:
                    sentid = sent_rk
            sent.append([form, lemma, tag, list(govs), list(labels), frame])

    print("Longueur max des phrases:", max_sent_len)
    
    # decoupage du train en train + validation
    # (pour réglage du nombre d'époques)
    if val_proportion:
        (sentences['val'], sentences['train']) = split_list(sentences['train'], proportion=val_proportion)
    #print('ici', frames)
    return sentences


def most_frequencies(sentences, w2f=None):
    if not w2f:
        w2f=default.dict()

    train_tokens = [tok for sent in sentences for tok in sent]

    (forms, lemmas, tags, heads, labels, frames) = list(zip(*train_tokens))

    for tok in train_tokens :
        form, lemma, tag, head, label, frame = tok
        if frame:
            w2f[lemma]


    true_frames = [frame[0] for frame in frames if frame]
    ms = Counter(true_frames)

    return ms.most_common(1), len(true_frames)