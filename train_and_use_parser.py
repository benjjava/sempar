import torch
import os
import sys
import argparse


from dataloader import *
from indices import *
from dataset import *
from biaffine_parser import *


from transformers import AutoModel, AutoTokenizer, AutoConfig

        
# ---------- MAIN ------------------
if __name__ == "__main__":

    usage = """ Biaffine graph parser (Dozat et al. 2018) for frame, based on Marie Candito code"""

    # read arguments
    argparser = argparse.ArgumentParser(usage = usage)

    argparser.add_argument('stack_mode', choices=['no','pipeline','simple','cat'], help="Stack option for the multitask learning. Default=simple'", default="simple")
    argparser.add_argument('conll_file', help='Contains either train/dev/test sentences (if split_info_file is provided), or should contain training sentences for train mode, or sentences to parse in test mode.', default=None)
    argparser.add_argument('split_info_file', help='split info file (each line = sentence id, tab, corpus type (train, dev, test)', default=None)
    argparser.add_argument('out_dir', help='output directory', default='../resultats')


    argparser.add_argument('-b', '--batch_size', help='batch size. Default=16', type=int, default=16)
    argparser.add_argument('-e', '--nb_epochs', help='Max nb epochs. Default=40', type=int, default=40)
    argparser.add_argument('-o', '--only_frame_epochs', help='First epochs only for frame identification. Default=0', type=int, default=0)
    argparser.add_argument('-w', '--w_emb_size', help='size of word embeddings. Default=100', type=int, default=100)
    argparser.add_argument('-l', '--l_emb_size', help='size of lemma embeddings. Default=100', type=int, default=100)
    argparser.add_argument('-p', '--p_emb_size', help='size of POS embeddings. Default=100', type=int, default=100)
    argparser.add_argument('--lstm_h_size', help='size of lstm embeddings. Default=600', type=int, default=600)
    argparser.add_argument('--lstm_dropout', help='lstm dropout. Default=0.33', type=float, default=0.33)
    argparser.add_argument('--mlp_arc_o_size', help='size of arc mlp after lstm. Default=500', type=int, default=500)
    argparser.add_argument('--mlp_lab_o_size', help='size of lab mlp after lstm. Default=500', type=int, default=500)
    argparser.add_argument('--mlp_frame_h_size', help='size of frame mlp hidden after lstm. Default=400', type=int, default=400)
    argparser.add_argument('--w_emb_file', help='If not "None", pre-trained word embeddings file. NB: first line should contain nbwords embedding size', default='None')
    argparser.add_argument('-s', '--other_sense', action="store_true", help='Whether to parse Other_sense. Default=False', default=False)
    argparser.add_argument('-c', '--config_name', help='Name of next files')

    argparser.add_argument('--bert_name', help='huggingface *bert model name. If not "None", will be used as pretrained-LM. Default:flaubert/flaubert_base_cased', default="flaubert/flaubert_base_cased")
    argparser.add_argument('-f', '--freeze_bert', action="store_true", help='Whether to freeze *bert parameters. Default=False', default=False)
    argparser.add_argument('-u', '--use_bias', action="store_true", help='Whether to add bias in all internal MLPs. Default=False', default=False)
    argparser.add_argument('--dyn_weighting', action="store_true", help='Whether to use dyn weight', default=False)
    argparser.add_argument('--frame_training', action="store_true", help='Whether to parse frame', default=False)
    argparser.add_argument('--role_training', action="store_true", help='Whether to parse arcs and labels roles', default=False)

    argparser.add_argument('--device_id', help='in train mode only: GPU cuda device id (in test mode: device is read in model). Default=0', type=int, default=1)
    argparser.add_argument('--pos_arc_weight', help='(for graph mode only) weight for positive arcs in binary cross-entropy. Default=1.5', type=float, default=1.5)
    argparser.add_argument('-r', '--learning_rate', help='learning rate, default=0.0001', type=float, default=0.00001)
    argparser.add_argument('-d', '--lex_dropout', help='lexical dropout rate, default=0.33', type=float, default=0.33)

    argparser.add_argument('--out_csv_dev', help='csv file with all previous score', default=None)
    argparser.add_argument('--out_csv_test', help='csv file with all previous score', default=None)
    argparser.add_argument('--out_parsed_file', help='Pertains in test mode only. If set to non None, filename into which predictions will be dumped', default=None)


    args = argparser.parse_args()



    model_file = args.out_dir+'/model'

    # before anything: check whether we will be able to dump the model
    pdir = os.path.dirname(model_file)
    if not pdir: pdir = '.'
    # if parent dir is writable
    if not os.access(pdir, os.W_OK):
        exit("Model file %s will not be writable!\n" % model_file)

    # --------------------- DEVICE ---------------------
    # si un GPU est disponible on l'utilisera
    if torch.cuda.is_available():
      # objet torch.device          
      DEVICE = torch.device("cuda:"+str(args.device_id))
    
      print('There are %d GPU(s) available.' % torch.cuda.device_count())    
      device_id = args.device_id #torch.cuda.current_device()
      gpu_properties = torch.cuda.get_device_properties(device_id)
      print("We will use GPU %d (%s) of compute capability %d.%d with %.2fGb total memory.\n" % 
            (device_id,
             gpu_properties.name,
             gpu_properties.major,
             gpu_properties.minor,
             gpu_properties.total_memory / 1e9))

    else:
      print('No GPU available, using the CPU instead.')
      DEVICE = torch.device("cpu")



    # ------------- DATA (WITHOUT INDICES YET) ------------------------------
	print('loading sentences...')
	sentences = load_frame_graphs(args.conll_file, args.split_info_file,val_proportion=None,other_sense=args.other_sense)

	if args.w_emb_file != 'None':
		w_emb_file = args.w_emb_file
		use_pretrained_w_emb = True
	else:
		w_emb_file = None
		use_pretrained_w_emb = False



    if args.bert_name != 'None':
		bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
		bert_config = AutoConfig.from_pretrained(args.bert_name)
		bert_model = AutoModel.from_pretrained(args.bert_name,return_dict=True)
	else:
		bert_tokenizer = None
		args.bert_name = None

	# ------------- INDICES ------------------------------
    # indices are defined on train sentences only
    print('indices...')
    indices = Indices(sentences['train'], w_embeddings_file=w_emb_file, bert_tokenizer=bert_tokenizer)

    data = {}
	for part in sentences.keys():
    	data[part] = DepGraphDataSet(part, sentences[part], indices, DEVICE)

    # ------------- THE PARSER ---------------------------
    biaffineparser = BiAffineParser(indices, DEVICE, 
    								stacked=args.stack_mode,
                                    w_emb_size=args.w_emb_size,
                                    l_emb_size=args.l_emb_size,
                                    p_emb_size=args.p_emb_size,
                                    lstm_h_size=args.lstm_h_size,
                                    lstm_dropout=args.lstm_dropout,
                                    mlp_frame_h_size=args.mlp_frame_h_size,
                                    mlp_arc_o_size=args.mlp_arc_o_size, 
                                    mlp_lab_o_size=args.mlp_lab_o_size,
                                    use_pretrained_w_emb=use_pretrained_w_emb, 
                                    use_bias=args.use_bias,
                                    bert_model=bert_model,
                                    freeze_bert=args.freeze_bert,
                                    dyn_weighting=args.dyn_weighting,
    )


    #train_data = data['train']
    val_data = data['dev']

    logstream = open(args.out_dir+'/log_train', 'w')

    biaffineparser.train_model(val_data, val_data, logstream,
                               args.nb_epochs,
                               args.batch_size,
                               args.learning_rate,
                               args.lex_dropout,
                               out_model_file=model_file,
                               score_csv=args.out_csv_dev,
								nb_epochs_frame_only=args.only_frame_nb_epoch, 
								frame_training=args.frame_training, 
								role_training=args.role_training,
								pos_weight=args.pos_arc_weight,
                                config_name=args.config_name)

    logstream.close()


    biaffineparser.load_state_dict(model_file)


    biaffineparser.predict_and_evaluate(data['test'], args.out_dir+'/parsed.txt', args.out_csv_test )