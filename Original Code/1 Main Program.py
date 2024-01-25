import numpy as np
import random
import torch
import os
import argparse
import warnings
warnings.filterwarnings('ignore')
from exp.exp_main import Exp_Main
from exp.exp_main_ml import Exp_Main_ML

def main(args):
	fix_feed = 2023
	random.seed(fix_feed)
	torch.manual_seed(fix_feed)
	np.random.seed(fix_feed)
	if args.is_training:
		for ii in range(args.itr):
			setting = '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
				args.model_id,
				args.model,
				args.data,
				args.seq_len,
				args.label_len,
				args.pred_len,
				args.d_model,
				args.n_heads,
				args.e_layers,
				args.d_layers,
				args.d_ff,
				args.factor,
				args.embed,
				args.distil,
				args.des,ii)
			if args.model in ['Facformer','Autoformer','Informer','Transformer','LSTM','RNN']:
				exp = Exp_Main(args)
			elif args.model in ['KNN','RF']:
				exp = Exp_Main_ML(args)
			print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
			if args.use_mark:
				print('use marked data')
				exp.train_mark(setting)
			else:
				exp.train(setting)
			print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
			if args.use_mark:
				exp.test_mark(setting,load=1)
			else:
				exp.test(setting,load=1)
			
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='for shelf lift using ml')
	parser.add_argument('--is_training',type=int,default=1,required=False,help='status')
	parser.add_argument('--model_id',type=str,default='test',required=False,help='model id')
	parser.add_argument('--model',type=str,default='Facformer',required=False,help=['Facformer','Autoformer','Transformer','LSTM','RNN','RF','KNN'])
	parser.add_argument('--data',type=str,default='shrimp_paste',help='choose a data,name of data')
	parser.add_argument('--data_path',type=str,default='./data/shrimp_paste/',help='path of the ori data')
	parser.add_argument('--use_constant',type=bool,default=False,help='whether to use constant temperature data')
	parser.add_argument('--use_mark',type=bool,default=False,help='whether use mark to mark T and D')
	parser.add_argument('--scale',type=bool,default=True,help='whether use scale')
	parser.add_argument('--embed',type=str,default='embed_T',help='time features encoding, options:[fixed, learned,embed_T]')
	parser.add_argument('--checkpoints',type=str,default='./checkpoints/',help='location of model checkpoints')
	parser.add_argument('--train_mode',type=str,default='generation',choices=['generation','once'])
	parser.add_argument('--fine_tuning',type=bool,default=False,help='whether support fine tuning')
	parser.add_argument('--batch_size',type=int,default=6,help='batch size')
	parser.add_argument('--seq_len',type=int,default=30,help='input sequence length')
	parser.add_argument('--label_len',type=int,default=15,help='start token length')
	parser.add_argument('--pred_len',type=int,default=15,help='prediction sequence length')
	parser.add_argument('--enc_in',type=int,default=4,help='encoder input size')
	parser.add_argument('--dec_in',type=int,default=4,help='decoder input size')
	parser.add_argument('--enc_mark_in',type=int,default=2,help='encoder input size')
	parser.add_argument('--dec_mark_in',type=int,default=2,help='decoder input size')
	parser.add_argument('--c_out',type=int,default=4,help='output size')
	parser.add_argument('--d_model',type=int,default=512,help='dimension of model')	
	parser.add_argument('--n_heads',type=int,default=8,help='num of heads')
	parser.add_argument('--dropout',type=float,default=0.05,help='dropout')
	parser.add_argument('--e_layers',type=int,default=2,help='num of encoder layers')
	parser.add_argument('--d_layers',type=int,default=1,help='num of decoder layers')
	parser.add_argument('--d_ff',type=int,default=2048,help='dimension of fcn')
	parser.add_argument('--activation',type=str,default='gelu',help='activation')
	parser.add_argument('--factor',type=int,default=1,help='attn factor')
	parser.add_argument('--distil',action='store_false',help='whether to use distilling in encoder, using this argument means not using distilling',default=True)
	parser.add_argument('--num_workers',type=int,default=0,help='data loader num workers')
	parser.add_argument('--itr',type=int,default=1,help='experiments times')
	parser.add_argument('--patience',type=int,default=3,help='early stopping patience')	
	parser.add_argument('--learning_rate',type=float,default=0.0001,help='optimizer learning rate')
	parser.add_argument('--train_epochs',type=int,default=10,help='train epochs')
	parser.add_argument('--lradj',type=str,default='type1',help='adjust learning rate')
	parser.add_argument('--output_attention',action='store_true',default=True,help='whether to output attention in encoder')
	parser.add_argument('--des',type=str,default='test',help='exp description')
	parser.add_argument('--use_amp',action='store_true',help='use automatic mixed precision training',default=False)
	parser.add_argument('--use_gpu',type=bool,default=True,help='use gpu')
	parser.add_argument('--gpu',type=int,default=0,help='gpu')
	parser.add_argument('--use_multi_gpu',action='store_true',help='use multiple gpus',default=False)
	parser.add_argument('--devices',type=str,default='0,1,2,3',help='device ids of multile gpus')
	parser.add_argument('--moving_avg',type=int,default=7,help='window size of moving average')
	parser.add_argument('--hidden_size',type=int,default=512,help='dimension of model')
	parser.add_argument('--n_layers',type=int,default=2,help='dimension of model')
	
	args = parser.parse_args()
	if args.model == 'Facformer':
		args.use_mark=True
		args.enc_in = 2
		args.dec_in = 2
	args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
	if args.use_gpu and args.use_multi_gpu:
		args.devices = args.devices.replace(' ','')
		device_ids = args.devices.split(',')
		args.device_ids = [int(id_) for id_ in device_ids]
		args.gpu = args.device_ids[0]
        
	main(args)