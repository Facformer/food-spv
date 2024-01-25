import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import DataEmbedding
from layers.Embed import DataEmbedding_mark
from layers.Transformer_EncDec import Decoder,DecoderLayer,Encoder,EncoderLayer,ConvLayer
from layers.SelfAttention_Family import FullAttention,AttentionLayer


class Model(nn.Module):
	def __init__(self,configs):
		super(Model,self).__init__()
		self.pred_len = configs.pred_len
		self.output_attention = configs.output_attention
		self.enc_embedding = DataEmbedding(configs.enc_in,configs.d_model,configs.embed,configs.dropout)
		self.dec_embedding = DataEmbedding(configs.dec_in,configs.d_model,configs.embed,configs.dropout)
		self.encoder = Encoder(
			[EncoderLayer(
				AttentionLayer(
				FullAttention(False,configs.factor,attention_dropout=configs.dropout,
						output_attention=configs.output_attention),configs.d_model,configs.n_heads),
				configs.d_model,
				configs.d_ff,
				dropout=configs.dropout,
				activation=configs.activation
				) for l in range(configs.e_layers)],
			norm_layer = torch.nn.LayerNorm(configs.d_model))
		self.decoder = Decoder(
			[DecoderLayer(
				AttentionLayer(FullAttention(True,configs.factor,attention_dropout=configs.dropout,output_attention=False),
					configs.d_model,configs.n_heads),
				AttentionLayer(FullAttention(False,configs.factor,attention_dropout=configs.dropout,output_attention=False),
					configs.d_model,configs.n_heads),
				configs.d_model,
				configs.d_ff,
				dropout=configs.dropout,
				activation=configs.activation,
				) for l in range(configs.d_layers)],
			norm_layer = torch.nn.LayerNorm(configs.d_model),
			projection=nn.Linear(configs.d_model,configs.c_out,bias=True)
			)
	def forward(self,x_enc,x_dec,enc_self_mask=None,dec_self_mask=None,dec_enc_mask=None):
		enc_out = self.enc_embedding(x_enc)
		enc_out,attns = self.encoder(enc_out,attn_mask=enc_self_mask)
		dec_out = self.dec_embedding(x_dec)
		dec_out = self.decoder(dec_out,enc_out,x_mask=dec_self_mask,cross_mask=dec_self_mask)
			return dec_out[:,-self.pred_len:,:],attns
			return dec_out[:,-self.pred_len:,:]