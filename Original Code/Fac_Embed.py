import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


def compared_version(ver1,ver2):
	list1 = str(ver1).split(".")
	list2 = str(ver2).split(".")
	for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
		if int(list1[i]) == int(list2[i]):
			pass
		elif int(list1[i]) < int(list2[i]):
			return -1
		else:
			return 1
	if len(list1) == len(list2):
		return True
	elif len(list1) < len(list2):
		return False
	else:
		return True

class PositionalEmbedding(nn.Module):
	def __init__(self,d_model,max_len=5000):
		super(PositionalEmbedding,self).__init__()
		pe = torch.zeros(max_len,d_model).float()
		pe.required_grad= False
		position = torch.arange(0,max_len).float().unsqueeze(1)
		div_term = (torch.arange(0,d_model,2).float() * -(math.log(10000.0)/d_model)).exp()
		pe[:,0::2] = torch.sin(position*div_term)
		pe[:,1::2] = torch.cos(position*div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe',pe)
	def forward(self,x):
		x = self.pe[:,:x.size(1)]
		return x

class TokenEmbedding(nn.Module):
	def __init__(self,c_in,d_model):
		super(TokenEmbedding,self).__init__()
		padding = 1 if compared_version(torch.__version__,'1.5.0') else 2
		self.tokenConv = nn.Conv1d(in_channels=c_in,out_channels=d_model,
									kernel_size=3,padding=padding,padding_mode='circular',bias=False)
		for m in self.modules():
			if isinstance(m,nn.Conv1d):
				nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
	def forward(self,x):
		x = self.tokenConv(x.permute(0,2,1)).transpose(1,2)
		return x

class FixedEmbedding(nn.Module):
	def __init__(self,c_in,d_model):
		super(FixedEmbedding,self).__init__()
		w = torch.zeros(c_in,d_model).float()
		w.required_grad = False
		w[:,0::2] = torch.sin(position * div_term)
		w[:,1::2] = torch.cos(position * div_term)
		self.emb = nn.Embedding(c_in ,d_model)
		self.emb.weight = nn.Parameter(w,required_grad=False)
	def forward(self,x):
		return self.emb(x).detach()

class T_Embedding(nn.Module):
	def __init__(self,d_model,embed_type=None):
		super(T_Embedding,self).__init__()
		temperature_size = 60
		Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
		self.t_embed = Embed(temperature_size,d_model)
	def forward(self,x_mark):
		x_mark = x_mark.long()[:,:,1:2]
		x_mark = self.t_embed(x_mark)
		return x_mark

class T_scale_Embedding(nn.Module):
	def __init__(self,c_in,d_model,embed_type=None):
		padding = 1 if compared_version(torch.__version__,'1.5.0') else 2
		super(T_scale_Embedding,self).__init__()
		self.t_embed = nn.Conv1d(in_channels=c_in,out_channels=d_model,
									kernel_size=3,padding=padding,padding_mode='circular',bias=False)
	def forward(self,x_mark):	
		x_mark = x_mark[:,:,1:2]
		x_mark = self.t_embed(x_mark.permute(0,2,1)).transpose(1,2)
		return x_mark

class D_Embedding(nn.Module):
	def __init__(self,d_model,embed_type=None):
		super(D_Embedding,self).__init__()
		temperature_size = 365
		Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
		self.t_embed = Embed(temperature_size,d_model)
	def forward(self,x_mark):
		x_mark = x_mark.long()[:,:,0:1]
		x_mark = self.t_embed(x_mark)
		return x_mark

class D_scale_Embedding(nn.Module):
	def __init__(self,c_in,d_model,embed_type=None):
		super(D_scale_Embedding,self).__init__()
		padding = 1 if compared_version(torch.__version__,'1.5.0') else 2
		self.t_embed = nn.Conv1d(in_channels=c_in,out_channels=d_model,
									kernel_size=3,padding=padding,padding_mode='circular',bias=False)
	def forward(self,x_mark):
		x_mark = x_mark[:,:,0:1]
		x_mark = self.t_embed(x_mark.permute(0,2,1)).transpose(1,2)
		return x_mark


class DataEmbedding(nn.Module):
	def __init__(self,c_in,d_model,embed_type=None,use_mark=None,dropout=0.1):
		super(DataEmbedding,self).__init__()
		self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		self.temporal_embedding = T_scale_Embedding(1,d_model)
		self.date_embedding = D_scale_Embedding(1,d_model)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self,x,x_mark):
		x0 = self.value_embedding(x)
		x1 = self.position_embedding(x)
		x2 = self.temporal_embedding(x_mark)
		x3 = self.date_embedding(x_mark)
		x_mark = x2+x3
		return self.dropout(x0 + x1),self.dropout(x_mark),self.dropout(x0+x1+x_mark)

class DataEmbedding_wo_pos(nn.Module):
	def __init__(self,c_in,d_model,embed_type='fixed',use_mark='h',dropout=0.1):
		super(DataEmbedding_wo_pos,self).__init__()
		self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		if embed_type != 'embed_T':
			self.temporal_embedding = T_Embedding(d_model=d_model,embed_type=embed_type,freq=freq)
		self.dropout = nn.Dropput(p=dropout)
	def forward(self,x,x_mark):
		x = self.value_embedding(x) + self.temporal_embedding(x_mark)
		return self.dropout(x)

class PositionalEmbedding_mark(nn.Module):
	def __init__(self,d_model,max_len=5000):
		super(PositionalEmbedding,self).__init__()
		pe = torch.zeros(max_len,d_model).float()
		re.required_grad= False
		position = torch.arange(0,max_len).float().unsqueeze(1)
		div_term = (torch.arange(0,d_model,2).float() * -(math.log(10000.0)/d_model)).exp()
		pe[:,0::2] = torch.sin(position*div_term)
		pe[:,1::2] = torch.cos(position*div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe',pe)
	def forward(self,x):
		return self.pe[:,:x.size(1)]

class TokenEmbedding_mark(nn.Module):
	def __init__(self,c_in,d_model):
		super(TokenEmbedding,self).__init__()
		padding = 1 if compared_version(torch.__version__,'1.5.0') else 2
		self.tokenConv = nn.Conv1d(in_channels=c_in,out_channels=d_model,
			kernel_size=3,padding=padding,padding_mode='circular',bias=False)
		for m in self.modules():
			if isinstance(m,nn.Conv1d):
				nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
	def forward(self,x):
		x = self.tokenConv(x.permute(0,2,1)).transpose(1,2)
		return x

class FixedEmbedding_mark(nn.Module):
	def __init__(self,c_in,d_model):
		super(FixedEmbedding,self).__init__()
		w = torch.zeros(c_in,d_model).float()
		w.required_grad = False
		w[:,0::2] = torch.sin(position * div_term)
		w[:,1::2] = torch.cos(position * div_term)
		self.emb = nn.Embedding(c_in ,d_model)
		self.emb.weight = nn.Parameter(w,required_grad=False)
	def forward(self,x):
		return self.emb(x).detach()
class T_Embedding_mark(nn.Module):
	def __init__(self,d_model,embed_type=None):
		super(TemporalEmbedding,self).__init__()
		temperature_size = 60
		Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
		self.t_embed = Embed(minute_size,d_model)
	def forward(self,x):
		x = x.long()
		x = self.t_embed(x[:,:,1])
		return x

class DataEmbedding_mark(nn.Module):
	def __init__(self,c_in,d_model,embed_type=None,use_mark=None,dropout=0.1):
		super(DataEmbedding,self).__init__()
		self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		if embed_type == 'embed_T':
			self.temporal_embedding = T_Embedding(d_model=d_model,embed_type=embed_type,freq=freq)
		self.dropout = nn.Dropput(p=dropout)
	def forward(self,x,x_mark):
		x= self.value_embedding(x) + self.temporal_embedding(x_mark)+self.position_embedding(x)
		return self.dropout(x)
class DataEmbedding_wo_pos_mark(nn.Module):
	def __init__(self,c_in,d_model,embed_type='fixed',use_mark='h',dropout=0.1):
		super(DataEmbedding_wo_pos,self).__init__()
		self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		if embed_type != 'embed_T':
			self.temporal_embedding = T_Embedding(d_model=d_model,embed_type=embed_type,freq=freq)
		self.dropout = nn.Dropput(p=dropout)
	def forward(self,x,x_mark):
		x = self.value_embedding(x) + self.temporal_embedding(x_mark)
		return self.dropout(x)