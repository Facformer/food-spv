import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
	def __init__(self,c_in):
		super(ConvLayer,self).__init__()
		self.downConv = nn.Conv1d(
			in_channels=c_in,
			out_channels=c_in,
			kernel_size=3,
			padding=2,
			padding_mode='circular')
		self.norm = nn.BatchNorm1d(c_in)
		self.activation = nn.ELU()
		self.maxPool = nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
	def forward(self,x):
		x = self.downConv(x.permute(0,2,1))
		x = self.norm(x)
		x = self.activation(x)
		x = self.maxPool(x)
		x = x.transpose(1,2)
		return x

class EncoderLayer(nn.Module):
	def __init__(self,attention1,attention2,mark_attention,d_model,d_ff=None,dropout=0.1,activation="relu"):
		super(EncoderLayer,self).__init__()
		d_ff = d_ff or 4 * d_model
		self.attention1 = attention1
		self.attention2 = attention2
		self.mark_attention = mark_attention
		self.conv1 = nn.Conv1d(in_channels=d_model,out_channels=d_ff,kernel_size=1)
		self.conv2 = nn.Conv1d(in_channels=d_ff,out_channels=d_model,kernel_size=1)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.activation = F.relu if activation == 'relu' else F.gelu

	def forward(self,x,xm,xxm,attn_mask=None):
		new_x, attn = self.attention1(x,x,x,attn_mask=attn_mask)
		new_xm,attn = self.mark_attention(xm,xm,xm,attn_mask=attn_mask)
		new_xxm,attn= self.attention2(xxm,xxm,xxm,attn_mask=attn_mask)
		x = x + self.dropout(new_x)
		y = x = self.norm1(x)
		y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
		y = self.dropout(self.conv2(y).transpose(-1,1))
		xm = xm + self.dropout(new_xm)
		ym = xm = self.norm1(xm)
		ym = self.dropout(self.activation(self.conv1(ym.transpose(-1,1))))
		ym = self.dropout(self.conv2(ym).transpose(-1,1))
		xxm = xxm + self.dropout(new_xxm)
		yxm = xxm = self.norm1(xxm)
		yxm = self.dropout(self.activation(self.conv1(yxm.transpose(-1,1))))
		yxm = self.dropout(self.conv2(yxm).transpose(-1,1))
		return self.norm2(x+y),self.norm2(xm+ym),self.norm2(xxm+yxm)

class Encoder(nn.Module):
	def __init__(self,attn_layers,conv_layers=None,norm_layer=None):
		super(Encoder,self).__init__()
		self.attn_layers = nn.ModuleList(attn_layers)
		self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
		self.norm = norm_layer
	def forward(self,x,xm,xxm,attn_mask=None):
		for attn_layer in self.attn_layers:
			x,xm,xxm = attn_layer(x,xm,xxm,attn_mask=attn_mask)
		if self.norm is not None:
			x = self.norm(x)
			xm = self.norm(xm)
			xxm = self.norm(xxm)
		return x,xm,xxm


class DecoderLayer(nn.Module):
	def __init__(self,self_attention1,cross_attention1,self_attention2,cross_attention2,
				mark_attention,mark_cross_attention,fac_attention,
				d_model,d_ff=None,dropout=0.1,activation='relu'):
		super(DecoderLayer,self).__init__()
		d_ff = d_ff or 4*d_model
		self.self_attention1 = self_attention1
		self.cross_attention1 = cross_attention1
		self.self_attention2 = self_attention2
		self.cross_attention2 = cross_attention2
		self.mark_attention = mark_attention
		self.mark_cross_attention = mark_cross_attention
		self.fac_attention = fac_attention
		self.conv1 = nn.Conv1d(in_channels=d_model,out_channels=d_ff,kernel_size=1)
		self.conv2 = nn.Conv1d(in_channels=d_ff,out_channels=d_model,kernel_size=1)
		self.conv3 = nn.Conv1d(in_channels=d_model,out_channels=d_ff,kernel_size=1)
		self.conv4 = nn.Conv1d(in_channels=d_ff,out_channels=d_model,kernel_size=1)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.activation = F.relu if activation =='relu' else F.gelu
	def forward(self,xp,xpm,xxpm,x_cross,xm_cross,xxm_cross,x_mask=None,cross_mask=None):
		xp = xp + self.dropout(self.self_attention1(xp,xp,xp,attn_mask=x_mask)[0])
		xp = self.norm1(xp)
		xp1 = self.dropout(self.cross_attention1(xp,x_cross,x_cross,attn_mask=cross_mask)[0])
		xp2 = self.dropout(self.activation(self.conv3(xpm.transpose(-1,1))))
		xp2 = self.dropout(self.conv4(xp2).transpose(-1,1))	
		xpm = xpm + self.dropout(self.mark_attention(xpm,xpm,xpm,attn_mask=x_mask)[0])
		xpm = self.norm1(xpm)
		xp3 = self.dropout(self.fac_attention(xpm,xm_cross,x_cross,attn_mask=cross_mask)[0])
		yxp = self.norm2(xp+xp1+xp2+xp3)
		yxp = self.dropout(self.activation(self.conv1(yxp.transpose(-1,1))))
		yxp = self.dropout(self.conv2(yxp).transpose(-1,1))
		xpm = xpm + self.dropout(self.mark_cross_attention(xpm,xm_cross,xm_cross,attn_mask=cross_mask)[0])
		yxpm = xpm = self.norm2(xpm)
		yxpm = self.dropout(self.activation(self.conv1(yxpm.transpose(-1,1))))
		yxpm = self.dropout(self.conv2(yxpm).transpose(-1,1))
		xxpm = xxpm + self.dropout(self.self_attention2(xxpm,xxpm,xxpm,attn_mask=x_mask)[0])
		xxpm = self.norm1(xxpm)
		xxpm = xxpm + self.dropout(self.cross_attention2(xxpm,xxm_cross,xxm_cross,attn_mask=cross_mask)[0])
		yxxpm = xxpm = self.norm2(xxpm)
		yxxpm = self.dropout(self.activation(self.conv1(yxxpm.transpose(-1,1))))
		yxxpm = self.dropout(self.conv2(yxxpm).transpose(-1,1))
		return self.norm3(xp+yxp),self.norm3(xpm+yxpm),self.norm3(xxpm+yxxpm)
    
class Decoder(nn.Module):
	def __init__(self,layers,norm_layer=None,projection1=None,projection2=None):
		super(Decoder,self).__init__()
		self.layers = nn.ModuleList(layers)
		self.norm = norm_layer
		self.projection1 = projection1
		self.projection2 = projection2
	def forward(self,xp,xpm,xxpm,x,xm,xxm,x_mask=None,cross_mask=None):
		for layer in self.layers:
			xp,xpm,xxpm = layer(xp,xpm,xxpm,x,xm,xxm,x_mask=x_mask,cross_mask=cross_mask)
		if self.norm is not None:
			xp = self.norm((xp+xxpm))
		if self.projection1 is not None:
			xp = self.projection1(xp)
			xp = self.projection2(xp)
		return xp






