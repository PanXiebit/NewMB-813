import torch
import torch.nn as nn

from src.modules.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from src.modules.decoder import Decoder, DecoderLayer
from src.modules.attn import FullAttention, ProbAttention, AttentionLayer
from src.modules.embed import DataEmbedding
from src.modules.embed import PositionalEmbedding


class Informer(nn.Module):
    def __init__(self, c_out, seq_len, label_len, out_len, mid_dim,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.d_model = d_model
        self.attn = attn
        self.output_attention = output_attention

        # Encoding TODO, remove encoding
        # self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.pos_emb = PositionalEmbedding(d_model, max_len=40)

        # Decoder
        self.x_dec = nn.Embedding(out_len, d_model)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # TODO. casual mask 交替存在，既有非自回归，也有自回归
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection_1 = nn.Linear(d_model, 1, bias=True)
        # self.projection_2 = nn.Linear(mid_dim, 24 * 72, bias=True)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = x_enc + self.pos_emb(x_enc)
        #print("enc_out: ", enc_out.shape)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        #print("enc_out: ", enc_out.shape)

        # TODO
        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.x_dec.weight.unsqueeze(0).repeat(enc_out.size(0), 1, 1)
        dec_out = torch.zeros((enc_out.size(0), self.pred_len, self.d_model)).to(enc_out.device)
        dec_out += self.pos_emb(dec_out)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection_1(dec_out)
        # dec_out = self.projection_2(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        stacks = list(range(e_layers, 2, -1))  # you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == "__main__":
    c_out = 24 * 72
    seq_len = 12
    label_len = 1
    out_len = 24
    model = Informer(c_out, seq_len, label_len, out_len, attn='full', distil=False)
    x = torch.randn(16, 12, 512)
    x_mark_enc = None
    x_dec = torch.randn(16, 24, 512)
    x_mark_dec = None
    out = model(x, x_mark_enc, x_dec, x_mark_dec)
    #print(out.shape)
