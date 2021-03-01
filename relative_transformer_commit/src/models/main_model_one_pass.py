
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.backbone import CnnBacknone
from src.modules.transformer_rel_encoder import TransformerEncoder
from src.modules.transformer_decoder import TransformerDecoder


class MainModel(nn.Module):
    def __init__(self, opts):
        super(MainModel, self).__init__()
        self.opts = opts
        self.hidden_size = opts.hidden_size
        self.cnnbackbone = CnnBacknone()
        self.encoder = TransformerEncoder(opts)
        self.decoder = TransformerDecoder(opts)
        self.pred_head = nn.Linear(self.hidden_size, 1)

    def forward(self, x, y):
        """

        :param x: [bs, 12, 4, 24, 72]
        :param y: [bs, 24, 4, 24, 72]
        :return:
        """
        x = self.cnnbackbone(x)

        bs, T, h = x.shape
        x_len = (torch.ones((bs)) * T).long().to(x.device)
        x_mask = self._get_mask(x_len)
        enc_out = self.encoder(x, x_len, x_mask)

        # TODO, dec_inp?
        y = self.cnnbackbone(y)  # [bs, y_len, 512]
        y_inp = torch.zeros((bs, y.size(1), h)).to(y.device) # [bs, y_len, 512]
        y_len = (torch.ones((bs)) * y.size(1)).long().to(y.device)
        y_mask = self._get_mask(y_len)
        dec_out = self.decoder(
            trg_embed=y_inp,
            encoder_output=enc_out,
            src_mask=x_mask,
            trg_mask=y_mask)

        prediction = self.pred_head(dec_out).squeeze(-1)
        # print("enc_out: ", enc_out.shape)
        # print("dec_out: ", dec_out.shape)
        # exit()
        return dec_out, prediction, y

    def decoder_one_pass(self, x):
        x = self.cnnbackbone(x)

        bs, T, h = x.shape
        x_len = (torch.ones((bs)) * T).long().to(x.device)
        x_mask = self._get_mask(x_len)
        enc_out = self.encoder(x, x_len, x_mask)

        dec_inp = torch.zeros((bs, 24, h)).to(x.device)
        y_len = (torch.ones((bs)) * dec_inp.size(1)).long().to(x.device)
        y_mask = self._get_mask(y_len)
        dec_out = self.decoder(
            trg_embed=dec_inp,
            encoder_output=enc_out,
            src_mask=x_mask,
            trg_mask=y_mask)
        # print("dec_out: ", dec_out.shape)
        preds = self.pred_head(dec_out)
        # print("preds: ", preds.shape)
        return preds.squeeze(-1)


    def _get_mask(self, x_len):
        pos = torch.arange(0, max(x_len)).unsqueeze(0).repeat(x_len.size(0), 1).to(x_len.device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask.unsqueeze(1)


if __name__ == "__main__":
    class Config():
        hidden_size = 512
        ff_size = 2048
        num_heads = 8
        dropout = 0.1
        emb_dropout = 0.1
        num_layers = 6
        local_num_layers = 0
        use_relative = True
        max_relative_positions = 24
        fp16 = False
        embedding_dim = 512
    # opts = Config()
    # x = torch.randn(16, 12, 4, 24, 72)
    # y = torch.randn(16, 24, 4, 24, 72)
    # model = MainModel(opts)
    # dec_out, prediction = model(x, y)
    # print(dec_out.shape, prediction.shape)