
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
        self.cnn_dropout = nn.Dropout(opts.cnn_dropout)
        self.encoder = TransformerEncoder(opts)
        self.decoder = TransformerDecoder(opts)
        self.conv1 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, stride=1,
                              padding=0)
        self.norm1 = nn.BatchNorm1d(self.hidden_size)
        self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, stride=1,
                              padding=0)
        self.norm2 = nn.BatchNorm1d(self.hidden_size)
        self.pred_head = nn.Linear(self.hidden_size, 1)

    def init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        """

        :param x: [bs, 12, 4, 24, 72]
        :param y: [bs, 24, 4, 24, 72]
        :return:
        """
        enc_x = self.cnnbackbone(x)
        enc_x = self.cnn_dropout(enc_x)

        bs, T, h = enc_x.shape
        x_len = (torch.ones((bs)) * T).long().to(x.device)
        x_mask = self._get_mask(x_len)
        enc_out = self.encoder(enc_x, x_len, x_mask)

        # TODO, dec_inp?
        y_inp = torch.zeros((bs, y.size(1), h)).to(y.device) # [bs, y_len, 512]
        # TODO.
        y_inp = torch.cat([enc_x, y_inp], dim=1)  # [bs, x_len + y_len, 512]

        y_len = (torch.ones((bs)) * y_inp.size(1)).long().to(y.device)
        y_mask = self._get_mask(y_len)
        dec_out = self.decoder(
            trg_embed=y_inp,
            encoder_output=enc_out,
            src_mask=x_mask,
            trg_mask=y_mask)

        # convolution
        dec_out = dec_out.transpose(1, 2)
        dec_out = self.conv1(dec_out)
        dec_out = self.norm1(dec_out)
        dec_out = self.conv2(dec_out)
        dec_out = self.norm2(dec_out)
        dec_out = dec_out.transpose(1, 2)

        dec_out = dec_out[:, -y.size(1):, :]

        prediction = self.pred_head(dec_out).squeeze(-1)
        y = self.cnnbackbone(y)  # [bs, y_len, 512]
        return dec_out, prediction, y

    def decoder_one_pass(self, x):
        enc_x = self.cnnbackbone(x)
        enc_x = self.cnn_dropout(enc_x)

        bs, T, h = enc_x.shape
        x_len = (torch.ones((bs)) * T).long().to(x.device)
        x_mask = self._get_mask(x_len)
        enc_out = self.encoder(enc_x, x_len, x_mask)

        dec_inp = torch.zeros((bs, 24, h)).to(x.device)
        # TODO
        dec_inp = torch.cat([enc_x, dec_inp], dim=1)
        y_len = (torch.ones((bs)) * dec_inp.size(1)).long().to(x.device)
        y_mask = self._get_mask(y_len)
        dec_out = self.decoder(
            trg_embed=dec_inp,
            encoder_output=enc_out,
            src_mask=x_mask,
            trg_mask=y_mask)
        # convolution
        dec_out = dec_out.transpose(1, 2)
        dec_out = self.conv1(dec_out)
        dec_out = self.norm1(dec_out)
        dec_out = self.conv2(dec_out)
        dec_out = self.norm2(dec_out)
        dec_out = dec_out.transpose(1, 2)

        dec_out = dec_out[:, -24:, :]
        # print("dec_out: ", dec_out.shape)
        preds = self.pred_head(dec_out)
        # print("preds: ", preds.shape)
        return preds.squeeze(-1)

    def decoder_autogressive(self, x):
        enc_x = self.cnnbackbone(x)
        enc_x = self.cnn_dropout(enc_x)

        bs, T, h = enc_x.shape
        x_len = (torch.ones((bs)) * T).long().to(x.device)
        x_mask = self._get_mask(x_len)
        enc_out = self.encoder(enc_x, x_len, x_mask)

        x_pad = torch.zeros((bs, 12, h)).to(x.device)
        x_prev = enc_x
        dec_outs = []
        # TODO
        for i in range(2):
            dec_inp = torch.cat([x_prev, x_pad], dim=1)      # [bs, 24, h]
            y_len = (torch.ones((bs)) * dec_inp.size(1)).long().to(x.device)
            y_mask = self._get_mask(y_len)
            dec_out = self.decoder(
                trg_embed=dec_inp,
                encoder_output=enc_out,
                src_mask=x_mask,
                trg_mask=y_mask)
            # # convolution
            # print("dec_out: ", i, dec_out.shape)
            # dec_out = dec_out.transpose(1, 2)
            # dec_out = self.conv1(dec_out)
            # dec_out = self.norm1(dec_out)
            # dec_out = self.conv2(dec_out)
            # dec_out = self.norm2(dec_out)
            # dec_out = dec_out.transpose(1, 2)  # [bs, 10, h]
            # print("dec_out: ", i, dec_out.shape)

            dec_outs.append(dec_out[:, -12:, :])
            x_prev = dec_out[:, -12:, :]

        dec_outs = torch.cat(dec_outs, dim=1)  # [bs, 24, h]
        # print("dec_out: ", dec_out.shape)
        preds = self.pred_head(dec_outs)
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