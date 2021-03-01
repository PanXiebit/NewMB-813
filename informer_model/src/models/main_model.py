
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.informer import Informer
from src.modules.backbone import CnnBacknone


class MainInformer(nn.Module):
    def __init__(self, c_out, seq_len, label_len, out_len, mid_dim):
        super(MainInformer, self).__init__()

        self.cnnbackbone = CnnBacknone()

        self.transformer = Informer(c_out, seq_len, label_len, out_len, mid_dim,
                                    factor=5, d_model=64, n_heads=4, e_layers=2,
                                    d_layers=2, d_ff=64, dropout=0.2, attn='full',
                                    embed='fixed', freq='h', activation='gelu',
                                    output_attention=False, distil=False)

    def forward(self, x):
        x = self.cnnbackbone(x)
        x = self.transformer(x)
        return x.squeeze(-1)


if __name__ == "__main__":
    x = torch.randn(16, 12, 4, 24, 72)
    model = MainInformer(c_out=24*72, seq_len=12, label_len=1, out_len=24, mid_dim=128)
    out = model(x)
    print(out.shape)