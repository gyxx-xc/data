from typing import Optional, Tuple
import torch
import torch.nn as nn
from labml_helpers.module import Module


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, layer_norm: bool = False):
        #input_size是输入张量的特征维度, hidden_size是隐藏状态的维度，layer_norm是一个布尔参数, 如果为True,则在计算门和单元状态之前应用层归一化。
        super().__init__()
        self.input_linear = nn.linear(input_size, 4 * hidden_size)
        self.hidden_linear = nn.linear(hidden_size, 4 * hidden_size, bias=False)

        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
            #创建4个层归一化层, 用于对4个门进行归一化
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
            #创建4个恒等变换层,相当于没有层归一化
            self.layer_norm_c = nn.Identity()
    # Input: x_t, h_{t-1}, c_{t-1}
    #Output: h_t, c_t
    # Shape of x: [batch_size, input_size]
    # Shape of h: [batch_size, hidden_size]
    # Shape of c: [batch_size, hidden_size]
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        figo = self.input_linear(x) + self.hidden_linear(h)

        figo = figo.chunk(4, dim=-1)
        #将计算得到的4倍隐藏状态维度的张量分割成4个张量,分别对应4个门。

        figo = [self.layer_norm[i](figo[i]) for i in range(4)]

        f, i, g, o = figo

        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.cells = nn.ModuleList(
            [LSTMCell(input_size, hidden_size)] + [LSTMCell(hidden_size, hidden_size) for _ in range(n_layers - 1)])

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        n_steps, batch_size = x.shape[:2]

        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]

        else:
            (h, c) = state
            h, c = list(torch.unbind(h)), list(torch.unhind(c))
            #传入的 2 个张量(shape 为 (self.n_layers, batch_size, hidden_size)) 拆分为 self.n_layers 个张量(shape 为 (batch_size, hidden_size))

        out = []

        for t in range(n_steps):
            inp = x[t]

            for layer in range(self.n_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]
            out.append(h[-1])

        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)

        return out, (h, c)

    nn.LSTM()
