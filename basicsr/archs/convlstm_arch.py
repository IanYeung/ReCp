import functools
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTR
from .arch_util import ResidualBlockNoBN, make_layer


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, tensor_size):
        height, width = tensor_size
        return torch.zeros((batch_size, self.hidden_dim, height, width), requires_grad=True).cuda(),\
               torch.zeros((batch_size, self.hidden_dim, height, width), requires_grad=True).cuda()


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = (input_tensor.size(3), input_tensor.size(4))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), tensor_size=tensor_size)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, tensor_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, tensor_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvBLSTM(nn.Module):
    # Constructor
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvBLSTM, self).__init__()
        self.forward_net = ConvLSTM(input_size, input_dim, hidden_dim // 2, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias,
                                    return_all_layers=return_all_layers)
        self.reverse_net = ConvLSTM(input_size, input_dim, hidden_dim // 2, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias,
                                    return_all_layers=return_all_layers)
        self.return_all_layers = return_all_layers

    def forward(self, xforward, xreverse):
        """
        xforward, xreverse = B T C H W tensors.
        """

        y_out_fwd, _ = self.forward_net(xforward)
        y_out_rev, _ = self.reverse_net(xreverse)

        if not self.return_all_layers:
            y_out_fwd = y_out_fwd[-1]  # outputs of last CLSTM layer = B, T, C, H, W
            y_out_rev = y_out_rev[-1]  # outputs of last CLSTM layer = B, T, C, H, W

        reversed_idx = list(reversed(range(y_out_rev.shape[1])))
        y_out_rev = y_out_rev[:, reversed_idx, ...]  # reverse temporal outputs.
        ycat = torch.cat((y_out_fwd, y_out_rev), dim=2)

        return ycat


@ARCH_REGISTRY.register()
class ConvBLSTMCompressor(nn.Module):
    def __init__(self, num_feat=64, num_chan=3, kernel_size=3, num_layers=1, nb_encoder=5, nb_decoder=5):
        super(ConvBLSTMCompressor, self).__init__()
        self.num_feat = num_feat

        ResidualBlockNoBN_f = functools.partial(ResidualBlockNoBN, num_feat=num_feat)
        self.conv_encoder = nn.Conv2d(num_chan, num_feat, (3, 3), (1, 1), (1, 1), bias=True)
        self.encoder = make_layer(ResidualBlockNoBN_f, nb_encoder)

        self.ConvBLSTM = ConvBLSTM(input_size=(32, 32), input_dim=num_feat, hidden_dim=num_feat,
                                   kernel_size=(kernel_size, kernel_size),
                                   num_layers=num_layers, batch_first=True,
                                   bias=True, return_all_layers=False)

        self.decoder = make_layer(ResidualBlockNoBN_f, nb_decoder)
        self.conv_decoder = nn.Conv2d(num_feat, num_chan, (3, 3), (1, 1), (1, 1), bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N input video frames

        out = x.view(B * N, C, H, W)
        out = self.lrelu(self.conv_encoder(out))
        out = self.encoder(out)
        out = out.view(B, N, self.nf, H, W)
        x_forward = out
        x_reverse = x_forward[:, list(reversed(range(x_forward.shape[1]))), ...]
        out = self.ConvBLSTM(x_forward, x_reverse)
        out = out.view(B*N, self.nf, H, W)
        out = self.lrelu(self.decoder(out))
        out = self.conv_decoder(out)
        out = out.view(B, N, C, H, W)
        out += x

        return out


if __name__ == '__main__':
    B, N, C, H, W = 1, 7, 3, 64, 64
    device = torch.device('cuda:0')
    x = torch.randn(B, N, C, H, W).to(device)
    model = ConvBLSTMCompressor(nf=64).to(device)
    out = model(x)
    print(out.shape)
    params = sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])
    print('EVSR Parameter count: {}'.format(params))
