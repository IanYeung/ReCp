import torch

from basicsr.archs import build_network


def get_encoderdecoder():
    opt = {'type': 'EncoderDecoder', 'depth': 2, 'nf': 64, 'num_in_ch': 3, 'num_out_ch': 3}
    net = build_network(opt)
    return net


def get_bic():
    opt = {'type': 'BIC', 'nf': 64, 'num_in_ch': 3, 'num_out_ch': 3}
    net = build_network(opt)
    return net


if __name__ == '__main__':

    device = torch.device('cuda:2')
    inp = torch.randn(16, 3, 128, 128).to(device)
    net = get_bic().to(device)
    out = net(inp)
    print(out.shape)
