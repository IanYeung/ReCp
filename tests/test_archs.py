import torch

from basicsr.archs import build_network


def get_encoderdecoder():
    opt = {'type': 'EncoderDecoder', 'depth': 2, 'nf': 64, 'num_in_ch': 3, 'num_out_ch': 3, 'color_space': 'rgb'}
    net = build_network(opt)
    return net


def get_bic():
    opt = {'type': 'BIC', 'nf': 64, 'num_in_ch': 3, 'num_out_ch': 3, 'color_space': 'rgb'}
    net = build_network(opt)
    return net


def get_basicvsr():
    opt = {'type': 'BasicVSR', 'num_feat': 64, 'num_block': 30, 'scale': 4, 'spynet_path': None}
    net = build_network(opt)
    return net


if __name__ == '__main__':

    device = torch.device('cuda:0')

    inp = torch.randn(1, 3, 128, 224).to(device)
    # inp = torch.randn(1, 7, 3, 128, 224).to(device)
    net = get_encoderdecoder().to(device)
    out = net(inp)
    print(out.shape)
