import torch

from basicsr.archs import build_network


def test_encoderdecoder():
    opt = {'type': 'EncoderDecoder', 'depth': 2, 'nf': 64, 'in_nc': 3, 'out_nc': 3}
    net = build_network(opt)

    device = torch.device('cuda:0')
    inp = torch.randn(16, 3, 128, 128).to(device)
    net = net.to(device)
    out = net(inp)
    print(out.shape)


if __name__ == '__main__':
    test_encoderdecoder()
