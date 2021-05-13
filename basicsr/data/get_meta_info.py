import glob
from os import path as osp


if __name__ == '__main__':
    root = '/data2/yangxi/datasets/Vimeo90k/vimeo_septuplet'
    seq_path_list = sorted(glob.glob(osp.join(root, 'sequences', '*', '*')))
    out = [
        '{}/{} 7 (256,448,3)\n'.format(seq_path.split('/')[-2], seq_path.split('/')[-1])
        for seq_path in seq_path_list
    ]
    with open('meta_info/meta_info_Vimeo90K_all_GT.txt', 'w') as f:
        f.writelines(out)
