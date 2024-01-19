import torch.nn as nn
from functools import partial
from experiments.util.arch_blocks import *


class network(ResNet):
    def __init__(self,
                 n_input,
                 n_output,
                 args):

        features = [[[4]],
                [[4] * 2] * 3,
                [[8] * 2] * 4,
                [[16] * 2] * 6,
                [[32] * 2] * 3]

        common_params = {
            'pooling_type': 'hardmax',
            'so3_size': 4,
            'downsample_by_pooling': args.downsample_by_pooling,
            'conv_dropout_p': args.p_drop_conv,
        }
        global OuterBlock
        OuterBlock = partial(OuterBlock,
                             res_block=partial(ILPOResBlock, **common_params))
        super().__init__(
            OuterBlock(n_input,             features[0], size=7),
            OuterBlock(features[0][-1][-1], features[1], size=args.kernel_size, stride=1),
            OuterBlock(features[1][-1][-1], features[2], size=args.kernel_size, stride=2),
            OuterBlock(features[2][-1][-1], features[3], size=args.kernel_size, stride=2),
            OuterBlock(features[3][-1][-1], features[4], size=args.kernel_size, stride=2),
            AvgSpacial(),
            nn.Dropout(p=args.p_drop_fully, inplace=True) if args.p_drop_fully is not None else None,
            nn.Linear(features[4][-1][-1], n_output))
            # nn.Linear(features[3][-1][-1], n_output))
            # nn.Linear(features[2][-1][-1], n_output))