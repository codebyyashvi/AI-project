import torch
import torch.nn as nn
from models.flow.denseflow.flows import Flow
from models.flow.denseflow.transforms import UniformDequantization, ScalarAffineBijection, Squeeze2d
from models.flow.denseflow.distributions import StandardNormal, ConvNormal2d
from .flow_modules import InvertibleDenseBlock, InvertibleTransition


def parameter_count(module):
    trainable, non_trainable = 0, 0
    for p in module.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            non_trainable += p.numel()
    return trainable, non_trainable


def dim_from_shape(x):
    return x[0] * x[1] * x[2]


class PoolFlow(Flow):

    def __init__(self, data_shape=(3, 64, 64), block_config=[3, 2, 1], layers_config=[3, 4, 8],
                 layer_mid_chnls=[32, 32, 32], growth_rate=6, num_bits=8, checkpointing=False):

        transforms = []
        current_shape = data_shape

        # Change range from [0,1]^D to [-0.5, 0.5]^D
        transforms.append(ScalarAffineBijection(shift=-0.5))

        # Initial squeeze
        transforms.append(Squeeze2d())
        current_shape = (current_shape[0] * 4,
                         current_shape[1] // 2,
                         current_shape[2] // 2)

        # scale = 1
        dim_initial = dim_from_shape(data_shape)
        dim_output = 0
        print(f"> Initial dim num: {dim_initial}")
        for i, num_layers in enumerate(block_config):
            idbt = InvertibleDenseBlock(current_shape[0], num_layers, layers_config[i], layer_mid_chnls[i],
                                        growth_rate=growth_rate, checkpointing=checkpointing)
            transforms.append(idbt)

            chnls = current_shape[0] + growth_rate * (num_layers - 1)
            current_shape = (chnls,
                             current_shape[1],
                             current_shape[2])

            # scale = scale * (2 ** (num_layers - 1))

            if i != len(block_config) - 1:
                # print('Using squeeze instead of invertible transition')
                # transforms.append(Squeeze2d())
                # d0 = dim_from_shape(current_shape)
                # current_shape = (current_shape[0] * 4,
                #                  current_shape[1] // 2,
                #                  current_shape[2] // 2)
                # d1 = dim_from_shape(current_shape)
                # dim_output += (d0 - d1)
                transforms.append(InvertibleTransition(current_shape[0]))
                # scale = scale / 2.

                d0 = dim_from_shape(current_shape)

                current_shape = (current_shape[0] * 2,
                                 current_shape[1] // 2,
                                 current_shape[2] // 2)
                d1 = dim_from_shape(current_shape)
                dim_output += (d0 - d1)

        dim_output += dim_from_shape(current_shape)
        # coef = dim_output/dim_initial
        coef = 1.
        transforms = [UniformDequantization(num_bits=num_bits, coef=coef), *transforms]
        # print('using VDQ')
        # transforms = [VariationalDequantization(encoder=DequantizationFlow(data_shape, num_bits=num_bits), num_bits=num_bits, coef=coef), *transforms]

        super(PoolFlow, self).__init__(base_dist=ConvNormal2d(current_shape),
                                       transforms=transforms, coef=coef)
        self.out_shape = current_shape

        print('> Parameters count:', parameter_count(self))
        print('> Output shape:', current_shape)
