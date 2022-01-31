'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math, random
import logging
# from utils.config_utils import print_

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

import pickle
from pathlib import Path


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(output, target, top_k=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res



def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6


def count_parameters_in_MB_block(block:list):
    # block [blocktype, in_channels, out_channels, stride, kernel_size]
    # print(block)
    if len(block) == 5:
        [blocktype, in_channels, out_channels, stride, kernel_size] = block.values()
    elif len(block) == 7:
        [blocktype, in_channels, out_channels, stride, kernel_size, sub_layer, btr] = block.values()

    if blocktype == 'ConvKXBNRELU':
        block_size = in_channels * out_channels * kernel_size**2 + 2 * out_channels

    elif blocktype == 'ConvRes':
        block_size = in_channels * out_channels * kernel_size**2 + 2 * out_channels \
                    + out_channels * out_channels * kernel_size**2 + 2 * out_channels

        if stride != 1 or in_channels != out_channels: block_size += in_channels * out_channels * 1**2 + 2 * out_channels

    elif blocktype == 'SuperResConvKXKX':
        block_size = in_channels * out_channels * kernel_size**2 + 2 * out_channels \
                    + out_channels * out_channels * kernel_size**2 + 2 * out_channels

        if in_channels != out_channels or True: block_size += in_channels * out_channels * 1**2 + 2 * out_channels

        for _ in range(sub_layer-1):
            block_size += out_channels * out_channels * kernel_size**2 + 2 * out_channels \
                        + out_channels * out_channels * kernel_size**2 + 2 * out_channels

    elif blocktype == 'SuperResConvK1KXK1':
        bottleneck_channels = btr * out_channels
        block_size = in_channels * bottleneck_channels * 1**2 + 2 * bottleneck_channels \
                    + bottleneck_channels * bottleneck_channels * kernel_size**2 + 2 * bottleneck_channels \
                    + bottleneck_channels * out_channels * 1**2 + 2 * out_channels

        if in_channels != out_channels or True: block_size += in_channels * out_channels * 1**2 + 2 * out_channels

        bottleneck_channels = btr * out_channels
        for _ in range(sub_layer-1):
            block_size += out_channels * bottleneck_channels * 1**2 + 2 * bottleneck_channels \
                        + bottleneck_channels * bottleneck_channels * kernel_size**2 + 2 * bottleneck_channels \
                        + bottleneck_channels * out_channels * 1**2 + 2 * out_channels

    elif blocktype == 'SuperResConvK1DWK1':
        # g = btr
        bottleneck_channels = btr * in_channels
        block_size = in_channels * bottleneck_channels * 1**2 + 2 * bottleneck_channels \
                    + bottleneck_channels * bottleneck_channels * kernel_size**2 / (bottleneck_channels) + 2 * bottleneck_channels \
                    + bottleneck_channels * out_channels * 1**2 + 2 * out_channels

        if stride != 1 or in_channels != out_channels: block_size += in_channels * out_channels * 1**2 + 2 * out_channels

        bottleneck_channels = btr * out_channels
        for _ in range(sub_layer-1):
            block_size += out_channels * bottleneck_channels * 1**2 + 2 * bottleneck_channels \
                        + bottleneck_channels * bottleneck_channels * kernel_size**2 / (bottleneck_channels) + 2 * bottleneck_channels \
                        + bottleneck_channels * out_channels * 1**2 + 2 * out_channels

    return block_size


def count_parameters_in_MB_model(model:list, num_class=10):
    model_size = 0
    for block in model:
        model_size += count_parameters_in_MB_block(block)

    model_size += model[-1]['out'] * num_class + 1
    return model_size / 1e6


def count_flops_in_MB_block(block:list, resolution):
    # block [blocktype, in_channels, out_channels, stride, kernel_size]
    if len(block) == 5:
        [blocktype, in_channels, out_channels, stride, kernel_size] = block.values()
    elif len(block) == 7:
        [blocktype, in_channels, out_channels, stride, kernel_size, sub_layer, btr] = block.values()

    if blocktype == 'ConvKXBNRELU':
        flops = (resolution / stride) ** 2 * in_channels * out_channels * kernel_size ** 2 + \
                 (resolution / stride) ** 2 * out_channels * 2
        # flops += (resolution / stride) ** 2 * out_channels * 1 # relu?

    elif blocktype == 'ConvRes':
        flops = (resolution / stride) ** 2 * in_channels * out_channels * kernel_size ** 2 + \
                 (resolution / stride) ** 2 * out_channels * 2 + \
                 (resolution / stride) ** 2 * out_channels * out_channels * kernel_size ** 2 + \
                 (resolution / stride) ** 2 * out_channels * 2
        if stride != 1 or in_channels != out_channels:
            flops += (resolution / stride) ** 2 * in_channels * out_channels * 1 ** 2 + \
                     (resolution / stride) ** 2 * out_channels * 2

        flops += (resolution / stride) ** 2 * out_channels * 1 # +

    elif blocktype == 'SuperResConvKXKX':
        # assert btr == 1
        flops =  (resolution / stride) ** 2 * in_channels * out_channels * kernel_size ** 2 + \
                 (resolution / stride) ** 2 * out_channels * 2 + \
                 (resolution / stride) ** 2 * out_channels * out_channels * kernel_size ** 2 + \
                 (resolution / stride) ** 2 * out_channels * 2

        if stride == 2: flops += (resolution) ** 2 * in_channels
        if in_channels != out_channels or True:
            flops += (resolution / stride) ** 2 * in_channels * out_channels * 1 ** 2 + \
                     (resolution / stride) ** 2 * out_channels * 2
            flops += (resolution / stride) ** 2 * out_channels

        # flops += (resolution / stride) ** 2 * out_channels * 2 # +
        flops += ((resolution / stride) ** 2 * out_channels + (resolution / stride) ** 2 * out_channels) # relu

        for _ in range(sub_layer-1):
            flops += (resolution / stride) ** 2 * out_channels * out_channels * kernel_size ** 2 + \
                    (resolution / stride) ** 2 * out_channels * 2 + \
                    (resolution / stride) ** 2 * out_channels * out_channels * kernel_size ** 2 + \
                    (resolution / stride) ** 2 * out_channels * 2

            # flops += (resolution / stride) ** 2 * out_channels * 2 # +
            flops += ((resolution / stride) ** 2 * out_channels + (resolution / stride) ** 2 * out_channels) # relu

    elif blocktype == 'SuperResConvK1KXK1':
        bottleneck_channels = btr * out_channels
        flops = (resolution) ** 2 * in_channels * bottleneck_channels * 1 ** 2 + \
                (resolution) ** 2 * bottleneck_channels * 2 + \
                (resolution / stride) ** 2 * bottleneck_channels * bottleneck_channels * kernel_size ** 2 + \
                (resolution / stride) ** 2 * bottleneck_channels * 2 + \
                (resolution / stride) ** 2 * bottleneck_channels * out_channels * 1 ** 2 + \
                (resolution / stride) ** 2 * out_channels * 2

        if stride == 2: flops += resolution ** 2 * in_channels
        if in_channels != out_channels or True:
            flops += (resolution / stride) ** 2 * in_channels * out_channels * 1 ** 2 + \
                     (resolution / stride) ** 2 * out_channels * 2
            flops += (resolution / stride) ** 2 * out_channels

        # flops += (resolution / stride) ** 2 * out_channels * 2 # relu?
        flops += resolution ** 2 * bottleneck_channels + (resolution / stride) ** 2 * bottleneck_channels + (resolution / stride) ** 2 * out_channels # relu?

        bottleneck_channels = btr * out_channels
        for _ in range(sub_layer-1):
            flops += (resolution / stride) ** 2 * out_channels * bottleneck_channels * 1 ** 2 + \
                    (resolution / stride) ** 2 * bottleneck_channels * 2 + \
                    (resolution / stride) ** 2 * bottleneck_channels * bottleneck_channels * kernel_size ** 2 + \
                    (resolution / stride) ** 2 * bottleneck_channels * 2 + \
                    (resolution / stride) ** 2 * bottleneck_channels * out_channels * 1 ** 2 + \
                    (resolution / stride) ** 2 * out_channels * 2

            # flops += (resolution / stride) ** 2 * out_channels * 2 # relu?
            flops += (resolution / stride) ** 2 * bottleneck_channels + (resolution / stride) ** 2 * bottleneck_channels + (resolution / stride) ** 2 * out_channels # relu?

    elif blocktype == 'SuperResConvK1DWK1':
        # g = btr
        bottleneck_channels = btr * in_channels
        flops = resolution ** 2 * in_channels * bottleneck_channels * 1 ** 2 + \
                resolution ** 2 * bottleneck_channels * 2 + \
                (resolution / stride) ** 2 * bottleneck_channels * bottleneck_channels * kernel_size ** 2 / (bottleneck_channels) + \
                (resolution / stride) ** 2 * bottleneck_channels * 2 + \
                (resolution / stride) ** 2 * bottleneck_channels * out_channels * 1 ** 2 + \
                (resolution / stride) ** 2 * out_channels * 2

        if stride == 2: flops += resolution ** 2 * in_channels
        if in_channels != out_channels or True:
            flops += (resolution / stride) ** 2 * in_channels * out_channels * 1 ** 2 + \
                     (resolution / stride) ** 2 * out_channels * 2
            flops += (resolution / stride) ** 2 * out_channels

        # flops += (resolution / stride) ** 2 * out_channels * 2 # relu?
        flops += resolution ** 2 * bottleneck_channels + (resolution / stride) ** 2 * bottleneck_channels + (resolution / stride) ** 2 * out_channels # relu?

        bottleneck_channels = btr * out_channels
        for _ in range(sub_layer-1):
            flops += (resolution / stride) ** 2 * out_channels * bottleneck_channels * 1 ** 2 + \
                    (resolution / stride) ** 2 * bottleneck_channels * 2 + \
                    (resolution / stride) ** 2 * bottleneck_channels * bottleneck_channels * kernel_size ** 2 / (bottleneck_channels) + \
                    (resolution / stride) ** 2 * bottleneck_channels * 2 + \
                    (resolution / stride) ** 2 * bottleneck_channels * out_channels * 1 ** 2 + \
                    (resolution / stride) ** 2 * out_channels * 2

            # flops += (resolution / stride) ** 2 * out_channels * 2 # relu?
            flops += (resolution / stride) ** 2 * bottleneck_channels + (resolution / stride) ** 2 * bottleneck_channels + (resolution / stride) ** 2 * out_channels # relu?

    return flops


def count_flops_in_MB_model(model:list, resolution, num_class=10):
    flops = 0.0
    for block in model:
        flops += count_flops_in_MB_block(block, resolution)
        resolution /= block['s']

    # flops += resolution ** 2 * model[-1]['s'] # avg pool
    flops += model[-1]['out'] * num_class # + 1

    return flops / 1e6


__block_layers__ = {
    'ConvKXBN': 1,
    'ConvKXBNRELU': 1,
    'SuperResConvKXKX': 2,
    'SuperResConvK1KXK1': 3,
    'SuperResConvK1DWK1': 3,
    # 'SuperResConvK1DWXK1': SuperResConvK1DWXK1,
}
def count_layers_in_model(model:list):
    layers = 0 # 0
    # print(type(model))
    for idx, block in enumerate(model):
        # print(block)
        block_type = block['class']
        L = block.get('L')
        L = L if L else 1
        layers += L * __block_layers__[block_type]

    return layers

# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

# TOTAL_BAR_LENGTH = 65.
# last_time = time.time()
# begin_time = last_time
# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.

#     cur_len = int(TOTAL_BAR_LENGTH*current/total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

#     sys.stdout.write(' [')
#     for i in range(cur_len):
#         sys.stdout.write('=')
#     sys.stdout.write('>')
#     for i in range(rest_len):
#         sys.stdout.write('.')
#     sys.stdout.write(']')

#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time

#     L = []
#     L.append('  Step: %s' % format_time(step_time))
#     L.append(' | Tot: %s' % format_time(tot_time))
#     if msg:
#         L.append(' | ' + msg)

#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#         sys.stdout.write(' ')

#     # Go back to the center of the bar.
#     for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#         sys.stdout.write('\b')
#     sys.stdout.write(' %d/%d ' % (current+1, total))

#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def set_logging(save_path, console=True, log_name='log.log', name=None):
    log_format = '%(asctime)s %(message)s'
    date_format = '%m/%d %I:%M:%S %p' #  '%m/%d %H:%M:%S' #
    # if 'log' in save_path or 'txt' in save_path: log_name = ''
    if not os.path.isdir(save_path): log_name = ''

    logger = logging.getLogger(name)
    
    logger.propagate = False
    logging.basicConfig(level=logging.INFO,
        format=log_format, datefmt=date_format, filename=os.path.join(save_path, log_name), filemode='w')
    # logging.basicConfig()

    fh = logging.FileHandler(os.path.join(save_path, log_name), mode='w')
    fh.setFormatter(logging.Formatter(log_format, date_format))
    fh.setLevel(logging.INFO)
    # logging.getLogger().addHandler(fh)
    logger.addHandler(fh)
    # fh.close()

    # 创建一个StreamHandler,用于输出到控制台
    level = logging.INFO if console else logging.WARN
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(ch)
    # logger.addHandler(ch)

    return logger


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.max = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        # self.max = 


def pickle_save(obj, path):
    file_path = Path(path)
    file_dir = file_path.parent
    file_dir.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as f:
        pickle.dump(obj, f)


def pickle_load(path):
    if not Path(path).exists():
        raise ValueError("{:} does not exists".format(path))
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    # net = [['ConvKXBN', 3, 32, 1, 3], ['SuperResConvKXKX', 32, 128, 1, 5, 1], ['SuperResConvKXKX', 128, 256, 2, 3, 2], ['SuperResConvKXKX', 256, 256, 2, 5, 2], ['SuperResConvKXKX', 256, 256, 2, 5, 4]]
    net = [
        {'class': 'ConvKXBNRELU', 'in': 3, 'out': 64, 's': 1, 'k': 3},
        {'class': 'SuperResConvKXKX', 'in': 64, 'out': 64, 's': 1, 'k': 3, 'L': 2, 'btr': 1},
        {'class': 'SuperResConvKXKX', 'in': 64, 'out': 128, 's': 2, 'k': 3, 'L': 2, 'btr': 1},
        {'class': 'SuperResConvKXKX', 'in': 128, 'out': 256, 's': 2, 'k': 3, 'L': 2, 'btr': 1},
        {'class': 'SuperResConvKXKX', 'in': 256, 'out': 512, 's': 2, 'k': 3, 'L': 2, 'btr': 1},
    ]
    param = count_parameters_in_MB_model(net)
    flops = count_flops_in_MB_model(net, 32)
    print(flops)

    set_logging('./', console=False)
    # logging.error('error')
    # logging.warning('warning')
    logging.info('info')
    logging.debug('debug')

    # logger.info('info')

    pass