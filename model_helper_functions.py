import os
import numpy as np
import torch
import math
# Save and Load Functions
from definitions import FEATURE_SELECTION


def save_checkpoint(save_path, model, optimizer, val_loss):

    if save_path == None:
        return

    save_path = os.path.join(save_path, 'checkpoint.pt')

    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }

    torch.save(state_dict, save_path)


def load_checkpoint(load_path, model, optimizer, device):

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['val_loss']


def save_metrics(save_path, train_losses, val_losses):

    if save_path == None:
        return

    save_path = os.path.join(save_path, 'metrics.pt')

    state_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses,
    }

    torch.save(state_dict, save_path)


def load_metrics(load_path, device):

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_losses'], state_dict['val_losses']


def save_params(save_path, params):

    if save_path == None:
        raise Exception('No save path.')

    torch.save(params, save_path)
    print(f'Parameters sve to ==> {save_path}')


# def create_sent_level_features(feat_type, sent, is_one_hot, from_selection=True):
#     # feat type with underscore to match token attributes
#     ftwu = f'{feat_type}_'
#     feature = [0]*len(FEATURE_SELECTION)

#     if is_one_hot:

#     for t in sent:
#         t_attr = getattr(t, ftwu)

#         if t_attr in FEATURE_SELECTION:
#             if is_one_hot:
#                 feature[FEATURE_SELECTION[t_attr]] = 1
#             else:
#                 feature[FEATURE_SELECTION[t_attr]] += 1

#     # encode counts to binary and flatten
#     if not is_one_hot:
#         feature = [int(b) for digit in feature for b in f'{digit:06b}']

#     return feature


def scale(tensor, extremes_t: tuple = None, min_r=0, max_r=1):
    tensor = tensor if torch.is_tensor(tensor) else torch.tensor(tensor)

    if extremes_t is None:
        min_t = torch.min(tensor)
        max_t = torch.max(tensor)
    else:
        min_t, max_t = extremes_t

    range_t = max_t - min_t

    if range_t > 0:
        std = (tensor - min_t) / range_t
        return std * (max_r - min_r) + min_r

    return torch.zeros(tensor.size())


def range_inc(start, stop, step, inc, inc_operator='*'):
    i = start
    while i < stop:
        yield i
        i += step
        if inc_operator == '*':
            step *= inc
        else:
            step += inc


def round_to_first_non_zero(nums, add_to_dist=0):
    for i in range(len(nums)):
        x = nums[i]
        if x == 0.0:
            continue
        dist = abs(round(math.log10(abs(x)))) + add_to_dist
        mp = 10**dist
        nums[i] = int(x * mp) / mp

    return nums
# min_v = torch.min(vector)
# range_v = torch.max(vector) - min_v
# if range_v > 0:
#     normalised = (vector - min) / range_v
# else:
#     normalised = torch.zeros(vector.size())
# tens = torch.randn((3, 5))
# nump = np.array(tens)
# print(tens)
# print(nump)

# # print(torch.min(nump))
# print(np.min(tens))

# x = torch.randn((2, 3))
# print(x)
# x = torch.tensor(x)
# print(x)
