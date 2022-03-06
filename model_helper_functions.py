import torch
# Save and Load Functions
from definitions import FEATURE_SELECTION


def save_checkpoint(save_path, model, optimizer, val_loss):

    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': val_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer, device):

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, val_loss_list, global_steps_list):

    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': val_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, device):

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


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


def scale(tensor, min_r=-33.0, max_r=11.0):
    range_t = max_r - min_r
    if range_t > 0:
        return (tensor - min_r) / range_t

    return torch.zeros(tensor.size())

# min_v = torch.min(vector)
# range_v = torch.max(vector) - min_v
# if range_v > 0:
#     normalised = (vector - min) / range_v
# else:
#     normalised = torch.zeros(vector.size())
