import torch
import numpy as np
def get_top_layers_from_key(state_dict, keyword):

    if not keyword:
        return state_dict
    top_state_dict = {}

    # 遍历 state_dict 的 key
    start_collecting = False
    for key in state_dict.keys():
        if keyword in key:
            start_collecting = True  # 当检测到key包含'4'时，开始收集
        if start_collecting:
            top_state_dict[key] = state_dict[key]  # 将当前及后续层加入top_state_dict
    if not start_collecting:
        return state_dict
    return top_state_dict
def calculate_state_dict_minus(state_dict1, state_dict2):
    assert state_dict1.keys() == state_dict2.keys(), "State dict keys do not match!"
    sum_state_dict = {}
    for key in state_dict1.keys():
        sum_state_dict[key] = state_dict1[key] - state_dict2[key]
    return sum_state_dict
def calculate_state_dict_inner_product(state_dict1, state_dict2):
    skip_keywords = ['num_batches_tracked', 'running_mean', 'running_var']
    assert state_dict1.keys() == state_dict2.keys(), "State dict keys do not match!"
    inner_product_sum = 0.0
    for key in state_dict1.keys():
        if not any(keyword in key for keyword in skip_keywords):
            inner_product = torch.sum(state_dict1[key] * state_dict2[key])
            inner_product_sum += inner_product.item()
    return inner_product_sum
def row_softmax(W):
    # 判断是否是一维数组（单行）
    if W.ndim == 1:
        exp_array = np.exp(W - np.max(W))  # 只对这一行做 softmax
        new_W = exp_array / np.sum(exp_array)
    else:
        exp_matrix = np.exp(W - np.max(W, axis=1, keepdims=True))
        # 计算每一行的 softmax
        new_W = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)

    return new_W
def weighted_average_state_dict(state_dicts, weights):
    assert len(state_dicts) == len(weights), "Number of state_dicts and weights must be the same!"
    keys = state_dicts[0].keys()
    for state_dict in state_dicts:
        assert state_dict.keys() == keys, "All state_dicts must have the same structure!"
    new_state_dict = {}
    for key in keys:
        weighted_param = weights[0] * state_dicts[0][key]
        for i in range(1, len(state_dicts)):
            weighted_param += weights[i] * state_dicts[i][key]

        new_state_dict[key] = weighted_param
    return new_state_dict