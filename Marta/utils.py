import random
import numpy as np
import pandas as pd
from copy import copy
import warnings

def generate_world_bounds(features, feature_bounds = (0, 1), endThreshold = .20, endArea = .40, max_depth = 100, depth = 0, current_bounds = None):
    if type(features) is int:
        features = np.arange(features)
    if type(feature_bounds) is tuple and len(feature_bounds) == 2:
        feature_bounds = [feature_bounds for _ in range(len(features))]
    if len(features) != len(feature_bounds):
        print(f"features {features} and feature_bounds {feature_bounds} not the same size")
        return
    if current_bounds is None:
        current_bounds = copy(feature_bounds)
    
    args = {"features": features,
            "feature_bounds": feature_bounds,
            "endThreshold": endThreshold,
            "endArea": endArea,
            "max_depth": max_depth,
            }
    bounds = set()

    feature = random.randint(0, len(features) - 1)
    start, end = current_bounds[feature]
    f_start, f_end = feature_bounds[feature]
    random_offset = (f_end - f_start) * endThreshold
    start_random, end_random = start + random_offset, end - random_offset
    if end_random <= start_random or \
            depth > max_depth or \
            np.multiply.reduce([b - a for a, b in feature_bounds]) * endArea > np.multiply.reduce([b - a for a, b in current_bounds]):
        return {tuple(current_bounds)}
 
    cut = random.uniform(start_random, end_random)
    
    current_bounds[feature] = (start, cut)
    bounds |= generate_world_bounds(current_bounds = copy(current_bounds), depth = depth + 1, **args)
    
    current_bounds[feature] = (cut, end)
    bounds |= generate_world_bounds(current_bounds = copy(current_bounds), depth = depth + 1, **args)
    return bounds

def generate_true_values(bounds_with_class, feature_bounds):
    if type(feature_bounds) is tuple and len(feature_bounds) == 2:
        feature_bounds = [feature_bounds for _ in range(len(list(bounds_with_class.items())[0][0]))]
    points = []
    for _ in range(10**4): #number of values "dots in graph"
        val = [random.uniform(feature_bounds[i][0], feature_bounds[i][1]) for i in range(len(feature_bounds))]
        for bounds, cls in bounds_with_class.items():
            if all([start < val[i] < end for i, (start, end) in enumerate(bounds)]):
                points.append(np.array([cls] + val))
    return np.array(points)

def add_unc(tv, errRange = 1, corr = 1):
    if 0.58 > corr or corr > 1:
        raise ValueError(f"corr must be between 0.58 and 1 got {corr}")
    _saved_corr = {
        1: 1
    }
    _p = np.poly1d(np.array([ 2.52675290e+06, -1.84002863e+07,  6.02862434e+07, -1.17400405e+08,
        1.51226264e+08, -1.35686740e+08,  8.69776830e+07, -4.02013544e+07,
        1.33565614e+07, -3.14334758e+06,  5.09552210e+05, -5.43413126e+04,
        3.54198523e+03, -1.26493380e+02,  4.12474405e+00, -1.00546571e+00]))
    
    def add_unc_single_value(tv, errRange, corr):
        if corr not in _saved_corr.keys():
            _saved_corr[corr] = _p(corr)
        err = errRange * random.random() * [-1,+1][random.randint(0,1)]
        ov = tv + err
        unc = abs(err * random.uniform(_saved_corr[corr] , 1))
        return ov, unc
    
    def add_unc_array(tv, errRange, corr):
        ov_unc = []
        for t in tv:
            ov_unc.append(add_unc_single_value(t, errRange, corr))
        return np.array(ov_unc)
    
    try:
        iter(tv)
    except:
        _add_unc = add_unc_single_value
    else:
        _add_unc = add_unc_array
    
    return _add_unc(tv, errRange, corr)

def data_to_world(data, errRange = 1, corr = 1):
    _, n_feat = data.shape
    n_feat -= 1
    args = {
        "errRange": errRange,
        "corr" : corr
    }
    arr = []
    i = 0
    arr.append(data[:, i])
    i += 1
    for _ in range(n_feat):
        arr.append(data[:, i])
        i += 1

    for i in range(n_feat):
        ovunc = add_unc(data[:, i + 1], **args)
        arr.append(ovunc[:, 0]) 
        arr.append(ovunc[:, 1])

    #Class, True Value 1, True Value 2, Observed Value 1, Uncertainty 1, Observed Value 2, Uncertainty 2
    return np.stack(arr, axis=1)

def sample_data(data, sample_per_class = 20):
    for i, value in enumerate(np.unique(data[:, 0])):
        indices, = np.where(data[:, 0] == value)
        data_per_class = data[indices]
        bagged_data_class = data_per_class[np.random.choice(len(data_per_class), size=sample_per_class, replace=False)]
        if i == 0:
            bagged_data = bagged_data_class
        else:
            bagged_data = np.concatenate((bagged_data, bagged_data_class))
    return bagged_data

def generate_world(seed, features, feature_bounds = (0, 1), endArea = .3, max_depth = 4, class_number = 2, errRange = 0.1, corr = 1):
    random.seed(seed)
    np.random.seed(seed)
    world_coords = generate_world_bounds(features = features, feature_bounds = feature_bounds, endThreshold = .2, endArea = endArea, max_depth = max_depth)
    bounds_with_class = {}
    if class_number > len(world_coords):
        warnings.warn("Class number greater than the amount of available leafs, try lowering the endArea parameter")
    random_list = list(np.arange(class_number)) + [random.randint(0, class_number - 1) for _ in range(len(world_coords) - class_number)]
    np.random.shuffle(random_list)
    for i, bounds in zip(random_list, world_coords):
        bounds_with_class[bounds] = i
    data = generate_true_values(bounds_with_class, feature_bounds)
    
    return data_to_world(data, errRange = errRange, corr = corr)

def generate_world_pandas(seed, features, feature_bounds = (0, 1), endArea = .3, max_depth = 4, class_number = 2):
    random.seed(seed)
    np.random.seed(seed)
    world_coords = generate_world_bounds(features = features, feature_bounds = feature_bounds, endThreshold = .2 , endArea = endArea, max_depth = max_depth)
    bounds_with_class = {}
    if class_number > len(world_coords):
        warnings.warn("Class number greater than the amount of available leafs, try lowering the endArea parameter")
    random_list = list(np.arange(class_number)) + [random.randint(0, class_number - 1) for _ in range(len(world_coords) - class_number)]
    np.random.shuffle(random_list)
    for i, bounds in zip(random_list, world_coords):
        bounds_with_class[bounds] = i
    data = generate_true_values(bounds_with_class, feature_bounds)
    df = pd.DataFrame(data, columns=['Class'] + [f'True Value {i}' for i in range(1, features+1)])
    return df