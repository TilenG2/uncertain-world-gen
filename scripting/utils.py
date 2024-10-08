import random
import numpy as np
import pandas as pd
from copy import copy
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable


def old_generate_world_bounds(xy1 = (-1, -1), xy2 = (1, 1), orig_size = None, endP = .40, edgeT = .1):
    args = {"orig_size": orig_size,
            "endP": endP,
            "edgeT": edgeT,
            }
    x1, y1 = xy1
    x2, y2 = xy2
    bounds = set()
    if orig_size is None:
        args["orig_size"] = orig_size = abs((x2 - x1) * (y2 - y1))

    if abs((x2 - x1) * (y2 - y1)) < orig_size * endP:
        return {(xy1, xy2)}
    split = ["x","y"][random.randint(0,1)]
    if split == "x":
        x = x1
        i = 0
        while abs(x - x1) < edgeT or abs(x2 - x) < edgeT:
            if i > 500:
                return {(xy1, xy2)}
            i += 1
            x = random.uniform(x1, x2)
        bounds |= old_generate_world_bounds( xy1,   (x, y2), **args)
        bounds |= old_generate_world_bounds((x, y1), xy2,    **args)
    elif split == "y":
        y = y1
        i = 0
        while abs(y - y1) < edgeT or abs(y2 - y) < edgeT:
            if i > 500:
                return {(xy1, xy2)}
            i += 1
            y = random.uniform(y1, y2)
        bounds |= old_generate_world_bounds( xy1,   (x2, y), **args)
        bounds |= old_generate_world_bounds((x1, y), xy2,    **args)
    return bounds

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

def generate_true_values(seed, class_with_bounds,
                         no_samples = 10**4,
                         equal_classes = False):
    random.seed(seed)
    points = []
        
    if equal_classes:
        for cls in class_with_bounds.keys():
            for _ in range(no_samples):
                val = [random.uniform(minV, maxV) for minV, maxV in random.choice(list(class_with_bounds[cls]))]
                points.append(np.array([cls] + val))
    else:
        bounds_set = set()
        for b in class_with_bounds.values():
            bounds_set |= b
        for _ in range(no_samples):
            bound = random.choice(list(bounds_set))
            for cls, bounds in class_with_bounds.items():
                if bound in bounds:
                    break
            val = [random.uniform(minV, maxV) for minV, maxV in bound]
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
        "correlation" : corr
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
        bagged_data_class = data_per_class[random.choice(len(data_per_class), size=sample_per_class, replace=False)]
        if i == 0:
            bagged_data = bagged_data_class
        else:
            bagged_data = np.concatenate((bagged_data, bagged_data_class))
    return bagged_data

def generate_world(world_seed, data_seed, features,
                   no_samples = 10**4,
                   equal_classes = False,
                   feature_bounds = (0, 1),
                   endArea = .3,
                   endThreshold = .2,
                   max_depth = 4,
                   class_number = 2,
                   errRange = 0.1,
                   corr = 1):

    args = {"features": features,
            "feature_bounds": feature_bounds,
            "endThreshold": endThreshold,
            "endArea": endArea,
            "max_depth": max_depth,
            }
    
    random.seed(world_seed)
    world_coords = generate_world_bounds(**args)
    
    from collections import defaultdict
    class_with_bounds = defaultdict(set)
    random_list = list(np.arange(class_number)) + [random.randint(0, class_number - 1) for _ in range(len(world_coords) - class_number)]
    random.shuffle(random_list)
    for i, bounds in zip(random_list, world_coords):
        class_with_bounds[i] |= {bounds}
    
    data = generate_true_values(data_seed, class_with_bounds, no_samples, equal_classes)
    
    return data_to_world(data, errRange = errRange, corr = corr)

def generate_world_pandas(world_seed, data_seed, features,
                   no_samples = 10**4,
                   equal_classes = False,
                   feature_bounds = (0, 1),
                   endArea = .3,
                   endThreshold = .2,
                   max_depth = 4,
                   class_number = 2,
                   errRange = 0.1,
                   corr = 1):
    
    data = generate_world(world_seed, data_seed, features,
                          no_samples = no_samples,
                          equal_classes = equal_classes,
                          feature_bounds = feature_bounds,
                          endArea = endArea,
                          endThreshold = endThreshold,
                          max_depth = max_depth,
                          class_number = class_number,
                          errRange = errRange,
                          corr = corr)
    
    df = pd.DataFrame(data, columns=['Class'] + [f'True Value {i}' for i in range(1, features+1)] + list(np.array(list(zip([f'Observed Value {i}' for i in range(1, features+1)], [f'Uncertainty {i}' for i in range(1, features+1)]))).flatten()))
    return df

def sample_data_orange(data, samples_per_class = 20):
    for i, value in enumerate(np.unique(data.Y)):
        indices, = np.where(data.Y == value)
        data_per_class = data[indices]
        sampled_data_class = data_per_class[random.choice(len(data_per_class), size=samples_per_class, replace=False)]
        if i == 0:
            sampled_data = sampled_data_class
        else:
            sampled_data = Table.concatenate((sampled_data, sampled_data_class))
    return sampled_data

def convert_np_to_orange(world, number_features):
    
    columns = ["Class"] + [f"True Value {feature}" for feature in range(1, number_features + 1)]
    for feature in range(1, number_features + 1):
        columns += [f"Observed Value {feature}", f"Uncertainty {feature}"]
            
    world = pd.DataFrame(world, columns=columns) 
    
    world["Class"] = world["Class"].astype(int)

    X = np.column_stack([world[f"Observed Value {i+1}"] for i in range(number_features)])
    Y = np.array(world["Class"])
    M = np.column_stack([world[f"Uncertainty {i+1}"] for i in range(number_features)])
    Xtv = np.column_stack([world[f"True Value {i+1}"] for i in range(number_features)])
    
    domain = Domain(
        attributes = [ContinuousVariable(f"Observed Value {i+1}") for i in range(number_features)],
        class_vars = DiscreteVariable("Class", values=[str(i) for i in range(max(Y+1))]),
        metas = [ContinuousVariable(f"Uncertainty {i+1}") for i in range(number_features)]
    )
    data = Table.from_numpy(domain, X=X, Y=Y, metas=M)
    
    domain = Domain(
        attributes = [ContinuousVariable(f"Observed Value {i+1}") for i in range(number_features)],
        class_vars = DiscreteVariable("Class", values=[str(i) for i in range(max(Y+1))])
    )
    # X = tv1.reshape(-1, 1)
    test_data = Table.from_numpy(domain, X=Xtv, Y=Y)
    return data, test_data