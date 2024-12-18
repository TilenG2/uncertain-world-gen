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
    """
    return a array set of bounds like 
    ((f1_begin, f1_end), (f2_begin, f2_end), ... (fn_begin, fn_end))
    """
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
                class_with_bounds_list = list(class_with_bounds[cls])
                weights = [np.prod([feature_end - feature_begin for feature_begin, feature_end in area]) for area in class_with_bounds_list]
                val = [random.uniform(minV, maxV) for minV, maxV in random.choices(class_with_bounds_list, weights=weights, k=1)[0]]
                points.append(np.array([cls] + val))
    else:
        bounds_set = set()
        for b in class_with_bounds.values():
            bounds_set |= b
        for _ in range(no_samples):
            bounds_set_list = list(bounds_set)
            weights = [np.prod([feature_end - feature_begin for feature_begin, feature_end in area]) for area in bounds_set_list]
            bound = random.choices(bounds_set_list, weights=weights, k=1)[0]
            for cls, bounds in class_with_bounds.items(): # find the correct class for selected bound
                if bound in bounds:
                    break
            val = [random.uniform(minV, maxV) for minV, maxV in bound]
            points.append(np.array([cls] + val))
    return np.array(points)

def add_unc(tv, correlation, errRange):
    error = np.random.uniform(-errRange, errRange, tv.shape)
    obs = tv + error
    X = np.abs(error)
    
    if errRange == 0: # To ensure not to have nan in uncertainty vector. If error_range is 0, set Uncertainty directly to 0
        Y = np.zeros_like(X)
    
    else:
        # Standardize X
        X_standardized = (X - X.mean()) / X.std()
        
        # Generate Z such that correlation is maintained
        Z = np.random.uniform(0, 1, X.shape)
        Z_standardized = (Z - Z.mean()) / Z.std()
        
        # Create Y with the specified correlation
        Y_standardized = correlation * X_standardized + np.sqrt(1 - correlation**2) * Z_standardized
        
        # Scale Y back to the original scale of X
        Y = Y_standardized * X.std() + X.mean()
        
        # Calculate the uncertainty Y = ZX
        #Y = Y * X.mean()  # Adjust Y with the mean of X to ensure scaling
    
    # Add the calculated uncertainty to the dataset
    unc = Y.astype(np.float32)

    return np.array([obs, unc, error])

def add_noise(data, correlation, errRange, mutation_rate, feature_bounds):
    real_values = []
    minval, maxval = feature_bounds
    minval -= errRange
    maxval += errRange
    for i in range(1, len(data[0])):
        tv = data[:,i]
        rv = add_unc(tv, correlation, errRange)[0]
        real_values.append(rv)
    mutated_values = range(len(data))
    mutated_values = np.random.choice(mutated_values, int(len(mutated_values) * mutation_rate), replace=False) # ne deluje če je len(mutated_values) * mutation_rate < 1
    for i in mutated_values:
        for j in range(0, len(data[0])-1):
            real_values[j][i] = np.random.uniform(minval, maxval)
    return real_values

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
        ov_unc_err = add_unc(data[:, i + 1], **args)
        arr.append(ov_unc_err[0]) 
        arr.append(ov_unc_err[1])
        arr.append(ov_unc_err[2])

    #Class, True Value 1, True Value 2, Observed Value 1, Uncertainty 1, Error 1, Observed Value 2, Uncertainty 2, Error 2
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

def sample_data_pandas(data, sample_per_class = 20):
    for i, value in enumerate(np.unique(data["Class"])):
        data_per_class = data[data["Class"]== value].reset_index(drop=True)
        bagged_data_class = data_per_class.iloc[np.random.choice(len(data_per_class), size=sample_per_class, replace=False)]
        if i == 0:
            bagged_data = bagged_data_class
        else:
            bagged_data = pd.concat([bagged_data, bagged_data_class])
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
    
    np.random.seed(data_seed)
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
    
    columns = ["Class"] + [f"True Value {feature}" for feature in range(1, features + 1)]
    for feature in range(1, features + 1):
        columns += [f"Observed Value {feature}", f"Uncertainty {feature}", f"Error {feature}"]
        
    df = pd.DataFrame(data, columns=columns)
    return df

def generate_real_world(world_seed, data_seed, features,
                        no_samples = 10**4,
                        equal_classes = False,
                        feature_bounds = (0, 1),
                        endArea = .3,
                        endThreshold = .2,
                        max_depth = 4,
                        class_number = 2,
                        errRange = 0.1,
                        corr = 1,
                        mutation_rate = 0.1,
                        errRange_real_world = 0.1,
                        corr_real_world = 1):

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
    
    np.random.seed(data_seed)
    data = np.concatenate([data, np.stack(add_noise(data, corr_real_world, errRange_real_world, mutation_rate, feature_bounds), axis=1)], axis=1)
    
    data_temp = data_to_world(np.vstack([data[:,0], data[:,features + 1:].T]).T, errRange = errRange, corr = corr)
    return np.concatenate([data[:,1:features + 1], data_temp], axis=1)

def generate_real_world_pandas(world_seed, data_seed, features,
                        no_samples = 10**4,
                        equal_classes = False,
                        feature_bounds = (0, 1),
                        endArea = .3,
                        endThreshold = .2,
                        max_depth = 4,
                        class_number = 2,
                        errRange = 0.1,
                        corr = 1,
                        mutation_rate = 0.1,
                        errRange_real_world = 0.1,
                        corr_real_world = 1):
    
    data = generate_real_world(world_seed, data_seed, features,
                        no_samples = no_samples,
                        equal_classes = equal_classes,
                        feature_bounds = feature_bounds,
                        endArea = endArea,
                        endThreshold = endThreshold,
                        max_depth = max_depth,
                        class_number = class_number,
                        errRange = errRange,
                        corr = corr,
                        mutation_rate = mutation_rate,
                        errRange_real_world = errRange_real_world,
                        corr_real_world = corr_real_world)
    
    columns = [f"True Value {feature}" for feature in range(1, features + 1)] + ["Class"] + [f"Real Value {feature}" for feature in range(1, features + 1)]
    for feature in range(1, features + 1):
        columns += [f"Observed Value {feature}", f"Uncertainty {feature}", f"Error {feature}"]
        
    df = pd.DataFrame(data, columns=columns)
    return df

def sample_data_orange(data, samples_per_class = 20):
    for i, value in enumerate(np.unique(data.Y)):
        indices, = np.where(data.Y == value)
        data_per_class = data[indices]
        sampled_data_class = data_per_class[np.random.choice(len(data_per_class), size=samples_per_class, replace=False)]
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