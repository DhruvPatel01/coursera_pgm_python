from string import ascii_lowercase
import pickle
from collections import namedtuple
import os, sys

from scipy.io import loadmat
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.pardir))
import commons
from commons import utils
from commons.factor import Factor


def pa3_data():
    allWords = loadmat('./data/PA3Data.mat')['allWords']

    data = []
    for datum in allWords:
        datum = datum[0]
        datum_new = {'img': [], 'ground_truth': [], 'word': []}
        for img, ground_truth in zip(datum['img'], datum['groundTruth']):
            img = img[0]
            ground_truth = ground_truth[0].item() - 1 # matlab as 1 based index
            datum_new['img'].append(img)
            datum_new['ground_truth'].append(ground_truth)
            datum_new['word'].append(ascii_lowercase[ground_truth])
        data.append(datum_new)

    return data


def pa3_models():
    mat = loadmat('./data/PA3Models.mat', simplify_cells=True)

    data = {
        'image_model': mat['imageModel'],
        'pairwise_model': mat['pairwiseModel'],
        'triplet_list': []
    }

    for triplet in mat['tripletList']:
        chars = tuple(triplet['chars']-1)  # 'a' is zero now and not 1
        factor_val = triplet['factorVal']
        data['triplet_list'].append((chars, factor_val))

    return data


def pa3_sample_cases(is_test=False):
    if is_test:
        mat = loadmat('./data/PA3TestCases.mat', simplify_cells=True)
    else:
        mat = loadmat('./data/PA3SampleCases.mat', simplify_cells=True)
    out = {}
    for k, v in mat.items():
        if k.startswith('__'):
            continue
        if k.lower().endswith('imagesinput'):
            char_new = {'img': [], 'ground_truth': [], 'word': []}
            for char in v:
                char_new['img'].append(char['img'])
                if 'ground_truth' in char:
                    ground_truth = char['groundTruth'] - 1 # matlab as 1 based index
                    char_new['ground_truth'].append(ground_truth)
                    char_new['word'].append(ascii_lowercase[ground_truth])
            out['part%s_sample_image_input'%k[4]] = char_new
        elif k.lower().endswith('factorsoutput'):
            factors = []
            for factor_dict in v:
                factors.append(Factor.from_matlab(factor_dict))
            out['part%s_sample_factors_output'%k[4]] = factors
        elif k.lower().endswith('factoroutput'):
            factor = Factor.from_matlab(v)
            out['part%s_sample_factor_output'%k[4]] = factor
        elif k.lower().endswith('factorsinput'):
            factors = []
            for factor_dict in v:
                factors.append(Factor.from_matlab(factor_dict))
            out['part%s_sample_factors_input'%k[4]] = factors
        else:
            raise NotImplementedError(k)

    return out


if __name__ == '__main__':
    pa3_data()
    pa3_models()