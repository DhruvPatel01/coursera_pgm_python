import subprocess
import itertools
from string import ascii_lowercase

import numpy as np

import sol


def factors_to_fg(factors, file_path):
    """Converts factor graph into .fg file

    This should only work if your factor names are
    integer and domains of that are also integers
    in the range of [0, cardinality(variable))

    Luckily PGM course assignments has this structure.
    """
    lines = [len(factors), ""]

    for f in factors:
        lines.append("%d" % (len(f.vars, )))
        lines.append(" ".join(map(str, f.vars)))
        lines.append(" ".join(str(len(d)) for d in f.domains))
        placeholder_idx = len(lines)
        lines.append(None)  # will be replace by nonzero count once we know

        domains = reversed(f.domains)
        # libDAI expects first variable to change fastest
        # but itertools.product changes the last element fastest
        # hence reversed list
        n_nonzero = 0
        for i, assignment in enumerate(itertools.product(*domains)):
            assignment = tuple(reversed(assignment))
            # if abs(f[assignment]) < 1e-5:
            #     continue
            n_nonzero += 1
            line = "%d %.9g" % (i, f[assignment])
            lines.append(line)
        lines[placeholder_idx] = "%d" % (n_nonzero, )
        lines.append("")

    with open(file_path, 'wt') as f:
        for line in lines:
            print(line, file=f)


def image_similarity(img1, img2):
    mean_sim = 0.283  # Avg sim score computed over held - out data.
    img1, img2 = img1.reshape(-1), img2.reshape(-1)
    cosine_dist = (img1 @ img2)/(np.sqrt((img1**2).sum()) * np.sqrt((img2**2).sum()))
    diff = (cosine_dist - mean_sim)**2

    if cosine_dist > mean_sim:
        return 1 + 5*diff
    else:
        return 1 / (1 + 5*diff)


def compute_image_factor(img, img_model):
    img = np.array(img)
    img = img.T.reshape(-1)  # matlab is column first oriented!
    X = img
    N = len(X)
    K = img_model['K']

    theta = np.array(img_model['params'][:N*(K-1)]).reshape(N, K-1).T
    bias = np.array(img_model['params'][N*(K-1):]).reshape(-1)

    W = (theta @ X) + bias
    W = np.concatenate([W, [0]])
    W -= W.max()
    W = np.exp(W)
    return W/W.sum()


def run_inference(factors):
    fg_path = './factors.fg'
    factors_to_fg(factors, fg_path)

    output = subprocess.run(['./inference/doinference-linux', './factors.fg', 'map'], 
                            text=True, capture_output=True)

    if output.returncode != 0:
        raise Exception("doinference command failed:" + output.stderr)

    lines = output.stdout.rstrip().split('\n')
    n_lines = int(lines[0])
    if len(lines) != n_lines+1:
        raise ValueError("Parsing error")

    values = [s.split(' ') for s in lines[1:]]
    return {int(k): int(v)-1 for k, v in values}
    # don't really know why -1 is needed here, but it works!


def build_ocr_network(images, image_model, pairwise_model=None, triplet_list=None):
    factors = sol.compute_single_factors(images, image_model)

    if pairwise_model is not None:
#         factors.extend(sol.compute_equal_pairwise_factors(images, image_model['K']))
        factors.extend(sol.compute_pairwise_factors(images, pairwise_model, image_model['K']))

    if triplet_list is not None:
        factors.extend(sol.compute_triplet_factors(images, triplet_list, image_model['K']))

    if not image_model.get('ignore_similarity', True):
        all_sim_factors = sol.compute_all_similarity_factors(images, image_model['K'])
        factors.extend(sol.choose_top_similarity_factors(all_sim_factors, 2))

    return factors


def compute_word_predictions(all_words, image_model, pairwise_model=None, triplet_list=None):
    predictions = []
    for i, word in enumerate(all_words):
        factors = build_ocr_network(word, image_model, pairwise_model, triplet_list)
        prediction = run_inference(factors)
        predictions.append([prediction[i] for i in range(len(prediction))])
    return predictions


def score_predictions(words, predictions, show_output=True):
    assert len(words) == len(predictions), "Length mismatch"

    n_words_correct = n_words_total = 0
    n_chars_correct = n_chars_total = 0

    for word, pred in zip(words, predictions):
        n_words_total += 1
        n_chars_total += len(pred)

        n_chars_tmp = sum(1 for x, y in zip(word['ground_truth'], pred) if x==y)
        n_chars_correct += n_chars_tmp

        if n_chars_tmp == len(pred):
            n_words_correct += 1

        if show_output:
            correct = ''.join(ascii_lowercase[c] for c in word['ground_truth'])
            predicted = ''.join(ascii_lowercase[c] for c in pred)
            print("%s predicted as %s" % (correct, predicted))

    char_acc = n_chars_correct/n_chars_total
    word_acc = n_words_correct/n_words_total
    if show_output:
        print("Char accuracy: %.3f" % char_acc)
        print("Word accuracy: %.3f" % word_acc)

    return char_acc, word_acc


def score_model(words, image_model, pairwise_model=None, triplet_list=None):
    preds = compute_word_predictions(words, image_model, pairwise_model, triplet_list)
    return score_predictions(words, preds)
