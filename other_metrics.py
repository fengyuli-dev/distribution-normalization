'''
Automatic generation evaluation metrics wrapper
The most useful function here is
get_all_metrics(refs, cands)
Source: https://github.com/jmhessel/clipscore/blob/main/generation_eval_utils.py
'''
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice



def get_all_metrics(refs, cands, return_per_cap=True):
    metrics = []
    names = []

    pycoco_eval_cap_scorers = [(Bleu(1), 'bleu1'),
                               (Bleu(4), 'bleu4'),     
                               (Cider(), 'cider')]

    for scorer, name in pycoco_eval_cap_scorers:
        overall, per_cap = pycoco_eval(scorer, refs, cands)
        if return_per_cap:
            metrics.append(per_cap)
        else:
            metrics.append(overall)
        names.append(name)

    metrics = dict(zip(names, metrics))
    return metrics


def tokenize(refs, cands):
    tokenizer = PTBTokenizer()
    refs = {idx: [{'caption':r} for r in c_refs] for idx, c_refs in enumerate(refs)}
    cands = {idx: [{'caption':c}] for idx, c in enumerate(cands)}
    refs = tokenizer.tokenize(refs)
    cands = tokenizer.tokenize(cands)
    return refs, cands


def pycoco_eval(scorer, refs, cands):
    '''
    scorer is assumed to have a compute_score function.
    refs is a list of lists of strings.
    cands is a list of predictions.
    '''
    refs, cands = tokenize(refs, cands)
    average_score, scores = scorer.compute_score(refs, cands)
    return average_score, scores
