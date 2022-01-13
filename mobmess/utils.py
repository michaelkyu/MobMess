from math import ceil

from plasx.utils import *
from mobmess.graph_utils import *

def split_indices(n, k=None, chunk=None):
    try:
        iter(n)
        n = len(n)
    except TypeError:
        assert isinstance(n, int)

    if n==0:
        return []
    else:
        assert (k is None) != (chunk is None), 'Exactly one of `k` OR `chunk` must be specified'

        if chunk is None:
            chunk = int(ceil(float(n) / k))    
        return [(chunk * a, min(chunk * (a+1), n)) for a in range(int(ceil(float(n) / chunk)))]

def to_chunks(it, k=None, chunk=None, gen=False):
    chunk_list = (it[i:j] for i, j in split_indices(len(it), k=k, chunk=chunk))
    if not gen:
        chunk_list = list(chunk_list)
    return chunk_list

