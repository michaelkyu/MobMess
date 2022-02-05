from math import ceil

from plasx.utils import *
from mobmess.graph_utils import *
from mobmess.pd_utils import *
from mobmess.sp_utils import *


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


def setdiag(X, values):
    """Set the diagonal of a scipy.sparse matrix. A reimplementation of scipy.sparse's setdiag method.

    This implementation avoids explicitly trying to set a diagonal element that is already 0 to be 0, again.

    The scipy's implementation would retain values of 0, which is inefficient.

    https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.sparse.csc_matrix.setdiag.html
    """

    try:
        iter(values)
    except:
        values = np.repeat(np.array([values], X.dtype), X.shape[0])

    assert X.shape[0] == X.shape[1]
    assert X.shape[0] == len(values)

    assert not scipy.sparse.isspmatrix_coo(X), 'Cannot apply on coo_matrix'

    # Filter out zeros
    i = np.arange(X.shape[0])
    to_use = (X.diagonal() != 0) | (values != 0)
    i, values = i[to_use], values[to_use]

    # Set diagonal
    X[i, i] = values

