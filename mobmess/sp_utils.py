#############################################################
#
# Utility functions for manipulating scipy.sparse matrices
#
#############################################################

import numpy as np
import scipy.sparse

from mobmess import nb_utils

def as_flat(x):
    if scipy.sparse.issparse(x) and (x.nnz==0):
        # Need to handle the special case when the input `x` is 1-by-0 scipy sparse matrix.
        # - This happens when you index a scipy matrix with empty indices, e.g. sp[[],[]].
        # - Solution: manually create an empty 1-D array with the same dtype
        return np.array([], dtype=x.dtype)
    else:
        return np.array(x).flatten()


def sp_as_indexable(sp):
    """Changes a coo_matrix into a csr_matrix, which can be indexed"""
    if scipy.sparse.isspmatrix_coo(sp):
        return sp.tocsr()
    else:
        return sp


def copy_sp(x, dtype=None):
    """
    Copies a scipy matrix, using fast numba-based copy and casting functions.
    """

    if (dtype is not None) and (dtype != x.dtype):
        # Cast to data type
        data = nb_utils.nb_cast(x.data, dtype)
    else:
        data = nb_utils.nb_copy(x.data)

    if scipy.sparse.isspmatrix_coo(x):
        row, col = nb_utils.nb_copy(x.row), nb_utils.nb_copy(x.col)
        return scipy.sparse.coo_matrix((data, (row, col)), shape=x.shape, copy=False)
    elif scipy.sparse.isspmatrix_csr(x):
        indptr, indices = nb_utils.nb_copy(x.indptr), nb_utils.nb_copy(x.indices)
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=x.shape, copy=False)
    elif scipy.sparse.isspmatrix_csc(x):
        indptr, indices = nb_utils.nb_copy(x.indptr), nb_utils.nb_copy(x.indices)
        return scipy.sparse.csc_matrix((data, indices, indptr), shape=x.shape, copy=False)
    else:
        raise Exception()


def sp_mask(x, y=None, i=None, j=None, invert=False, triu=False, tril=False, blocks=None):
    """Create sparse matrix with same values as x, but only at indices
    where y is also non-zero.  This is like a masking operation.
    
    Note that the output is slightly different than x.multiply(y >
    0). Differences occur when np.inf is a value in x. This function
    will preserve np.inf to stay as np.inf, while x.multiply(y > 0)
    will change it to np.nan (because np.inf * True = np.nan)

    i, j : Instead of specifying a matrix `y`, specify the indices `i` and `j` to keep.

    blocks : 
    
        Keep only a block diagonal matrix. Specify a list of blocks, e.g. [[5,4], [3,6]], where each block is the indices in that block. Note that this isn't a classical definition of a block diagonal matrix, as each block's indices can be scattered across the matrix. If you were to permute the rows and columns of the matrix, then you could make a classical block diagonal

    invert :

        if True, then output everything EXCEPT at (i,j) or where y is zero
    
        Default: False
    """
    
    if y is not None:
        assert y.dtype == np.bool
        i, j = y.nonzero()
        assert as_flat(sp_as_indexable(y)[i,j]).all()

    elif triu or tril:
        # Retain only the upper triangle (triu) or the lower triangle (tril).
        # Don't keep the diagonal in either case.
        assert triu != tril
        assert not invert

        i, j = x.nonzero()
        mask = (i<j) if triu else (j<i)
        i, j = i[mask], j[mask]

    elif blocks is not None:
        assert invert is False, "Not implemented"

        ### Keep only blocks

        blocks_arr = np.concatenate(list(blocks))
        blocks_len_cumsum = np.cumsum(np.array([len(x) for x in list(blocks)]))

        new_x = x.tocsr()[blocks_arr,:][:,blocks_arr]
        new_x.sort_indices()

        ub = np.repeat(blocks_len_cumsum, new_x.indptr[blocks_len_cumsum] - new_x.indptr[np.append(0,blocks_len_cumsum[:-1])])
        lb = np.repeat(np.append(0, blocks_len_cumsum[:-1]), new_x.indptr[blocks_len_cumsum] - new_x.indptr[np.append(0,blocks_len_cumsum[:-1])])
        mask = (new_x.indices >= lb) & (new_x.indices < ub)

        new_x.data[~ mask] = 0
        new_x.eliminate_zeros()
        new_x.sort_indices()

        blocks_arr_idx = np.argsort(blocks_arr)
        new_x = new_x[blocks_arr_idx,:][:,blocks_arr_idx]
        new_x.sort_indices()        

        return new_x

    if invert:
        new_x = copy_sp(sp_as_indexable(x))
        new_x[i,j] = 0
        new_x.eliminate_zeros()
    else:
#        if len(i)==0:  return x.copy()

        # Need to handle the special case where i and j are empty. 
        # - Normally, when i and j are not empty, then x[i,j] returns a np.matrix that's converted to an array
        # - But if i and j are empty, then x[i,j] returns an 1-by-0 scipy sparse matrix.
        # if len(i)==0:
        #     data = np.array([], dtype=x.dtype)
        # else:
        #     data = as_flat(sp_as_indexable(x)[i,j])
        data = as_flat(sp_as_indexable(x)[i,j])

        # Filter out indices where x is 0
        idx = data != 0
        data, i, j = data[idx], i[idx], j[idx]

        new_x = scipy.sparse.coo_matrix((data, (i,j)), shape=x.shape)

    return new_x



def triu_take(m, k=1):
    """Return the upper-triangle of a matrix, flattened out into a 1D array.

    k :

        Same as `k` for np.triu_indices

        k=1 excludes the diagonal (default)
        k=0 includes the diagonal
    """

    assert len(m.shape)==2 and m.shape[0]==m.shape[1], 'Input is not a square matrix'

    import pandas as pd
    if isinstance(m, pd.DataFrame):
        m = m.values

    if isinstance(m, np.ndarray):
        return m[np.triu_indices(m.shape[0], k=1)]
    else:
        raise Exception()
