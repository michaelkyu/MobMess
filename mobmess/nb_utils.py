from plasx.nb_utils import *


@jit(nopython=True, nogil=True)
def cov_to_dense(pos, data, length, copy_data=False, data_cumsum=True):
    """Converts a sparse representation of coverage, a triple (pos, data,
    length), to a np.array that is a dense representation

    *** NOTE ***
    If copy_data=False, then `data` will be replaced with effectively np.cumsum(data)

    In the normal workflow, cov_to_dense() is called by get_coverage()
    and get_coverage_nb(). Initially, `data` is the not the actual
    coverage values, but the change in coverage values, i.e. add
    data[i] to a running total. By running the default params
    cov_to_dense(copy_data=False, data_cumsum), `data` is converted
    into actual coverage values.

    """
    
    if copy_data:
        data = data.copy()

    # Verify that the specific contig length is greater than any coordinate listed (otherwise, segfault will occur)
    assert length >= pos.max()

#    p, n = 0, 0
    p, n = np.uint32(0), np.uint32(0)
    cov = np.zeros(length, np.uint32)
    if data_cumsum:
        for i in range(pos.size):
            for j in range(p, pos[i]):
                cov[j] = n
            n += data[i]
            data[i] = n
            p = pos[i]
    else:
        for i in range(pos.size):
            for j in range(p, pos[i]):
                cov[j] = n
            p = pos[i]
            n = data[i]
    for j in range(p, cov.size):
        cov[j] = n
    return cov


@jit(nopython=True, nogil=True)
def get_coverage_nb(pos, is_start, length, get_dense=True, dtype=np.int32):
    # Convert True --> 1, and False --> -1
#    data = 2 * is_start.astype(np.int32) - 1
    data = 2 * is_start.astype(dtype) - 1

    # pos[i] = nucleotide position
    # data[i] = net number of reads that start or stop at that position
    pos, data = sum_duplicates_sorted(pos.reshape(-1, 1), data, data.dtype)
    pos = pos.flatten()
    if get_dense:
        cov = cov_to_dense(pos, data, length)
    else:
        # It seems that returning None causes an error in ani_utils.get_query_coverage. So instead, I will return an array of length 1 
#        cov = None
        cov = np.zeros(1, np.uint32)

#    cov = cov_to_dense(pos, data, length)
    
    return cov, pos, data




def sum_duplicates_sorted_base(key, data=None, dtype=np.int64):
    """Numba implementation of scipy.sparse.coo._sum_duplicates(), with
    the assumption that rows and cols are sorted

    """

    if data is None:
        data = np.ones(key.shape[0], dtype)

    if key.shape[0] == 0:  return key, data

    key_dim = key.shape[1]
    boundaries = ((key[1:, :] == key[:-1, :]).sum(1) != key_dim).nonzero()[0] + 1
    n = boundaries.size + 1

    uniq_sums = np.zeros(n, data.dtype)
    uniq_key = np.zeros((n, key_dim), key.dtype)
    uniq_key[0, :] = key[0, :]
    uniq_key[1:, :] = key[boundaries, :]

    start = 0
    for i in range(boundaries.size):
        end = boundaries[i]
        for j in range(start, end):
            uniq_sums[i] += data[j]
        start = end

    # close it up
    i = boundaries.size
    end = data.size
    for j in range(start, end):
        uniq_sums[i] += data[j]

    return uniq_key, uniq_sums


sum_duplicates_sorted = jit(sum_duplicates_sorted_base, nopython=True, nogil=True, parallel=False)

@jit(nopython=True, nogil=True)
def sum_duplicates(key, data, first_mergesort=False):
    """Alternative implementation of scipy.sparse.coo._sum_duplicates()"""
    idx = lexsort_nb2(key.T[::-1, :], first_mergesort=first_mergesort)
    key, data = key[idx, :], data[idx]
    key, data = sum_duplicates_sorted(key, data)
    return key, data


def sum_duplicates_py(key, data, first_mergesort=False):
    """This is just like sum_duplicates() but a python version.

    The reason for this is to allow the use of numpy's native mergesort for the first key, rather than numba's.
    """

    idx = lexsort_nb2_py(key.T[::-1, :], first_mergesort=first_mergesort)
    key, data = key[idx, :], data[idx]
    key, data = sum_duplicates_sorted(key, data)
    return key, data


def sum_duplicates_mt(key, data=None, threads=None, linear_merge=True, verbose=False):
    """Performs multi-threaded sum_duplicates, by multi-threading the lexsort"""
    if data is None:
        data = np.ones(key.shape[0], dtype=np.int64)

    if key.shape[0] == 0: return key, data

    if threads is None:  threads = utils.get_max_threads()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        splits = utils.split_indices(key.shape[0], threads)
        arg_list = zip(*[(key[i:j, :], data[i:j]) for i, j in splits])
        if verbose: utils.tprint('Starting multi-threaded sum duplicates')
        results = executor.map(sum_duplicates_py, *arg_list)

    if linear_merge:
        if verbose: utils.tprint('(Linear) concatenating sum_duplicates across threads')
        start = time.time()
        key, data = merge(*zip(*results), verbose=verbose)
        key, data = sum_duplicates_sorted(key, data)
        if verbose: utils.tprint('Concatenating time:', time.time() - start)
    else:
        if verbose: utils.tprint('Concatenating sum_duplicates across threads')
        key, data = map(np.concatenate, zip(*results))

        if verbose: utils.tprint('Start final sum_duplicates in sum_duplicates_mt')
        start = time.time()
        key, data = sum_duplicates_py(key, data)
        if verbose: utils.tprint('final sum_duplicates in sum_duplicates_mt time:', time.time() - start)

    return key, data
