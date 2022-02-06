import os
import math
import functools
import shutil
import tempfile
import time
import string
import textwrap
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse

from numba import jit


from mobmess.dummy_module import DummyModule

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    mpl = DummyModule("matplotlib")
    plt = DummyModule("matplotlib")

try:
    import seaborn as sns
except ImportError:
    sns = DummyModule("seaborn")

from mobmess import utils, ani_utils
from mobmess.nb_utils import nb_add, nb_subtract, nb_divide, nb_idx, nb_repeat
from mobmess.plot_basic import *
    


@jit(nopython=True, nogil=True)
def rolling(starts, ends,
            left, right,
            ids,
            key, ki,
            w,
            directions=None):
    """
    Enumerates all pairs of features in a genome

    starts : start[i] is the starting coordinate of feature i

    ends: the end coordinate of features

    left/right : Scan between (starts[left], ends[left]) to (starts[right], ends[right])

    w : minimum genomic length between two features to enumerate

    ids : A mapping of feature i to a numerical ID of its function (e.g. a numerical ID of a COG function)

    key / ki : array to keep track of feature pairs
    """

#    print('internal starts:', starts[left:right])

    # import pdb
    # pdb.set_trace()

    for i in range(left, right):
        # TODO: deal with circularity

        if (key.shape[0] - ki) < 10000:
            key = extend_2d_arr(key)

        s, e = starts[i], ends[i]
        if directions is not None:
            d = directions[i]

        # Update `left` : which is the leftmost gene that could
        # POSSIBLY be within the window distance from the current gene
        # `i`
        for j in range(left, i + 1):
            # (ends[j]>s) => overlaps, so it's in the window
            # ((s-starts[j])<=w) => the end of the previous interval is sufficiently close to the start
            if (ends[j] > s) or ((s - starts[j]) <= w):
                break
        left = j

        # print('i:', i, 's/e:', s, e)
        # print('left:', left, 'left s:', starts[j], 'left e:', ends[j])

        # Iterate through every pair
        for j in range(left, i):

#            print(s, e, d, '|', starts[j], ends[j], directions[j])

            # print(j, (ends[j]>s) or ((s-starts[j])<=w))

            # Even though the leftmost gene (defined by starting
            # coordinate) passed the test, we must test every gene
            # after.  E.g. the leftmost gene might overlap because it
            # is really really long, however, the following genes may
            # not be within the window because they are very short.
#            if (ends[j] > s) or ((s - starts[j]) <= w):

            # -- New addition (2-27-21): don't count accesions on the same exact gene, unless w==0
            if ((ends[j] > s) or ((s - starts[j]) <= w)) and ((w==0) or (ends[j] != s) or (starts[j] != s)):
                ii, jj = ids[i], ids[j]

                if directions is None:
                    # Do this to remember which gene came first (but not their directions)
                    key[ki, :2] = ii, jj

                else:
                    # # Pair_orientation
                    # # - If 0, then both genes are facing in the same direction (forward/forward or reverse/reverse)
                    # # - If 1, then genes are facing outwards away from each other
                    # # - If 2, then genes are facing inwards towards each other
                    # # -----
                    # # - `d` is the direction of the current gene, directions[j] is that of a preceding gene in question
                    # # - If 0, then the direction is forward (coded on the 5'-to-3' direction on this strand, which itself is assumed to be written in 5'-to-3'
                    # # - If 1, then the direction is reverse (coded on the 5'-to-3' on the complement strand)
                    # pair_orientation = np.int32((d + 2 * directions[j]) % 4)
                    # key[ki, :3] = ii, jj, pair_orientation

                    # Pair_orientation
                    # - If 0, then both genes are facing forward
                    # - If 1, then genes are facing outwards away from each other
                    # - If 2, then genes are facing inwards towards each other
                    # - If 3, then both genes are facing reverse
                    # -----
                    # - `d` is the direction of the current gene, directions[j] is that of a preceding gene in question
                    # - If 0, then the direction is forward (coded on the 5'-to-3' direction on this strand, which itself is assumed to be written in 5'-to-3'
                    # - If 1, then the direction is reverse (coded on the 5'-to-3' on the complement strand)
                    pair_orientation = np.int32(2*d + directions[j])
                    key[ki, :3] = ii, jj, pair_orientation

#                    print('in:', ii, jj, pair_orientation)

                # # Old implementation where I didn't care which gene came first, nor their direction
                # if ii < jj:
                #     key[ki, :2] = ii, jj
                # else:
                #     key[ki, :2] = jj, ii

                ki += 1

    return key, ki


@jit(nopython=True, nogil=True)
def rolling_multiple_contigs(contigs,
                             starts,
                             ends,
                             ids,
                             w,
                             do_reduce=False,
                             contigs_groupings=None,
                             mark_contigs=False,
                             directions=None):
    """
    Counts pairs of features across multiple contigs
    
    Assumes that `contigs` is sorted.
    """

    boundaries = get_boundaries2(contigs)

    # print('contigs:', contigs)
    # print('boundaries:', boundaries)
    # print('starts:', starts)
    # print('directions:', directions)

    ncols = 2
    if mark_contigs:
        ncols += 1
    if directions is not None:
        ncols += 1
    key = np.zeros((100000, ncols), ids.dtype)
        
    # if mark_contigs:
    #     key = np.zeros((100000, 3), ids.dtype)
    # else:
    #     key = np.zeros((100000, 2), ids.dtype)
    
    ki = 0

    for bi in range(boundaries.size - 1):
        left, right = boundaries[bi], boundaries[bi + 1]
        prev_ki = ki
        key, ki = rolling(starts, ends, left, right, ids, key, ki, w, directions=directions)

        if mark_contigs:
            if contigs_groupings is None:
                c = contigs[left]
            else:
                c = contigs_groupings[contigs[left]]
            key[prev_ki:ki, -1] = c
    key = key[:ki, :]

    data = np.ones(key.shape[0], np.int64)
    if do_reduce:
        key, data = sum_duplicates(key, data)

    return key, data


def rolling_multiple_contigs_mt(contigs,
                                starts,
                                ends,
                                ids,
                                w,
                                cap_per_contig=False,
                                do_reduce=False,
                                mark_contigs=False,
                                contigs_groupings=None,
                                threads=None,
                                split_multiplier=1,
                                directions=None):
    """Multi-threaded version of `rolling_multiple_contigs`.
    
    do_reduce : If set, then sum the duplicates

    Returns:

    key : 

        An n-by-k ndarray where every row represents a pair of
        genes. If `mark contigs` is False, then k==2, and there are
        just two columns represents the pairings. If `mark_contigs` is
        True, then k==3 and the third column is the contig ID for the
        gene pair. If `contigs_groupings` is not None, then the third
        column are not the contig IDs, but are the contig's grouping
        IDs

    data :
    
        A 1D-array of length n, where each value represents the number
        of instances of the corresponding gene pair in `key`.

    cap_per_contig :
        
        If True, only allow each pair to be counted at most once in
        every contig (or contig grouping, if specified)

    """

    if threads is None:  threads = utils.get_max_threads()

    if split_multiplier is None:
        split_multiplier = 4

    if contigs_groupings is not None:
        assert mark_contigs

    if cap_per_contig:
        assert do_reduce is True
        assert mark_contigs is True

    # if directions is not None:
    #     directions = directions.astype(ids.dtype, copy=False)

    start = time.time()
    splits = utils.split_groups(contigs, threads * split_multiplier)
    threads = min(threads, len(splits))
    utils.tprint('Threads: {}, splits: {}'.format(threads, len(splits)))
    arg_list = zip(*[(contigs[i:j],
                      starts[i:j],
                      ends[i:j],
                      ids[i:j],
                      w,
                      do_reduce,
                      contigs_groupings,
                      mark_contigs,
                      None if (directions is None) else directions[i:j]) for i, j in splits])
    if threads==1:
        results = map(rolling_multiple_contigs, *arg_list)
        key_list, data_list = zip(*results)
    else:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            results = executor.map(rolling_multiple_contigs, *arg_list)
            key_list, data_list = zip(*results)
    utils.tprint('Multi-threaded time:', time.time() - start)
    utils.tprint('Raw pairs after multi-threading:', sum(x.size for x in data_list))

    if do_reduce:
        utils.tprint('Merging duplicates across threads:')
        start = time.time()
        key, data = merge(key_list, data_list, threads=threads, verbose=True)
        key, data = sum_duplicates_sorted(key, data)

        if cap_per_contig:
            ## This implements the functionality of `cap_per_contig`
            ## because it replaces the input `data`, which may have
            ## large numbers representing the many instances of a gene
            ## in the same contig or contig group, with an array of
            ## 1's (effectively, saying that there was only one
            ## instance in the contig or contig group) 
            ##
            ## -- Note that the last column, key[:,-1] is the IDs of
            ## the contigs (or the IDs of contig groups, if
            ## contigs_groupings is not None). Thus, `cap_per_contig`
            ## is by a default a cap of 1 on the instances per contig,
            ## or it is a cap of 1 per contig group if
            ## contigs_groupings is not None.
            
            key, data = sum_duplicates_sorted(key[:, :-1], np.ones(data.size, data.dtype))

        utils.tprint('Merging time:', time.time() - start)
    else:
        key, data = np.vstack(key_list), np.concatenate(data_list)

    utils.tprint('Final pairs:', key.shape[0])
    return key, data

def gene_content_jacc(func, rettype='dense_df'):
    """Compute the jaccard similarity of the gene contents between two contigs"""

    genes_2_contigs, rownames, colnames = utils.sparse_pivot(utils.remove_unused_categories(func), index='accession', columns='contig', rettype='spmatrix')
    colnames = np.asarray(colnames)

    # predicted_plasmids_jacc = enrich.jaccard_from_intersection(genes_2_contigs.T.dot(genes_2_contigs), np.asarray(genes_2_contigs.sum(0)).flatten())

    tmp = (genes_2_contigs > 0).astype(np.uint32)
    jacc = jaccard_from_intersection(tmp.T.dot(tmp), np.asarray(tmp.sum(0)).flatten())

    if rettype=='spmatrix':
        return jacc, colnames
    elif rettype=='dense_df':
        df = pd.DataFrame(jacc.toarray(), colnames, colnames)
        return df

def jaccard_from_intersection(sp, sizes=None, dtype=np.float64, copy=True, **kwargs):
    return norm_intersection(sp, sizes=sizes, dtype=dtype, copy=copy, method='jaccard', **kwargs)

def norm_intersection(sp, sizes=None, method='jaccard', dtype=np.float64, copy=True, rowsizes=None, colsizes=None):
    """Compute a normalized overlap coefficient, such as Jaccard, given
    the intersection and marginal sizes

    """

    # Cast dtype (default to np.float64) to anticipate divisions
    # between integers.  If the dtype is kept as an integer, then
    # floating point divisions will be rounded to 0
    if (dtype != sp.dtype) or copy:
        sp = utils.copy_sp(sp, dtype=dtype)

    assert method in ['jaccard', 'minsize', 'rowsize']

    if (rowsizes is None) and (colsizes is None):
        assert sizes is not None
        assert isinstance(sizes, np.ndarray) and (sizes.ndim==1)

        rowsizes, colsizes = sizes, sizes
    else:
        assert (rowsizes is not None) and (colsizes is not None)

    if scipy.sparse.isspmatrix_coo(sp):
        sp.eliminate_zeros()
        i, j = sp.row, sp.col
            
        if method == 'jaccard':
#            denom = sizes[i] + sizes[j] - sp.data
            denom = rowsizes[i] + colsizes[j] - sp.data
        elif method == 'minsize':
            denom = np.minimum(sizes[i], sizes[j])
        else:
            raise Exception()

        sp.data = nb_divide(sp.data, denom)

    elif scipy.sparse.isspmatrix_csr(sp) or scipy.sparse.isspmatrix_csc(sp):

        if method in ['jaccard', 'minsize']:
            if scipy.sparse.isspmatrix_csc(sp):
                # Flip rowsizes and colsizes (rather than wasting time converting csc to csr matrix)
                rowsizes, colsizes = colsizes, rowsizes

            if method == 'jaccard':
                denom = nb_subtract(nb_add(nb_repeat(rowsizes, nb_subtract(sp.indptr[1:], sp.indptr[:-1])),
                                           nb_idx(colsizes, sp.indices)),
                                    sp.data)
            elif method == 'minsize':
                denom = np.minimum(nb_repeat(rowsizes, nb_subtract(sp.indptr[1:], sp.indptr[:-1])),
                                   nb_idx(colsizes, sp.indices))

        elif method == 'rowsize':
            if scipy.sparse.isspmatrix_csr(sp):
                denom = nb_idx(rowsizes, sp.indices)
            elif scipy.sparse.isspmatrix_csc(sp):
                denom = nb_repeat(rowsizes, nb_subtract(sp.indptr[1:], sp.indptr[:-1]))

        sp.data = nb_divide(sp.data, denom)
    else:
        raise Exception()

        assert method == 'jaccard'

        # This is general code that will work for any sparse matrix
        # type, but it may be slow, so I'm raising an Exception()
        # instead of using this code
        i, j = sp.nonzero()
        data = utils.as_flat(sp[i, j])
        sp[i, j] = data / (sizes[i] + sizes[j] - data)
        sp.eliminate_zeros()

    return sp


def norm(x, marginals, row=True, col=True):
    """
    Normalize a matrix, by dividing the rows and/or columns by the same values

    row : If True, then divide each row by the same value
    
    col : If True, then divide each column by the same value
    """

    if scipy.sparse.isspmatrix(x):
        if row:
            x = utils.sp_nnz_div_1d(x, marginals.reshape(-1, 1))
        if col:
            x = utils.sp_nnz_div_1d(x, marginals.reshape(1, -1), copy=not row)
    else:
        raise Exception()

    return x


@jit(nopython=True, nogil=True)
def argindex(x):
    """
    Takes in a sorted array, e.g. 1, 3, 3, 5, 7, 7, 10, 11, 11, 4, 4, ...

    and returns an indexing, e.g. 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6...

    Duplicate numbers are given the same index
    """

    y = np.empty(x.size, x.dtype)
    y[0] = 0

    # y_slide = y[:-1]
    # x_slide = x[:-1]

    for i in range(1, x.size):
        # y[i] = y_slide[i] + (x[i] != x_slide[i])
        y[i] = y[i - 1] + (x[i] != x[i - 1])

    return y


def test_argindex():
    x = np.array([1, 3, 3, 5, 7, 7, 10, 11, 11, 4, 4])
    y = argindex(x)
    print(y)
    assert (y == np.array([0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6])).all()

def enrich_single_labels(functions, labels, descriptions=None, contigs_groupings=None, pair_ratio=True, ratio=True, labels_order=None):
    """Calculates the marginal frequencies of gene functions across
    different label groups (e.g. plasmid vs. chromosome).

    pair_ratio :

        If True, then calculate the ratio of frequency between two
    label classes. Assumes there are exactly two classes.

    """

    labels_vals = utils.fast_loc(labels, functions['contig'])

    # # Change labels index to categorical, for faster indexing
    # labels.index = pd.Categorical(labels.index.values, categories=functions['contig'].cat.categories).codes
    # labels = labels.sort_index()
    # labels = labels[labels.index != -1]
    # labels_vals = labels.loc[functions['contig'].cat.codes.values].values

    if labels_order is None:
        labels_order = np.sort(np.unique(labels_vals))[::-1]

    # For each label, do separate call to enrich_single(), then concat results

    # print(functions.shape)
    # print(labels_vals.shape)
    # for lab, subfunc in functions.groupby(labels_vals):
    #     display(lab)
    #     display(subfunc)
    #     enrich_single(subfunc, contigs_groupings=contigs_groupings, ratio=ratio).drop(columns='idx')
                

    dic_list = {lab : enrich_single(subfunc, contigs_groupings=contigs_groupings, ratio=ratio).drop(columns='idx') \
                for lab, subfunc in functions.groupby(labels_vals)}
    dic = pd.concat([dic_list[lab].rename(columns=lambda x: '{}_{}'.format(x, lab)) for lab in labels_order], axis=1)
    
    if pair_ratio:
        assert len(labels_order)==2
 
        odds = (dic_list[labels_order[0]] / dic_list[labels_order[1]])
        odds = odds[[c for c in odds if c.endswith('_ratio' )]]
        odds = odds.rename(columns=lambda x: '{}_{}'.format(x.rsplit('_ratio',1)[0], 'odds'))
        dic = pd.concat([dic, odds], axis=1)

    if descriptions is not None:
        dic = descriptions.merge(dic, left_index=True, right_index=True, how='right')

    return dic

def enrich_single(functions, descriptions=None, contigs_groupings=None, ratio=False):
    """
    Calculates marginal statistics about the frequency of function accessions across a set of contigs and contigs groupings

    contigs_groupings :

        pd.Series mapping contig names to their contig group (group names must be castable to integers)    

        of pd.DataFrame mapping contig names to multiple type of contigs groupings

    descriptions :

        Either pd.Series or pd.DataFrame mapping function accessions to their descriptions (and other metadata)
    """

#    contigs, _, _, ids, _ = utils.anvio_functions_2_indices(functions)
    contigs, ids = functions['contig'].cat.codes, functions['accession'].cat.codes
    categories = functions['accession'].cat.categories

    if isinstance(contigs_groupings, pd.DataFrame):
        dic = None
        for group_name in contigs_groupings.columns:
            contigs_groupings_vals = contigs_groupings.loc[functions['contig'].cat.categories, group_name].values.astype(np.int32)
            if dic is None:
                dic = enrich_single_helper(ids, contigs, categories, ratio=ratio,
                                           contigs_groupings=contigs_groupings_vals, group_name=group_name, count_contigs=True)
            else:
                dic_tmp = enrich_single_helper(ids, contigs, categories, ratio=ratio,
                                               contigs_groupings=contigs_groupings_vals, group_name=group_name, count_contigs=False)
                dic = dic.merge(dic_tmp[['accession', group_name] + (['{}_ratio'.format(group_name)] if ratio else [])], on='accession')
    else:
        if contigs_groupings is None:
            group_name = None
        else:
            group_name = contigs_groupings.name
            contigs_groupings = contigs_groupings.loc[functions['contig'].cat.categories].values.astype(np.int32)

        dic = enrich_single_helper(ids, contigs, categories, ratio=ratio, contigs_groupings=contigs_groupings, group_name=group_name)

    dic = dic.set_index('accession')

    if descriptions is not None:
        dic = descriptions.merge(dic, left_index=True, right_index=True, how='right')

    return dic

def enrich_single_helper(ids,
                         contigs,
                         categories,
                         count_contigs=True,
                         descriptions=None,
                         contigs_groupings=None,
                         group_name=None,
                         ratio=False):
    """Helper function to enrich_single() and enrich_pair() to calculate
    marginal statistics/frequencies. NOTE: Assumes more raw formats of
    input parameters than enrich_single() and enrich_pair().

    ids, contigs :

        Integer codes of function accessions and contigs. `ids` are indices into `categories`
    
    categories :

        Names of function accessions.

    contigs_groupings :
    
        np.ndarray where contigs_groupings[i] is the group for contig with index i (i is an integer)

        NOTE: this is a different format than used in enrich_single().
        
    
    ratio :
    
        If True, then normalize the frequency by the total number of
        gene functions, contigs, or contigs groups (Default: False)
    
    Returns
    --------

    dic :

        pd.DataFrame where each row is a function accession

    """

    if group_name is None:
        group_name = 'groups'

    dic = pd.DataFrame({'accession': categories,
                        'idx': np.arange(categories.size)})  # The indices in the matrix
    if descriptions is not None:
        dic['description'] = descriptions[categories].values

    if count_contigs:
        ## Count the number of accession instances, and the number of instances capped at 1 per contig
        tmp_key, tmp_data = sum_duplicates_mt(np.vstack((ids, contigs)).T)
        tmp_ids, tmp_data = sum_duplicates_sorted(tmp_key[:, 0].reshape(-1, 1), tmp_data)
        instances = np.zeros(categories.size, np.int64)
        instances[tmp_ids.flatten()] = tmp_data
        tmp_ids, tmp_data = sum_duplicates_sorted(tmp_key[:, 0].reshape(-1, 1))
        instances_contig_capped = np.zeros(categories.size, np.int64)
        instances_contig_capped[tmp_ids.flatten()] = tmp_data

        # TODO: when enrich_pair() is changed to call this function
        # enrich_single_helper(), make sure to move this line of code
        # setting dic['marginals'] to enrich_pair()
        #dic['marginals'] = np.array(sp.sum(1)).flatten()  # Row sums (should be same as column sums)

        dic['instances'] = instances  # Number of accession instances
        #if ratio: dic['instances_ratio'] = instances_contig_capped / ids.size
        if ratio: dic['instances_ratio'] = instances / ids.size
        dic['contigs'] = instances_contig_capped  # Number of accession instances, capped at 1 per contig
        if ratio: dic['contigs_ratio'] = instances_contig_capped / np.unique(contigs).size

    # Number of unique "contigs groupings" for each accession
    if contigs_groupings is not None:
        tmp_key, tmp_data = sum_duplicates_mt(np.vstack((ids, contigs_groupings[contigs])).T)
        tmp_ids, tmp_data = sum_duplicates_sorted(tmp_key[:, 0].reshape(-1, 1))
        instances_contig_groups_capped = np.zeros(categories.size, np.int64)
        instances_contig_groups_capped[tmp_ids.flatten()] = tmp_data
        dic[group_name] = instances_contig_groups_capped

        if ratio:
            dic['{}_ratio'.format(group_name)] = instances_contig_groups_capped / np.unique(contigs_groupings[contigs]).size

    # else:
        # TODO: This code is only relevant for enrich_pair(). Need to know what to do with it when modifying enrich_pair()

        # TODO : need to figure out what single-variable parameter
        # should be used for calculating hyperz, when
        # cap_per_contig=False or contigs_groupings=None

        # dic[group_name] = dic['contigs'].values

    return dic


def enrich_pair(functions,
                w,
                window_type=None,
                cap_per_contig=None,
                contigs_groupings=None,
                descriptions=None,
                threads=None,
                sort=True,
                overlap=True,
                single_counts=True,
                norm=True,
                norm_kwargs=None,
                symmetric=True,
                split_multiplier=None,
                include_directions=True,
                drop_duplicates=None):
    """Calculates 2-gram statistics.

    fmt :

        If 'anvio', then `functions` is an anvio-formatted table of gene
        annotations.

    window_type :

        If 'genes', then the window is defined as the number of genes
        inbetween. If 'coordinates', then the window is defined as the
        number of nucleotides/amino-acids apart.

    cap_per_contig :

        If True, then limit the number of stuff

    contigs_groupings :

        pd.Series mapping contig names to their contig group (group names must be castable to integers)



    Returns
    ---------

    sp:
    
        scipy sparse matrix. sp[i,j] is the number of times that gene
        j occurs AFTER (in the forward direction) after gene i. If
        symmetric=True, then sp[i,j] is the number of times gene j
        occurs BEFORE OR AFTER gene i

    dic:

        Dataframe where the index is the gene accession. It has a
        column 'idx' which is the integer index for looking up counts
        in `sp`. Also contains marginal statistics, e.g. frequency of
        each gene.
    
    sp_df:
    
        Dictionary containing various normalized values of `sp`

    """

    if cap_per_contig is None:
        cap_per_contig = True

    if drop_duplicates:
        utils.tprint('Dropping duplicates. Rows:', len(functions))
        functions = functions.drop_duplicates(['contig', 'start', 'stop', 'accession'] + (['direction'] if include_directions else []))
        utils.tprint('Done dropping duplicates. Rows:', len(functions))

    if include_directions:
        contigs, starts, ends, ids, directions, _ = utils.anvio_functions_2_indices(functions, sort=sort, directions=True)
    else:
        contigs, starts, ends, ids, _ = utils.anvio_functions_2_indices(functions, sort=sort)
        directions = None
    categories = functions['accession'].cat.categories.values

    utils.tprint('Total features:', ids.size,
                 'Unique features:', np.unique(ids).size,
                 'Unique contigs:', np.unique(contigs).size)

    if window_type is None:
        window_type = 'coordinates'
    assert window_type in ['coordinates', 'genes']
    if window_type == 'genes':
        # Collapse starts to 0,1,2,..., to reflect the gene order
        starts = argindex(starts)
        ends = starts
#        print(starts)

    # For COGs, the ids (category codes) default to np.int16 because
    # there aren't many distinct COGs. Here, make sure that
    # contigs_groupings and ids are the same dtype, because they'll be
    # combined during rolling
    ids = ids.astype(np.int32)

    # Group together similar contigs
    # -- contigs_groupings[i] = group for contig i, as indexed by functions['contig'].cat.categories
    contigs_groupings_orig = contigs_groupings
    if contigs_groupings is not None:
        # If contigs_groupings is specified, then it doesn't make
        # sense to have cap_per_contig=False (if that were the case,
        # then the results would be the same regardless of whether
        # contigs_groupings is set)
        assert cap_per_contig
        contigs_groupings = contigs_groupings.loc[functions['contig'].cat.categories]
        assert (~ contigs_groupings.isna()).all()
        contigs_groupings = contigs_groupings.values.astype(np.int32)
        utils.tprint('Number of contig groups:', np.unique(contigs_groupings).size)
        assert ids.dtype == contigs_groupings.dtype

    mark_contigs = cap_per_contig or (contigs_groupings is not None)

    start_time = time.time()
    key, data = rolling_multiple_contigs_mt(contigs, starts, ends, ids, w,
                                            cap_per_contig=cap_per_contig,
                                            do_reduce=True,
                                            mark_contigs=mark_contigs,
                                            contigs_groupings=contigs_groupings,
                                            threads=threads,
                                            split_multiplier=split_multiplier,
                                            directions=directions)
    utils.tprint('Total counting time:', time.time() - start_time)

    # print(key)
    # print(data)

    sp_df = OrderedDict()
    if include_directions:
#        for d, d_name in [(0, 'parallel'), (1, 'outward'), (2, 'inward')]:
        for d, d_name in [(0, 'forward'), (1, 'outward'), (2, 'inward'), (3, 'reverse')]:
            # All three possible directions (see rolling() for an explanation of this coding)
            mask = key[:,2] == d
            key2 = key[mask, :2]
            sp_df[d_name] = scipy.sparse.coo_matrix((data[mask], (key2[:, 0], key2[:, 1])), shape=(categories.size, categories.size), copy=False).tocsr()
            if symmetric and d_name in ['inward', 'outward']:
                sp_df[d_name] = utils.symmetric_sum(sp_df[d_name])
        
        if symmetric:
            # sp_df['forward'][i,j] counts when gene i comes BEFORE gene j and they are both facing forward
            # sp_df['reverse'][i,j] counts when gene i comes BEFORE gene j and they are both facing reverse
            # - NOTE: 'forward' and 'reverse' are not symmetric matrices, however they are transposes of each other
            forward = utils.sp_mask(sp_df['reverse'], tril=True) + utils.sp_mask(sp_df['forward'], triu=True).T
            reverse = utils.sp_mask(sp_df['forward'], tril=True) + utils.sp_mask(sp_df['reverse'], triu=True).T
            sp_df['forward'], sp_df['reverse'] = forward + reverse.T, reverse + forward.T
        
        key, data = sum_duplicates_mt(key[:,:2], data)

    sp = scipy.sparse.coo_matrix((data, (key[:, 0], key[:, 1])), shape=(categories.size, categories.size), copy=False)
    assert np.all(sp.data != 0)
    utils.tprint('Unique pairs:', sp.nnz)
    if symmetric:
        sp = utils.symmetric_sum(sp)

    dic = pd.DataFrame({'accession': categories,
                        'idx': np.arange(categories.size)})  # The indices in the matrix
    if descriptions is not None:
#        dic['description'] = descriptions[categories].values
        dic['description'] = [descriptions.get(x,x) for x in categories]

    if single_counts:
        ##### TODO: replace code below with a call to enrich_single_helper() funtion

        utils.tprint('Calculating single-variable information and statistics')

        ## Count the number of accession instances, and the number of instances capped at 1 per contig
        tmp_key, tmp_data = sum_duplicates_mt(np.vstack((ids, contigs)).T)
        tmp_ids, tmp_data = sum_duplicates_sorted(tmp_key[:, 0].reshape(-1, 1), tmp_data)
        instances = np.zeros(categories.size, np.int64)
        instances[tmp_ids.flatten()] = tmp_data
        tmp_ids, tmp_data = sum_duplicates_sorted(tmp_key[:, 0].reshape(-1, 1))
        instances_contig_capped = np.zeros(categories.size, np.int64)
        instances_contig_capped[tmp_ids.flatten()] = tmp_data

        dic['marginals'] = np.array(sp.sum(1)).flatten()  # Row sums (should be same as column sums)
        dic['instances'] = instances  # Number of accession instances
        dic['contigs'] = instances_contig_capped  # Number of accession instances, capped at 1 per contig

        # Number of unique "contigs groupings" for each accession
        if contigs_groupings is not None:
            tmp_key, tmp_data = sum_duplicates_mt(np.vstack((ids, contigs_groupings[contigs])).T)
            tmp_ids, tmp_data = sum_duplicates_sorted(tmp_key[:, 0].reshape(-1, 1))
            instances_contig_groups_capped = np.zeros(categories.size, np.int64)
            instances_contig_groups_capped[tmp_ids.flatten()] = tmp_data
            dic['groups'] = instances_contig_groups_capped
        else:
            # TODO : need to figure out what single-variable parameter
            # should be used for calculating hyperz, when
            # cap_per_contig=False or contigs_groupings=None
            dic['groups'] = dic['contigs'].values

    dic.set_index('accession', inplace=True)
    # Add source information to each accession
    if 'source' in functions.columns:
        dic = dic.merge(functions[['accession', 'source']].drop_duplicates().set_index('accession'), left_index=True, right_index=True, how='left')

    if norm:
        if norm_kwargs is None: norm_kwargs = {}
        utils.tprint('Computing normalizations')
        sp_df.update(make_norms(sp, dic, **norm_kwargs))

    #     sp_df = make_norms(sp, dic, **norm_kwargs)
    # else:
    #     sp_df = None

    if overlap:
        utils.tprint('Counting how often two annotations overlap exactly')

        ## New method, window_0
        same, same_dic, same_df = overlap_accessions(functions,
                                            cap_per_contig=cap_per_contig,
                                            contigs_groupings=contigs_groupings_orig,
                                            threads=threads,
                                            split_multiplier=split_multiplier)
        assert np.all(same_dic.index == dic.index)
        sp_df['same'] = same
#        sp_df['count_nr'] = sp_df['count'] - same

#         # Calculate PMI with respect to non-redundant overlaps
# #        sp_df['pmi_instances_nr'] = make_norms(sp_df['count_nr'], dic,
#         sp_df['pmi_instances_nr'] = make_norms(sp_df['count'], dic,
#                                                pmi=True,
#                                                count=False, hyperz=False, jacc=False, cond=False)['pmi_instances']

    utils.tprint('Finished')
    return sp, dic, sp_df


@jit(nopython=True, nogil=True)
def add_base(x, i, y, j):
    x[i] += y[j]


@jit(nopython=True, nogil=True)
def mult_base(x, i, y, j):
    x[i] *= y[j]


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


def get_greedy(key, k, top=True, method=None, symmetric=False):
    """Top indices (i,j)

    method=='representative':

        Returns the top indices (i,j) of the top `k` elements of
        `key`. But it does not repeat the same i's or same j's
        across different elements

    method=='all':

        Returns the top indices (i,j) of the top `k` elements of
        `key`. Doesn't care about repeating i's or j's

    top :

        If top=False, then return the bottom k elements instead.

    """

    if method=='representative':
        keycoo = key.tocoo()

        to_keep_idx = np.argsort(keycoo.data)
        if top:
            to_keep_idx = to_keep_idx[::-1]

        i, j = keycoo.row, keycoo.col
        i_set = set()
        j_set = set()
        i_list, j_list = [], []
        for a in to_keep_idx:
            ia, ja = i[a], j[a]
            if (ia not in i_set) and (ja not in j_set):
                i_set.add(ia)
                j_set.add(ja)
                i_list.append(ia)
                j_list.append(ja)
                if len(i_set)==k:
                    break
        i, j = np.array(i_list), np.array(j_list)            

        idx = np.argsort(np.array(key.tocsr()[i,j]).flatten())
        i, j = i[idx], j[idx]
        if symmetric:
            i,j = i[i<=j], j[i<=j]
        return i, j

    elif method=='all':

        keycoo = key.tocoo()
        data, i, j = keycoo.data, keycoo.row, keycoo.col

        if top:
            to_keep_idx = np.argpartition(data, data.size - k)[-k:]
        else:
            to_keep_idx = np.argpartition(data, k)[:k]
        i, j = i[to_keep_idx], j[to_keep_idx]

        return i, j
    else:
        raise Exception()

def topk(sp,
         k,
         key=None,
         dic=None,
         row_include=None,         
         row_blackout=None,
         col_include=None,
         col_blackout=None,
         other_keynames=None,
         rettype='long',
         hist_kwargs=None,
         symmetric=False,
         tail=True,
#         greedy=None,
         representative=False,
         verbose=None):
    """Keep only the top k entries of a sparse matrix sp. Zero the other entries
    
    rettype : if 'long', then convert this into a long format dataframe
    
    symmetric :

        If True, then assume key values for (i,j) and (j,i) are the
        same. In this case, just show (i,j) pairs where i<j.

    tail :

        If True, also show the bottom k pairs. The top and bottom pairs will be
        concatenated in a single dataframe, with a buffer of 5 rows of 0's
        between

    representative : 

        If True, return the top indices (i,j) of the top `k` elements
        of `key`. But it does not repeat the same i's or same j's
        across different elements

    """
    
    if verbose is None:
        verbose = False

    # Convert `sp` into a dictionary
    if hasattr(sp, 'keys'):
        sp_dict = sp.copy()
    else:
        if isinstance(sp, (list, tuple)):
            sp_dict = OrderedDict([('%s_%s' % ('data', i), x) for i, x in enumerate(sp)])
            singleton = False
        else:
            sp_dict = {'data': sp}
            singleton = True

    # key is not specified, so assume it is one of the matrices
    if key is None:
        keyname = list(sp_dict.keys())[0]
        print('Assuming key is %s' % keyname)
        key = sp_dict[keyname]

    # `key` is a key in the dictionary of matrices
    elif isinstance(key, str) and key in sp_dict.keys():
        keyname = key
        key = sp_dict[keyname]

    # `key` is a matrix itself
    else:
        found = False
        for sp_key, v in sp_dict.items():
            if key is v:
                keyname = sp_key
                found = True
                break
        if not found:
            keyname = 'key'
            sp_dict['key'] = key

    if hist_kwargs is None: hist_kwargs = {}
    #utils.plot_hist(key.tocoo().data, log=True, xlabel='key', ylabel='frequency', show=True, **hist_kwargs)
    histplot(key.tocoo().data, log=True, xlabel='key', ylabel='frequency', show=True, **hist_kwargs)

    # Get indices of the top `k` values
    if not scipy.sparse.isspmatrix(key):
        assert isinstance(key, pd.DataFrame)
        if row_blackout is not None:
            blackout = key.index.isin(row_blackout)
        key = key.tocoo()

    def include_2_blackout(include):
        """Convert row_include/col_include to row_blackout/col_blackout"""

        try:
            # Try converting `include` to an integer array.
            # If successful, then assume `include` are integers
            np.array(include, dtype=int)

        except:
            # If failure, then assume `include` are strings that are keys into `dic`
            if isinstance(include, str):
                include = [include]

            if callable(include):
                include = np.where([include(x) for x in dic.index])[0]
            else:
                include = np.where(dic.index.isin(include))[0]

        blackout = np.setdiff1d(np.arange(key.shape[0]), include)
        return blackout

    if row_include is not None:
        assert row_blackout is None
        row_blackout = include_2_blackout(row_include)
    if col_include is not None:
        assert col_blackout is None
        col_blackout = include_2_blackout(col_include)

    if (row_blackout is not None) or (col_blackout is not None):
        if row_blackout is not None:
            utils.tprint('Masking rows', verbose=verbose)
            key = utils.sp_mask_rows(key, row_blackout)
            print('nnz:', key.nnz)
        if col_blackout is not None:
            utils.tprint('Masking columns', verbose=verbose)
            key = utils.sp_mask_cols(key, col_blackout)
            print('nnz:', key.nnz)
        key = key.tocoo()

    if key.nnz == 0:
        raise Exception("No non-zero entries")

    if symmetric:
        k = 2 * k
    print('k:', k, 'data.size:', key.data.size)
    k = min(k, key.data.size)

    print('nnz:', key.nonzero()[0].size)

    utils.tprint('Distribution of key values after masking')
    utils.plot_hist(key.tocoo().data, log=True, xlabel='key', ylabel='frequency', show=True, **hist_kwargs)


#    i, j = get_greedy(key, k, method='representative' if representative else 'all', top=True, symmetric=symmetric)
    i, j = get_greedy(key, k, method='representative' if representative else 'all', top=True)

    if tail:
        assert Exception("Need to double check this code. Also, I need to define the parameter `greedy`")

        if greedy:
            #i_tail, j_tail = get_greedy(key, k, method='A', top=False, symmetric=symmetric)
            i_tail, j_tail = get_greedy(key, k, method='A', top=False)
        else:
            to_keep_idx = np.argpartition(key.data, k)[:k]
            i_tail, j_tail = key.row[to_keep_idx], key.col[to_keep_idx]        
            # if symmetric:
            #     i_tail, j_tail = i_tail[i_tail<=j_tail], j_tail[i_tail<=j_tail]
        
    if rettype == 'long':
        # display(sp_dict)
        # display(dic)
        ret = utils.wide_2_long_df({k: utils.sp_mask(sp, i=i, j=j) for k, sp in sp_dict.items()},
                                   dic=dic.reset_index())
        
        if other_keynames is None:
            other_keynames = []
        elif isinstance(other_keynames, str):
            other_keynames = [other_keynames]

        ret = ret.sort_values([keyname] + other_keynames, ascending=False)

        if tail:
            ret_tail = utils.wide_2_long_df({k: utils.sp_mask(sp, i=i_tail, j=j_tail) for k, sp in sp_dict.items()},
                                            dic=dic.reset_index())
            ret_tail = ret_tail.sort_values([keyname] + other_keynames, ascending=True)

            ret_tail = ret_tail[ret.columns]

            empty = pd.DataFrame(0, columns=ret.columns, index=np.arange(5))

            ret = pd.concat([ret, empty, ret_tail], axis=0, sort=False, ignore_index=True)

        return ret
    else:
        ret = [utils.sp_mask(sp, i=i, j=j) for sp in sp_dict.values]
        if singleton:
            return ret[0]
        else:
            return ret


def make_norms(sp,
               dic,
               count=True,
               hyperz=True,
               jacc=True,
               cond=True,
               pmi=True):
    """Do normalizations of a count matrix"""

    start = time.time()

    # Create a csr matrix in anticipation that it will make most
    # calculations faster when calling norm()-->utils.sp_nnz_div_1d()
    sp = sp.tocsr()
    sp_float64 = utils.copy_sp(sp, np.float64)

    if isinstance(dic, pd.DataFrame):
        dic = {k: v.values for k, v in dic.items()}

    ret = {}
    if count:
        ret['count'] = sp
    if hyperz:
        ret['hyperz_marg'] = norm(sp_float64, dic['marginals'])
        ret['hyperz_groups'] = norm(sp_float64, dic['groups'])
    if jacc:
        ret['jacc_marg'] = jaccard_from_intersection(sp_float64, dic['marginals'])
        ret['jacc_groups'] = jaccard_from_intersection(sp_float64, dic['groups'])
    if cond:
        # Conditional probability: Prob(col j | row i)
        ret['cond_marg'] = norm(sp_float64, dic['marginals'], row=True, col=False)
        ret['cond_groups'] = norm(sp_float64, dic['groups'], row=True, col=False)
        ret['cond_contigs'] = norm(sp_float64, dic['contigs'], row=True, col=False)
    if pmi:
#        ret['pmi_instances'] = norm(sp_float64, dic['instances'], row=True, col=True)
        ret['pmi_instances'] = norm(sp_float64, dic['instances'], row=True, col=True) * sp_float64.sum()
        ret['pmi_instances'].data = np.log(ret['pmi_instances'].data)
    # if pmi:
    #     ret['pmi_nr__instances'] = norm(sp_float64, dic['instances'], row=True, col=True)
    #     ret['pmi_nr_instances'] = norm(sp_float64, dic['instances'], row=True, col=True) * sp_float64.sum()
    #     ret['pmi_nr_instances'].data = np.log(ret['pmi_instances'].data)

    utils.tprint('Normalize time:', time.time() - start)

    return ret


def get_instances(functions, a, b, w):
    """Get instances of two genes that occur within a window distance from each other"""

    # Filter for contigs that contain both `a` and `b`
    contigs_a = functions.loc[functions['accession'] == a, 'contig']
    contigs_b = functions.loc[functions['accession'] == b, 'contig']
    shared_contigs = np.intersect1d(contigs_a.unique(), contigs_b.unique())
    functions = functions[functions['contig'].isin(shared_contigs)]

    # Take a reduced set of annotations over a or b
    functions_minimal = functions[functions['accession'].isin((a, b))]

    # Calculate all pairings of a and b
    contigs, starts, ends, ids, sort_idx = utils.anvio_functions_2_indices(functions_minimal)
    # Instead of using the gene function ids, return the index position (by inputting np.arange(start))
    # pairs, _ = rolling_multiple_contigs(contigs, starts, ends, np.arange(starts.size), w)

    pairs, _ = rolling_multiple_contigs_mt(contigs, starts, ends, np.arange(starts.size), w,
                                           cap_per_contig=True,
                                           do_reduce=True,
                                           mark_contigs=True,
                                           contigs_groupings=None)

    if a != b:
        # Filter for pairs that involve both genes (as opposed to
        # pairing of `a` to itself, or `b` to itself)
        pairs_ids = ids[pairs]

        # display(pairs_ids)
        # pairs_orig = pairs

        pairs = pairs[pairs_ids[:, 0] != pairs_ids[:, 1], :]
        # display(pairs)

    # Get instances of the genes that are paired with the other gene
    instances = functions_minimal.iloc[sort_idx[np.unique(pairs.flatten())]]

    return instances

def draw_arrow(ax, start, end, strand, ypos, height, small_relative,
               facecolor=None, alpha=None, linewidth=None, edgecolor=None):
    """
    Modified from coolbox. 

    https://github.com/GangCaoLab/CoolBox/blob/619db912ac09d6ce423c469d63627b19516edbe7/coolbox/core/track/bed/plot.py
    """
    
    # This ensures that we have a proper arrow
    small_relative = min(small_relative, end - start)

    half_height = height / 2
    if strand == '+':
        x0 = start
        x1 = end - small_relative
        y0 = ypos
        y1 = ypos + height

        # x1 = max(x0, x1)

        """
        The vertices correspond to 5 points along the path of a form like the following,
        starting in the lower left corner and progressing in a clock wise manner.
        -----------------\
        ---------------- /
        """

        vertices = [(x0, y0), (x0, y1), (x1, y1), (x1 + small_relative, y0 + half_height), (x1, y0)]

    else:
        x0 = start + small_relative
        x1 = end
        y0 = ypos
        y1 = ypos + height

        """
        The vertices correspond to 5 points along the path of a form like the following,
        starting in the lower left corner and progressing in a clock wise manner.
        /-----------------
        \\-----------------
        """

        vertices = [(x0, y0), (x0 - small_relative, y0 + half_height), (x0, y1), (x1, y1), (x1, y0)]
    
    if linewidth is None:
        linewidth = 1
    if edgecolor is None:
        edgecolor = 'dimgrey'

    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(vertices, closed=True, fill=(facecolor is not None),
                         edgecolor=edgecolor, linewidth=linewidth, facecolor=facecolor, alpha=alpha))

def plot_contig(gene_calls, length=None, functions=None, ax=None, tier=None,
                colors=None, fontsize=None, show=False, rotation=None, full_xlim=False):
    """
    Vanilla matplotlib plotting of genes in a contig
    """

    if tier is None:
        tier = False

    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(16, 1.5))
        ax = plt.gca()

    gene_calls = gene_calls.copy()

    if colors is None:
        colors = pd.Series(sns.color_palette(n_colors=gene_calls['accession'].nunique()), np.asarray(gene_calls['accession'].unique()))

    if not (isinstance(colors, str) and (colors in gene_calls.columns)):
        # Create a new column which is the color
        gene_calls['color'] = colors.loc[gene_calls['accession'].values].values
        colors = 'color'

    label_column = 'accession_label' if 'accession_label' in gene_calls.columns else 'accession'

    # Plot genes one-by-one from left to right. Keep a tally of number of "open" genes that haven't finished plotting before the start of the current genes.
    # -- If tier==True, then overlapping genes are juxtaposed at different levels, otherwise they are stacked on top of each other.
    open_genes = set([])
    max_levels = 0
    for idx, (_, start, stop, strand, accession, color) in \
        enumerate(gene_calls[['start', 'stop', 'direction', label_column, colors]].to_records()):

        # Draw a line border around arrows
        if 'linewidth' in gene_calls.columns:
            linewidth = gene_calls.iloc[idx]['linewidth']
        else:
            linewidth = None

        if 'edgecolor' in gene_calls.columns:
            edgecolor = gene_calls.iloc[idx]['edgecolor']
        else:
            edgecolor = None

        open_genes = open_genes - set([x for x in open_genes if x[1] <= start])
        max_levels = max(max_levels, len(open_genes))
        y = max_levels if tier else 0

        draw_arrow(ax, start, stop, '+' if (strand=='f') else '-', y, 1, 100, facecolor=color, alpha=0.75,
                   linewidth=linewidth, edgecolor=edgecolor)

        ax.text((start + stop) / 2, y + 0.5, str(accession), fontsize=fontsize,
                horizontalalignment='center', verticalalignment='center', rotation=rotation)

        open_genes.add((start, stop))

    xmin, xmax = gene_calls['start'].min(), gene_calls['stop'].max()
    buff = (xmax - xmin) * 0.1
    xmin, xmax = max(0, xmin - buff), xmax + buff
    if length is not None:
        xmax = min(length, xmax)
    if full_xlim:
        xmin = 0
        if length is not None:
            xmax = length    
    ax.set_xlim(xmin, xmax)

    ax.set_ylim(-0.1, (max_levels if tier else 0) + 1.1)
    ax.grid(b=False, which='both', axis='y')
    ax.set_yticklabels([])

    # Turn on minor ticks
#    ax.tick_params(which="x", bottom=True, pad=0)
    ax.tick_params(axis="x", bottom=True, pad=0)
    from matplotlib.ticker import AutoMinorLocator
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    if show:
        plt.show()

# def examine_gene_pair(functions, a, b, same_dic, same, descriptions, w, slop, contigs, **kwargs):
def examine_gene_pair(functions, a, b, descriptions, w, slop, contigs, **kwargs):
    """Calls plot_gene_pair(), but first prints out a small bit of useful info"""

    # ai, bi = utils.make_dict(same_dic['accession']).loc[[a,b]]
    # ai, bi = same_dic.loc[[a,b], 'idx']
    # print('Number of times a and b overlap exactly:', same[ai,bi])

    return plot_gene_pair(functions, a, b, w, slop, contigs, descriptions=descriptions, **kwargs)


def visualize_alignment(
        functions,
        genes=None,
        gene_calls=None,
        w=None,
        neighborhood=None,
        contigs=None,
        contig_lengths=None,
        contigs_metadata=None,
        contig_descriptions=None,
        merge_overlapping=True,
        black_genes=True,
        gene_descriptions=None,
        fasta=None,
        sources=None,
        sources_display=None,
        sources_anchor=None,
        figsize=None,
        dist=None,
        aln_blocks=None,
        aln_blocks_is_1_based=None,
        aln_blocks_height=None,
        representatives=None,
        max_loci=None,
        assignments=None,
        center=True,
        show=None,
        verbose=True,
        width=None,
        most_common_gene=None,
        output=None,
        cov_dict=None,
        cov_keys=None,
        color_singletons=False,
        rotate=False,
        rotate_deltas=None,
        contigs_order=None,
        svg=None,
        anchor_direction='f',
        renumber_accessions=True,
        write_loci=None,
        threads=None,
        **kwargs):
    """
    Visualize alignment and shared gene content of multiple contigs

    functions : gene functions annotation table (anvio format)

    genes : list of gene functions/families to anchor the visualization

    w : minimum distance between the genes
    
    neighborhood : show the region that is `neighborhood` nucleotides upstream and downstream of instances

    bed_height : visualization track height of each instance

    Mapping coverage
    ----------------
    
    cov_dict :

    cov_keys : 


    Output
    ------

    output :

    """

    ###########################################
    # Read table of gene function annotations #
    ###########################################
    
    # Read gene functions table
    functions = utils.merge_gene_calls_and_functions(functions, gene_calls, verbose=True)
#    functions = utils.merge_gene_calls_and_functions(functions, gene_calls, add_dummy_annotations=True, verbose=True)
    print('functions columns:', list(functions.columns))
    if any(c not in functions.columns for c in ['contig', 'start', 'stop', 'direction', 'accession', 'source']):
        raise Exception("The columns 'contig', 'start', 'stop', 'direction', 'accession', 'source' need to be in your function annotations table")
    dtype_mapping = {'contig':'category', 'accession':'category', 'source':'category', 'direction':'category', 'description':'category', 'function':'category'}
    dtype_mapping = {k : v for k, v in dtype_mapping.items() if k in functions.columns}
    functions = functions.astype(dtype_mapping)
    
    # Filter table to a subset of contigs
    if contigs is not None:
        print('Input contigs:', len(contigs))
        functions = utils.subset(functions, contig=contigs)
        print('Contigs in functions:', functions['contig'].nunique())

    # Get list of all annotation sources used
    if sources is not None:
        # Filter table to specified sources
        functions = utils.subset(functions, source=sources)
        sources = [s for s in sources if s in functions['source'].cat.categories]
    else:
        #sources = list(functions['source'].unique())
        sources = functions['source'].cat.categories
        
    # This is the set of gene accession sources that will be used to
    # identify anchor genes, as well as to collapse multiple
    # overlapping accessions to a representative accession
    if sources_anchor is None:
        sources_anchor = sources
    sources_anchor = [s for s in sources_anchor if s in sources]

    # List of gene function sources to be annotated with numbers
    # 0,1,2... -- The ordering of sources is used to tiebreak gene
    # functions that have the same priority to be assigned a low
    # number, e.g. 0
    sources_numbered = sources_anchor + sorted([x for x in sources if x not in sources_anchor])

    # List of gene function sources to be displayed in output PDF file
    if sources_display is None:
        sources_display = sources_numbered
    sources_display = [s for s in sources_display if s in sources_display]

    # If gene_descriptions are not specified, then get them from the functions table
    if gene_descriptions is None:
        if 'description' in functions.columns:
            gene_descriptions = functions.set_index('accession')['description'].dropna().drop_duplicates()
        elif 'function' in functions.columns:
            gene_descriptions = functions.set_index('accession')['function'].dropna().drop_duplicates()
        else:
            # Can't find any description in the functions table. So just do a dummy mapping of a gene family name to itself.
            gene_descriptions = {x : x for x in functions['accession'].drop_duplicates()}
    gene_descriptions = pd.Series(gene_descriptions)
    
    # print(sources)
    # print(sources_anchor)
    # print(sources_numbered)
    # print(sources_display)

    ##############################################
    # Identify genes to anchor the visualization #
    ##############################################

    def get_anchor_genes(functions, cover_target, priority_method, sources, start_genes=None, contig_list=None):
        # Iteratively identify the most common gene as the next anchor. Iterate until enough anchors to cover all contigs
        genes = []

        remaining_functions = utils.subset(functions, source=sources)

        if contig_list is None:
            contig_list = list(remaining_functions['contig'].unique())
        contigs_idx = pd.Series({b:a for a, b in enumerate(contig_list)})
        remaining_functions['contig_idx'] = contigs_idx.loc[remaining_functions['contig'].values].values

        while len(remaining_functions) > 0:

            # Sort gene families by (1) the number of contigs they occur in, (2) their position in `sources`, and (3) their leftmost position
            accession_df = remaining_functions.groupby('accession', observed=True).agg(
                ncontigs=pd.NamedAgg('contig', 'nunique'),
                instances=pd.NamedAgg('contig', 'size'),
                sources_idx=pd.NamedAgg('source', lambda x:len(sources) - sources.index(x.iloc[0])),
                contig_idx=pd.NamedAgg('contig_idx', 'min'),
                start=pd.NamedAgg('start', 'min'))

            if (start_genes is None) or (len(start_genes)==0):
                if priority_method=='anchor':
                    if accession_df['ncontigs'].max() > 1:
                        accession_df = accession_df.nlargest(1, ['ncontigs', 'instances', 'sources_idx'], keep='all').nsmallest(1, 'start').iloc[[0]]
                    else:
                        accession_df = accession_df.nsmallest(1, ['contig_idx'], keep='all').nsmallest(1, 'start').iloc[[0]]

                    next_gene = accession_df.index[0]
                else:
                    raise Exception()
            else:
                # First use a predefined list of genes
                next_gene = start_genes[0]
                start_genes = start_genes[1:]
                accession_df = accession_df.loc[[next_gene]]
                
            genes.append(accession_df)

            if cover_target=='contig':
                # Remove all contigs that contain this accession
                covered_contigs = np.asarray(remaining_functions.loc[remaining_functions['accession']==next_gene, 'contig'].unique())
                remaining_functions = remaining_functions[~ utils.catin(remaining_functions['contig'], covered_contigs)]
            elif cover_target=='gene':
                # Remove all genes that are annotated to this accession
                tmp = remaining_functions.merge(remaining_functions.loc[remaining_functions['accession']==next_gene, ['contig', 'start', 'stop']].drop_duplicates(),
                                                on=['contig', 'start', 'stop'], how='left', indicator=True)
                remaining_functions = tmp[tmp['_merge']=='left_only'].drop(columns=['_merge'])
            else:
                raise Exception()

        # Return a dataframe, where the index is the genes in order of selection, and the columns are ['ncontigs', 'instances', 'sources_idx', 'start']
        genes = pd.concat(genes)

        return genes

    if genes is None:
        genes = []
    elif isinstance(genes, str):
        genes = [ genes ]

    if (most_common_gene is True):
        utils.tprint(f'Input {len(genes)} anchor genes:', verbose=verbose)
        covered_contigs = np.array(utils.subset(functions, accession=genes)['contig'].unique())
        remaining_functions = utils.subset(functions, contig=covered_contigs, invert=True)
        new_anchor_genes = list(get_anchor_genes(remaining_functions, 'contig', 'anchor', sources_anchor).index)
        utils.tprint(f'Adding {len(new_anchor_genes)} anchors genes:', verbose=verbose)
        genes = list(genes) + new_anchor_genes

    print('Anchor gene families:', list(genes))

    # Get subset of annotations that contain the anchor genes
    if len(genes)==0:
        pass
    elif w is None:
        # If w (the distance between anchors) is unset, then this is an easy calculation. 
        instances = functions[functions['accession'].isin(genes)].copy()
    else:
        if len(genes)==2:
            # Instances of gene pairs that occur within a window distance from each other.
            # `instances` : functions annotation dataframe where each row is a gene
            instances = get_instances(functions, genes[0], genes[1], w)
        else:        
            raise Exception()

    if fasta is not None:
        fasta = utils.get_fasta_dict(fasta)

    if contig_lengths is None:
        assert fasta is not None, "If you don't specify contig_lengths, you must specify the fasta sequences"
        contig_lengths = {k : len(v) for k, v in fasta.items()}
    contig_lengths = pd.Series(contig_lengths)

    # if 'length' in contigs.columns:
    #     contig_lengths = contigs['length']

    #######################
    # Get loci to be plot #
    #######################
    #
    # -- if `neighborhood` is specified, then plot only the neighborhood of `neighborhood` upstream and downstream around the anchor genes
    #    -- otherwise, plot the entirety of every contig

    if len(genes)==0:
        instances = None
        contigs_encompassing = functions['contig'].unique()
        genes_encompassing = functions.copy()
        loci = pd.DataFrame({'contig' : contigs_encompassing,
                             'start' : 0,
                             'stop' : contig_lengths.loc[contigs_encompassing].values})

    else:
        if 'e_value' not in instances.columns:
            instances['e_value'] = 0.

        # The set of contigs that contain these gene pair instances
        contigs_encompassing = instances['contig'].unique()

        # Functions dataframe containing all other genes in these contigs
        genes_encompassing = utils.subset(functions, contig=contigs_encompassing).reset_index(drop=True)
        if verbose: print('Total loci:', instances.shape[0])

        if (neighborhood is None) or (neighborhood<=0):
            # Set default neighborhood to be very high (10 Mb)
            neighborhood = int(1e7)
        print('neighborhood:', neighborhood)

#        return instances, neighborhood, contig_lengths

        # slop and then merge overlapping loci (might have been created by slopping)
        loci = slop(instances, neighborhood, sizes=contig_lengths)
        loci = interval_merge_self(loci)

        if verbose: print('Loci after merging:', loci.shape[0])

    utils.tprint(f'Contigs to be plotted: {contigs_encompassing.size}', verbose=verbose)
    
    # no_assignments = assignments=='identity'

    # Print info about the number of unique contigs groups (or taxonomy if no group assignments were specified)
    # if assignments is None:
    #     # Lookup taxonomic assignments in `contigs`
    #     assignments = contigs[utils.ordered_ranks[::-1]]
    # elif assignments=='identity':
    #     # Assign each contig to its own group
    #     unique_contigs = np.asarray(functions['contig'].unique())
    #     assignments = pd.Series(np.arange(len(unique_contigs)), unique_contigs).to_frame('group')
    # else:
    #     raise Exception()

    if assignments is None:
        # Assign each contig to its own group
        unique_contigs = np.asarray(functions['contig'].unique())
        assignments = pd.Series(np.arange(len(unique_contigs)), unique_contigs).to_frame('group')
        no_assignments = True

    if not no_assignments:
        print('Unique contigs groupings:')
        tmp = assignments.loc[contigs_encompassing].apply(lambda x: x.dropna().unique().size, axis=0)    
        display(tmp[tmp > 0])

    ##################################################################################
    # Handle the situations where a contig has the anchor gene in two different loci #
    ##################################################################################

    utils.tprint('Handling loci that may have multiple instances of the anchor genes(s)', verbose=verbose)

    loci['original_contig'] = loci['contig']
    contig_counts = loci['contig'].value_counts()
    to_concat = []
    new_assignments = []
    assignments = assignments.rename_axis('contig')
    for c in contig_counts[contig_counts >= 2].index:
        new_names = ['{}_locus_{}'.format(c, i) for i in range(contig_counts[c])]

        # Change each instance of a contig in `loci` from <contig_name> to <contig_name>_locus_<i>
        loci.loc[loci['contig']==c, 'contig'] = new_names

        # To reflect changes in `loci`, create copies of gene annotations in `genes_encompassing`
        this_contigs_genes = genes_encompassing[genes_encompassing['contig']==c]
        to_concat.extend([this_contigs_genes.copy().assign(contig=x) for x in new_names])
        
        # Create copies in `assignments`
        tmp = assignments.loc[np.repeat(c, contig_counts[c])].reset_index().assign(contig=new_names)
        new_assignments.append(tmp)

    to_concat.append(genes_encompassing[utils.categorical_isin(genes_encompassing['contig'], contig_counts[contig_counts < 2].index)])
    genes_encompassing = utils.better_pd_concat(to_concat, ignore_index=True)
    genes_encompassing['original_contig'] = [x.split('_locus_')[0] for x in genes_encompassing['contig']]
    assignments = pd.concat([assignments.reset_index()] + new_assignments).set_index('contig')

    #############################################
    # Potentially take a subset of loci to show #
    #############################################

    if representatives:
        # Show just one region for each species
        assert representatives in assignments.columns
        tmp = loci.copy()
        tmp['group'] = assignments.loc[loci['original_contig'], representatives].values
        print('Unique contig groups:', tmp['group'].unique().size)
        loci = tmp.groupby('group', group_keys=False).apply(lambda x: x.sample(n=1, random_state=123))

    if max_loci is not None and (max_loci < len(loci)):
        if verbose: print('Randomly sampling %s out of %s loci' % (max_loci, len(loci)))
        loci = loci.sample(n=max_loci, random_state=123)

    # Take just the portion that is needed, by doing a Bedtools intersection
    utils.tprint('Intersecting loci', verbose=verbose)
    intersect_bed = interval_merge(genes_encompassing, loci, suffix='loci')
    loci_genes = utils.unused_cat(intersect_bed[[c for c in intersect_bed if not c.endswith('_loci')]])
    contigs_encompassing = np.asarray(loci_genes['contig'].unique())
    
    #################################
    # Plot distance between contigs #
    #################################

    if dist=='jacc':
        # Reorder loci based on jaccard similarity of gene content
        utils.tprint('Calculating gene content jaccard', verbose=verbose)
        dist = gene_content_jacc(utils.unused_cat(loci_genes))
        dist_xlabel = 'Similarity in gene annotations (Jaccard index)'

    if contigs_order is not None:
        contigs_order = pd.Series(np.arange(len(contigs_order)), contigs_order).to_frame('order')

    # Plot heatmap of contig-by-contig ANI
    if dist is not None:

        if len(contigs_encompassing) > 1:
            assert contigs_order is None, "Cannot specify an ordering of contigs but also compute an ordering from a dendrogram"

            # Try converting dataframe from sparse to dense
            sub_dist = dist.loc[contigs_encompassing, contigs_encompassing]
            try:
                sub_dist = sub_dist.sparse.to_dense()
            except:
                pass

            utils.tprint('Plotting histogram', verbose=verbose)
            histplot(utils.triu_take(sub_dist.values), bins=np.linspace(0,1,21),
                     xlabel=dist_xlabel, ylabel='Number of contig pairs', title="Contig similarities", show=show)            
            fig_dist = plt.gcf()

            # Note that the `assignments` parameter in kwargs might be handy to use
            utils.tprint('Plotting heatmap', verbose=verbose)
            _, fig_heatmap, Z, _ = clustermap(sub_dist, figsize=figsize,
                                              show=show, number_labels=True, square=True, optimal_ordering=True, **kwargs)
            fig_heatmap.fig.suptitle(dist_xlabel)

            contigs_order = pd.Series(np.arange(sub_dist.shape[0]), sub_dist.index[scipy.cluster.hierarchy.leaves_list(Z)]).to_frame('order')
        else:
            if show: print('Only one contig, so not showing heatmap')
            fig_heatmap, fig_dist = None, None
    else:
        fig_heatmap, fig_dist = None, None

    if contigs_order is not None:
        # Reorder loci according to the leaves ordering in the heatmap dendrogram
        loci = loci.merge(contigs_order, left_on='contig', right_index=True, how='left').sort_values('order').drop(columns=['order'])
        
    ######################
    # Process gene names #
    ######################

    # (1) Rename genes from their accessions to the descriptions of those accessions
    # (2) Replace all whitespace with '_'
    def rename(x):
        if (gene_descriptions is not None):
            x2 = '.'.join(x.split('.')[:-1]) if (('.' in x) and (x not in gene_descriptions.index)) else x
            if (x2 in gene_descriptions.index) and (x2 != gene_descriptions[x2]):
                x = x + ':' + gene_descriptions[x2]
        return x.replace(' ', '_')

    # Sort the accessions based on a pre-determined ranking of which accessions are more important to show versus others
    # -- Show (1) the genes of interest, (2) COGs/Pfam, and (3) mmseqs
    def sort_and_show(x, frequency=None, anchors=None):
        x = x.sort_values('order', ascending=False)
        x_labeled = x.dropna(subset=['order'])

        ret = {'accession_representative' : x['accession_representative'].iloc[0],
               'ncontigs' : x['ncontigs'].iloc[0],
               'naccessions' : len(x)}

        if anchors is not None:
            ret['accession_label'] = '|'.join(x_labeled['accession'])

        for source in sources:
            ret[source] = tuple([rename(y) for y, s in x[['accession_representative', 'source']].to_records(index=False) if s==source])

        ret = pd.Series(ret).to_frame().T
        return ret

    def concat_accession_instances(x):
        """Merges all instances of an accession (e.g. "A" or "1") by
        creating a list of all contigs, COGs, Pfams, mmseqs,
        identical, etc. annotations that the accession occurs in

        """

        # Record the accession_representative (i.e. the representative gene function)
        ret = {'accession_representative' : x.iloc[0]['accession_representative']}

        for c in sources:
            ret[c] = tuple(pd.Series(np.concatenate(x[c].tolist())).drop_duplicates().tolist())
        for c in ['ncontigs','ncontigs_gain','instances','order']:
            ret[c] = int(x[c].iloc[0])

        return pd.Series(ret).to_frame().T

    # this ensures dropping duplicates where the same gene gets annotated to the same accession more than once
    loci_genes = loci_genes.drop_duplicates(subset=['contig','start','stop','accession'])

    loci_genes['ncontigs'] = loci_genes.drop_duplicates(['contig', 'accession'])['accession'].value_counts().loc[loci_genes['accession'].values].values
    loci_genes['instances'] = loci_genes['accession'].value_counts().loc[loci_genes['accession'].values].values

    alphabet = list(string.ascii_uppercase) # Set alphabet to ['A', 'B', 'C', etc.]
    if len(genes) > len(alphabet):
        # There are more than 26 anchor genes. So, need to create an extended alphabet of AA, AB, AC, ... AAA ... to ZZZ
        import itertools
        alphabet = [''.join(x) for x in itertools.chain.from_iterable(itertools.combinations(alphabet, r) for r in range(1, 4))]
        assert len(alphabet) >= len(genes)        

    # Concatenate the descriptions of genes that lie exactly on
    # top each other, e.g 'COG:X' and 'COG:Y' --> 'COG:X | COG:Y'
    if merge_overlapping:
        
        assert renumber_accessions, "renumber_accession=False is not supported yet."
        
        utils.tprint('Merging gene accessions that lie on the same genes', verbose=verbose)

        ######################################################
        # Algorithm for merging overlapping gene annotations
        #
        # (1) Mapping accessions to symbols. Start with A, B, C, etc. for anchor accession. Then greedily rename other
        #     accessions to 0, 1, 2, etc. until all genes, defined by (contig,start,stop) triples, are covered. The symbols
        #     A,B,C...1,2,3... define a ranking of accession. Output table: loci_genes.  
        #        -- Question: what is this type of cover
        #           called? It is some kind of bipartite/set/vertex cover... Similar to
        #           https://math.stackexchange.com/questions/838581/bipartite-graph-set-cover
        # 
        # (2) For each gene, defined by a (contig,start,stop), identify all of the symbols (A,B,C...1,2,3...) that are annotated
        #     to it. Also, pivot the table to create a column for each accession source (e.g. one column for COGs, another for
        #     Pfams, etc.). Output table: loci_genes_pivot
        #
        # (3) For each symbol (A,B,C...1,2,3...), create a table of all other accessions that overlap with it (i.e. found
        #     together on the same gene). Note that the accessions don't need to perfectly overlap, so the resulting table will
        #     just be a "cover" approximation. For instance, if an accession with symbol A is found on 4 genes, and second
        #     accession (with or without a symbol) is found on only 2 of those gene, then we'll say the oversimplified statement
        #     that the second accession is found together with A. Output_table: loci_genes_merged
        #

        utils.tprint('Greedily identify a subset of accessions that covers all genes ')         
        accession_cover_df = get_anchor_genes(loci_genes, 'gene', 'anchor', sources_numbered, start_genes=genes, contig_list=loci['contig'].tolist())
        # Rename accessions to A, B,..., 0,1,2,...
        accession_cover_df['accession'] = alphabet[:len(genes)] + np.arange(len(accession_cover_df) - len(genes)).astype(str).tolist()
        # Ranking of accessions
        accession_cover_df['order'] = len(genes) - np.arange(len(accession_cover_df))
        accession_cover_df['ncontigs_gain'] = accession_cover_df['ncontigs'].values
        accession_cover = accession_cover_df.index

        # display(accession_cover)
        # display(accession_cover_df)

        loci_genes = loci_genes.rename(columns={'accession':'accession_representative'}).merge(
            accession_cover_df[['accession', 'order', 'ncontigs_gain']], 
            left_on='accession_representative', right_index=True, how='left')
        
        # display(loci_genes)

        # Pivot table so that each function source is its own column
        utils.tprint('Pivot table of accessions and merge symbols per gene instance')
        loci_genes_pivot = loci_genes.groupby(['contig', 'start', 'stop'], observed=True).apply(
            sort_and_show, anchors=accession_cover).reset_index().drop(columns=['level_3'])

        # display(loci_genes_pivot)

        utils.tprint("Compile each symbol's overlapping accessions")
        loci_genes = loci_genes.drop(columns=['ncontigs']).merge(
            loci_genes_pivot, on=['contig','start','stop','accession_representative'],
            how='left', indicator=True)
        loci_genes_pivot = utils.subset(loci_genes, _merge='both').drop(columns=['_merge'])
        loci_genes = loci_genes.drop(columns=['_merge'])
        loci_genes_merged = loci_genes_pivot.groupby('accession', observed=True).apply(concat_accession_instances).reset_index('accession').reset_index(drop=True)

        # Add a column that show the contigs that contain each accession representative
        # -- accession_2_contigs is derived by loci_genes. Deriving from loci_genes_pivot would be an issue because of multiple accessions, e.g. A and B,
        # -- that are in the same contig, but only accession A ends up being assigned to column `accession_representative` in loci_genes_pivot.
        accession_2_contigs = loci_genes.groupby('accession_representative')['contig'].apply(tuple)
        loci_genes_merged = loci_genes_merged.merge(accession_2_contigs.to_frame(), left_on='accession_representative', right_index=True, how='left')

    else:
        loci_genes_pivot = loci_genes.copy()
        loci_genes_pivot['accession'] = [rename(x) for x in loci_genes_pivot['accession']]

    utils.tprint('Done merging accessions')

    if len(genes)==0:
        # Temporary hack: set the direction of all contigs to be "forward"
        loci['direction'] = 'forward'
    elif len(genes) > 0:
        # Add information about direction of anchor gene. For orienting contigs when plotting
        loci = loci.merge(loci_genes_pivot[loci_genes_pivot['accession'].isin(alphabet if renumber_accessions else genes)].sort_values(['contig', 'accession']).drop_duplicates(subset=['contig']).set_index('contig')['direction'],
                          left_on='contig', right_index=True)

    ###############################
    # Color black the query genes #
    ###############################

    if black_genes:
        black_color = (0.5,0.5,0.5) # Gray (not black)
        if merge_overlapping:
            colors = pd.Series([black_color for _ in alphabet], index=alphabet)
        else:
            colors = pd.Series([black_color, black_color], index=[rename(a), rename(b)])
        colors = colors[~ colors.index.duplicated()]  # Drop duplicates, to cover the case that a and b are the same
    else:
        colors = pd.Series([])

    # Color the other accessions: order the colors based on accession priority
    uncolored_genes = [x for x in loci_genes_pivot.drop_duplicates('accession').sort_values('order', ascending=False)['accession'] if x not in colors]
    if not color_singletons:
        singletons = np.setdiff1d(loci_genes_merged.loc[loci_genes_merged['ncontigs_gain']==1, 'accession'].unique(), alphabet)
        uncolored_genes = np.setdiff1d(uncolored_genes, singletons)
        colors = pd.concat([colors, pd.Series({g : (1.,1.,1.) for g in singletons})])
    colors = pd.concat([colors, pd.Series(sns.husl_palette(len(uncolored_genes)), uncolored_genes)])


    ##################
    # Rotate sequences
    ##################

    if (rotate or (rotate_deltas is not None)) and (len(genes) > 0):
        utils.tprint('Rotating sequences', verbose=verbose)

        if rotate_deltas is None:
            # Iterate anchors, and rotate sequences such that the anchor is set to the beginning
            remaining_loci = loci_genes_pivot.copy()
            tmp = []
            for anchor_gene in alphabet[:len(genes)]:
                rotate_deltas = remaining_loci[remaining_loci['accession']==anchor_gene].sort_values(['contig', 'start']).drop_duplicates(subset=['contig'])
                # For anchors pointing in the reverse direction, move them to the end (so that they are at the start, when the contig is flipped)
                rotate_deltas['rotate'] = [-start if (direction==anchor_direction) else (contig_lengths[c] - stop) for \
                                           c, start, stop, direction in rotate_deltas[['contig', 'start', 'stop', 'direction']].to_records(index=False) ] 
                tmp.append(rotate_deltas.set_index('contig')['rotate'])

                # Filter for the remaining loci to be rotated
                remaining_loci = utils.unused_cat(remaining_loci[~ utils.catin(remaining_loci['contig'], rotate_deltas['contig'])])
            rotate_deltas = pd.concat(tmp)
        
        print('Contig rotations')
        display(pd.Series(rotate_deltas).to_dict())

        loci_genes_pivot = rotate_func_aln(lengths=contig_lengths['length'], sequences=rotate_deltas, func=loci_genes_pivot)

        if aln_blocks is not None:
            # Rotate fasta and recompute MUMMER alignments
            fasta = utils.transform_sequences(utils.subset_dict(fasta, loci['contig'].unique()), rotate_deltas)
            aln_blocks = utils.run_mummer(fasta, verbose=True, minmatch=11, threads=threads)

    ######################################
    # Recenter contigs around the anchor #
    ######################################

    if center:
        utils.tprint('Centering sequences around anchor gene(s)', verbose=verbose)

        # ## TODO: handle case where anchor genes occur multiple times in the same contig
        # assert loci_genes_pivot['contig'].nunique() == (loci_genes_pivot['accession']=='A').sum()

        def get_gaps(loci_genes_pivot, anchor_gene, xmin, xmax):
            loci_genes_pivot = loci_genes_pivot[loci_genes_pivot['accession']==anchor_gene]

            # Calculate the "left" gap : the distance between the left side of the anchor gene to the left side of the contig.
            # - Similarly calculate the "right" gap
            left_gap = (loci_genes_pivot.groupby('contig', observed=True)['start'].min() - xmin).rename('left_gap')
            right_gap = (xmax - loci_genes_pivot.groupby('contig', observed=True)['stop'].max()).rename('right_gap')
            
            # Swap left and right gap, if orientation of A is reversed
            # Note: if a loci has multiple instances of the anchor gene, then choose one of them randomly to orient
            gaps = pd.concat([xmin, xmax, left_gap, right_gap, loci_genes_pivot.drop_duplicates(['contig']).set_index('contig')['direction']], 1)
            gaps['xmin'], gaps['xmax'], gaps['left_gap'], gaps['right_gap'] = zip(*[(start,stop,x,y) if d=='f' else (stop,start,y,x) for start,stop,x,y,d in gaps[['xmin','xmax','left_gap','right_gap','direction']].to_records(index=False)])

            # Rescale xmin/xmax across all loci, so that they are on the same plotting scale
            # - this is the key operation that aligns the genomes
            gaps['xmin'] = gaps['xmin'] + np.int32([1 if x=='f' else -1 for x in gaps['direction']]) * (gaps['left_gap'] - gaps['left_gap'].max())
            gaps['xmax'] = gaps['xmax'] + np.int32([-1 if x=='f' else 1 for x in gaps['direction']]) * (gaps['right_gap'] - gaps['right_gap'].max())

            return gaps

        if len(genes)==0:
            gaps = loci.rename(columns={'start' : 'xmin', 'stop' : 'xmax'})
            gaps['xmax'] = gaps['xmax'].max()
            gaps = gaps.set_index('contig')
        else:

            # Iterate anchors, and center contigs' axes boundaries around the anchors
            remaining_loci = loci.copy()
            gaps_list = []
            for anchor_gene in alphabet[:len(genes)]:
                contigs_with_anchor = np.array(loci_genes_pivot.loc[loci_genes_pivot['accession']==anchor_gene, 'contig'].unique())                
                loci_with_anchor = remaining_loci[remaining_loci['contig'].isin(contigs_with_anchor)]
                contigs_with_anchor = np.array(loci_with_anchor['contig'].unique())
#                print('anchor:', anchor_gene, len(remaining_loci), contigs_with_anchor)

                # For every contig, get the min/max coordinate across its genes
                xmin = loci_with_anchor.groupby('contig')['start'].min().rename('xmin')
                xmax = loci_with_anchor.groupby('contig')['stop'].max().rename('xmax')
                
                # Calculate gaps
                gaps = get_gaps(loci_genes_pivot[loci_genes_pivot['contig'].isin(contigs_with_anchor)], anchor_gene, xmin, xmax)
                gaps_list.append(gaps)

                # Update the remaining loci that need to calculate gaps
                remaining_loci = remaining_loci[~ remaining_loci['contig'].isin(contigs_with_anchor)]

                if len(remaining_loci) == 0:
                    break

            gaps = pd.concat(gaps_list)


    ################################
    # Compute alignments if needed #
    ################################
    
    if aln_blocks is None:
        utils.tprint('Calculating MUMmer sequence alignments')
        assert fasta is not None, "Alignments were not specified, so you need to input a fasta file to calculate alignments"
        aln_blocks = ani_utils.run_mummer(fasta, verbose=False, minmatch=11)
#        display(aln_blocks)
    else:
        utils.tprint('Reading prespecified MUMmer sequence alignments')
        aln_blocks = utils.read_table(aln_blocks)

    #################
    # Layout Figure #
    #################

    if width is None:
        width = 18

#    height = 0.1
    height = 0.25
#    height = 1

    hspace = 0.5
    if aln_blocks is not None:
        if aln_blocks_height is None:
            aln_blocks_height = 0.5

        hspace += aln_blocks_height

    plot_cov = cov_dict is not None
    if plot_cov:
        if cov_keys is None:
            cov_keys = cov_dict.columns
        cov_colors = sns.color_palette(n_colors=len(cov_keys))
        height += 1


    nheaders = int(aln_blocks is not None) + int(plot_cov)
    outer_height = nheaders + 1 
    inner_height = len(loci)*height + (len(loci)-1)*hspace
    inner_hspace_ratio = (len(loci)-1)*hspace / inner_height
    print('block height:', height, 'hspace between blocks:', hspace, 'outer height:', outer_height, 'inner height:', inner_height, 'inner hspace ratio:', inner_hspace_ratio)
    fig = plt.figure(figsize=(width, outer_height + inner_height))
                                 
    outer_grid = fig.add_gridspec(2, 1, wspace=0, hspace=0, height_ratios=(outer_height, inner_height))
#    header_grid = outer_grid[0,0].subgridspec(nheaders, 1, wspace=0, hspace=0.5)
    header_grid = outer_grid[0,0].subgridspec(nheaders + 1, 1, wspace=0, hspace=0.5)
#    inner_grid = outer_grid[1,0].subgridspec(len(loci), 1, wspace=0, hspace=hspace)
    inner_grid = outer_grid[1,0].subgridspec(len(loci), 1, wspace=0, hspace=inner_hspace_ratio)

    axes_genes, axes_cov, axes_text = [], [], []
    for idx in range(len(loci)):   
        loci_grid = inner_grid[idx,0].subgridspec(2 if plot_cov else 1, 2, width_ratios=(16,2), wspace=0, hspace=0)
        axes_cov.append(fig.add_subplot(loci_grid[0,0]) if plot_cov else None)
        axes_genes.append(fig.add_subplot(loci_grid[-1,0]))
        axes_text.append(fig.add_subplot(loci_grid[-1,1]))

    if plot_cov:
        # Create a custom legend of the coverage plots 
        # - From https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
        cov_legend_ax = fig.add_subplot(header_grid[nheaders-1,0])
        cov_legend_ax.legend([mpl.lines.Line2D([0], [0], color=c, lw=4) for c in cov_colors],
                         cov_keys,
                         loc='center', ncol=len(cov_keys), frameon=False)
        cov_legend_ax.set_axis_off()

    #############
    # Plot loci #
    #############

    utils.tprint('Plotting sequences', verbose=verbose)

    for idx, (ax, ax_cov, ax_text, (_, r)) in enumerate(zip(axes_genes, axes_cov, axes_text, loci.iterrows())):
        this_contig = r['contig']
        this_original_contig = r['original_contig']
        this_contig_length = contig_lengths[r['original_contig']]

        # String describing the taxonomic assignments of this contig
        assignments_str = ' '.join('%s:%s' % (k,v) for k, v in assignments.loc[this_contig].dropna().to_dict().items())
        assignments_str = '\n'.join(textwrap.wrap(assignments_str, width=80, subsequent_indent='    '))

        # Create string describing contig and locus
        locus_description = [ '{} ({:n} bp)'.format(this_contig, this_contig_length) ]
        if contig_descriptions is not None:
            locus_description.append(contig_descriptions[r['original_contig']])
        if not no_assignments:
            locus_description.append('Assignments:\n{}'.format(assignments_str))
        locus_description = '\n'.join(locus_description)

        # Plot genes
        plot_contig(loci_genes_pivot[loci_genes_pivot['contig']==this_contig].copy(),
                    show=False,
                    rotation=None if renumber_accessions else 'vertical', colors=colors,
                    ax=ax)

        # Plot coverage
        if plot_cov:
            for k, color in zip(cov_keys, cov_colors):
                
                if cov_dict.loc[this_original_contig, k] is None:
                    pos, data, start = np.array([]), np.array([]), np.array([])
                else:
                    pos, data, start = cov_dict.loc[this_original_contig, k]

                if k=='coverage_RF':
                    ax_cov_twin = ax_cov.twinx()
                    circ.plot_coverage(pos=pos, data=data, start=start, length=this_contig_length, format='delta', ax=ax_cov_twin, color=color)
                    ax_cov_twin.set_ylabel(None)
                    ax_cov_twin.tick_params(labelright=True, right=True, left=False, labelleft=False)
                else:
                    circ.plot_coverage(pos=pos, data=data, start=start, length=this_contig_length, format='delta', ax=ax_cov, color=color)

            ax_cov.set_ylabel(None)
            ax_cov.tick_params(labelleft=True, left=True, right=False, labelright=False)
            ax_cov.grid(b=False, which='both', axis='y')

        # Create vertical bars to demarcate the boundaries of the locus
        for x in [r['start'], r['stop']]: ax.axvline(x)

        ##### Adjust axes orientation, limits, ticks, etc.
        for ax2 in [ax] + ([ax_cov] if plot_cov else []):
            # Adjust orientation, so that first gene is always pointing forward
            if (len(genes) > 0) and (r['direction']!=anchor_direction):
#                print('Flipping AXIS ------*******----- %s' % anchor_direction)
                ax2.invert_xaxis()

            # Remove top and bottom spines
            for spines_pos in ['top','bottom','left','right']:
                ax2.spines[spines_pos].set_visible(False)

            # Center anchor
            if center:
                # Set the x-limits so that every contig is plotted on the same scale
                ax2.set_xlim(gaps.loc[r['contig'], 'xmin'], gaps.loc[r['contig'], 'xmax'])

                # Because of this x-limits hack, some xticks will be negative values or go beyond the contig length. REMOVE THESE
                xticks = ax2.get_xticks().tolist()
#                ax2.set_xticks(xticks)
                ax2.set_xticklabels([str(int(x)) if ((x >= 0) and (x < this_contig_length)) else '' for x in xticks])

        if plot_cov:
            ax_cov.tick_params(bottom=False, labelbottom=False)
            ax_cov.set_xlabel(None)

        # Annotate contig description
        ax_text.text(0.1, 0.5, locus_description, horizontalalignment='left', verticalalignment='center', transform=ax_text.transAxes)
        ax_text.set_axis_off()

    ##########################################
    # Draw alignment ribbons between contigs #
    ##########################################

    if aln_blocks is not None:
        aln_blocks = utils.subset(aln_blocks, query=loci['contig'].values, ref=loci['contig'].values)

        # Change aln_blocks from 1-based closed intervals (mummer's format) to 0-based half-open [,) intervals
        # -- This is the default setting
        if (aln_blocks_is_1_based is None) or (aln_blocks_is_1_based):
            aln_blocks['query_end'] -= 1
            aln_blocks['query_start'] -= 1

#        query_length = (aln_blocks['query_end'] - aln_blocks['query_start'] + 1).abs()
        query_length = (aln_blocks['query_end'] - aln_blocks['query_start']).abs()
        aln_blocks['ani'] = (query_length - aln_blocks['errors']) / query_length

        min_ani, max_ani = aln_blocks['ani'].min(), aln_blocks['ani'].max()
        norm = mpl.colors.Normalize(vmin=min_ani, vmax=max_ani)
        cmap = plt.get_cmap('viridis')
        alpha = 0.4

        # Draw ribbons
        for idx, (ax, next_ax, (_, r)) in enumerate(zip(axes_genes, (axes_cov if plot_cov else axes_genes)[1:], loci.iloc[:-1].iterrows())):
            next_loci = loci.iloc[idx+1]
            sub_blocks = utils.subset(aln_blocks, query=r['contig'], ref=next_loci['contig'])
            
            # Iterate through and draw every alignment block
            for _, start, end, next_start, next_end, ani in sub_blocks[['query_start', 'query_end', 'ref_start', 'ref_end', 'ani']].to_records():

                if overlap(start, end, r['start'], r['stop']) and overlap(next_start, next_end, next_loci['start'], next_loci['stop']):

                    # If the alignment block spills over the boundaries of either this or the next locus, then take the relevant slice of the block
                    start = max(start, r['start'])
                    end = min(end, r['stop'])
                    next_start = max(next_start, next_loci['start'])
                    next_end = min(next_end, next_loci['stop'])

                    if (r['direction']!=anchor_direction):
                        start, end = end, start
                    if (next_loci['direction']!=anchor_direction):
                        next_start, next_end = next_end, next_start
                    rgba = cmap(norm(ani))

                    draw_ribbon(ax, next_ax, start, end, next_start, next_end, facecolor=rgba[:3], alpha=alpha)

        # Add colorbar
        # - See https://matplotlib.org/3.1.1/tutorials/colors/colorbar_only.html
        colorbar_ax = fig.add_subplot(header_grid[0,0])
        cb1 = mpl.colorbar.ColorbarBase(colorbar_ax, alpha=alpha, cmap=cmap, norm=norm, orientation='horizontal')
        cb1.set_label('Average Nucleotide Identity')
        colorbar_ax.xaxis.set_label_position('top')
        colorbar_ax.xaxis.set_ticks_position('top')

    if show:
        plt.show()

    if merge_overlapping and show:
        print('Dictionary of merges: representative accession --> list of merged accessions')
        utils.pprint(pprint_merged(loci_genes_merged).set_index('accession'))

    if output is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages('{}_genome_plot.pdf'.format(output)) as pdf:
            pdf.savefig(fig_dist)
            if fig_heatmap is not None:
                pdf.savefig(fig_heatmap.fig)
            pdf.savefig(fig)

        # # Only save alignment
        # fig.savefig('{}_genome_plot.pdf'.format(output))
        # if svg:
        #     fig.savefig('{}_genome_plot.svg'.format(output))

        
        # # This might be uncommented in the future. For now, I will not produce these tables, to minimize number of files
        # pprint_merged(loci_genes_pivot, wrap=0).drop(columns=['ncontigs']).to_csv('{}_annotations.txt'.format(output), sep='\t', header=True, index=False)
        # pprint_merged(loci_genes_merged, wrap=0).to_csv('{}_merged_annotations.txt'.format(output), sep='\t', header=True, index=False)


    ######################################################
    # Create table of gene annotations and append to pdf #
    ######################################################

    if output is not None:

        def color_cell(x):
            if x.name=='accession':
                return ['background-color: %s' % mpl.colors.to_hex(c) for c in colors.loc[x]]
            else:
                return ['' for y in x]

        def justify_cell(x):
            if x.name not in ['accession']:
                return ['text-align: center' for y in x]
            else:
                return ['text-align: center' for y in x]

        # Style the dataframe
        tmp = pprint_merged(loci_genes_merged, wrap=10000, each_wrap=True, fillna='----')[['accession', 'accession_representative', 'ncontigs'] + sources_display].rename(columns={'ncontigs':'contigs'})
        tmp = tmp.style.apply(color_cell, axis=0)
        tmp = tmp.apply(justify_cell, axis=0)
        tmp = tmp.set_properties(**{'border-color': 'black', 'border' : '1px solid black'})
        tmp = tmp.set_table_styles([{'props' : [('border-collapse', 'collapse')]}])
        tmp = tmp.hide_index()

        # Check necessary packages, pdfkit and wkhtmltopdf, for writing dataframe to pdf file
        try:
            import pdfkit
        except ImportError:
            raise Exception("You need to install pdfkit (`pip install pdfkit`).")
        try:
            from PyPDF2 import PdfFileMerger
        except ImportError:
            raise Exception("You need to install pypdf2 (`conda install -c conda-forge pypdf2`)")
        wkhtmltopdf_cmd = shutil.which('wkhtmltopdf')
        if wkhtmltopdf_cmd is None:
            raise Exception("The command wkhtmltopdf could not be found. You may need to install wkhtmltopdf (`conda install -c conda-forge wkhtmltopdf`)."\
                            "If wkhtmltopdf is already installed, you may need to set the $PATH environment variable to include the location of wkhtmltopdf.")

        # Write dataframe to pdf file
        pdfkit.from_string(tmp.render(), f'{output}_annotations.pdf',
                           options={'--page-height' : '410mm', '--page-width': '{}mm'.format(int(175 * len(sources_display)))},
                           configuration=pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_cmd))

        if write_loci:
            # Write dataframe of loci to text file
            tmp = loci.drop(columns=['contig']).rename(columns={'original_contig' : 'contig'})[['contig', 'start', 'stop', 'direction']]
            if contigs_metadata is not None:
                tmp = tmp.merge(contigs_metadata, left_on='contig', right_index=True, how='left')
                if 'description' in tmp.columns:
                    tmp['description'] = [x.replace('\n', ' ') for x in tmp['description']]
            tmp.to_csv('{}_loci.txt'.format(output), sep='\t', header=True, index=False)

        # Merge pdf files
        merger = PdfFileMerger()
        for pdf in [f'{output}_genome_plot.pdf', f'{output}_annotations.pdf']:
            merger.append(pdf)
        merger.write(f'{output}')
        merger.close()
        os.remove(f'{output}_genome_plot.pdf')
        os.remove(f'{output}_annotations.pdf')

    if not show:
        plt.close(fig)
        plt.close(fig_dist)
        if fig_heatmap is not None:
            plt.close(fig_heatmap.fig)

    return instances, loci, loci_genes, loci_genes_pivot, loci_genes_merged, fig, fig_dist, fig_heatmap



def rotate_func_aln(lengths, sequences=None, func=None, aln_blocks=None):
    """
    Assumes that aln_blocks is 1-index based
    """

    def rotate_one_contig(func, length, seq_delta, start_col, stop_col, index_1=False):
        """ Rotates one contig.

        start_col : column name of start coordinate, e.g. 'start'
        stop_col : column name of stop coordinate, e.g. 'stop'

        Assumes coordinates are 0-based half open (start, stop]
        """

        new_start_col = 'tmp_' + start_col
        new_stop_col = 'tmp_' + stop_col

        # The bottom analyses assumes that `start` < `stop`. 
        # - This may not be the case, so create a dummy start and stop col that does satisfy these conditions
        func[new_start_col] = np.minimum(func[start_col], func[stop_col])
        func[new_stop_col] = np.maximum(func[start_col], func[stop_col])
        func['tmp_direction'] = func[start_col] <= func[stop_col]
        if index_1:
            # For (start, stop) intervals that are flipped, need to convert them to 0-based indices oriented (start, stop]
#            func[new_start_col] += (~ func['tmp_direction']).astype(int)
            func[new_start_col] -= 1

        # Update gene function starts/stops
        func[new_start_col] += seq_delta
        func[new_stop_col] += seq_delta
        
        #####
        # Note: whenever taking the modulo of new_stop_col (which is
        # assumed to be 0-based inidices, need to first subtract 1 and
        # then add 1 after the modulo

        # Functions to break into two parts
        to_break = (func[new_start_col] // length) != ((func[new_stop_col] - 1) // length)

        # Second half of parts to break
        second_halves = func[to_break].copy()
        second_halves[new_stop_col] = ((second_halves[new_stop_col]-1) % length)+1
        second_halves[new_start_col] = 0
        if 'partial' in second_halves.columns:
            second_halves['partial'] = 1

        # First half of parts to break
        func.loc[to_break, new_start_col] = func[new_start_col] % length
        func.loc[to_break, new_stop_col] = length

        # Genes not to break
        func.loc[~to_break, new_start_col] = func.loc[~to_break, new_start_col] % length
        func.loc[~to_break, new_stop_col] = ((func.loc[~to_break, new_stop_col]-1) % length)+1

        func = pd.concat([func, second_halves])

        # Convert back to 1-based indices
        if index_1:
            func[new_start_col] += 1

        # Convert (start, stop) intervals back to the original orientation, such that `start` is possible greater than `stop`
        func[start_col] = [x if d else y for d, x, y in zip(func['tmp_direction'], func[new_start_col], func[new_stop_col])]
        func[stop_col] = [y if d else x for d, x, y in zip(func['tmp_direction'], func[new_start_col], func[new_stop_col])]
        func = func.drop(columns=[new_start_col, new_stop_col, 'tmp_direction'])

        return func


    new_func = []
    new_aln_blocks = []
    contig_2_func = {a:b for a, b in func.groupby('contig', observed=True)}
    if aln_blocks is not None:
        query_contig_2_aln_blocks = {a:b for a,b in aln_blocks.groupby('query', observed=True)}
        ref_contig_2_aln_blocks = {a:b for a,b in aln_blocks.groupby('ref', observed=True)}

    for contig_name, seq_delta in sequences.items():

        # Sequence length
        length = lengths[contig_name]

        new_func.append(rotate_one_contig(contig_2_func[contig_name].copy(), length, seq_delta, 'start', 'stop'))

        # if aln_blocks is not None:
        #     new_aln_blocks.append(rotate_one_contig(query_contig_2_aln_blocks[contig_name].copy(), length, seq_delta, 'query_start', 'query_end', index_1=True))
        #     new_aln_blocks.append(rotate_one_contig(ref_contig_2_aln_blocks[contig_name].copy(), length, seq_delta, 'ref_start', 'ref_end', index_1=True))

    func = pd.concat(new_func).astype(func.dtypes).sort_values(['contig', 'start', 'stop'])

    # if aln_blocks is not None:
    #     aln_blocks = pd.concat(new_aln_blocks).astype(aln_blocks.dtypes).sort_values(['query', 'query_start', 'query_end', 'ref', 'ref_start', 'ref_end'])

    # return func, aln_blocks

    return func
            
def pprint_merged(merged_table, wrap=75, each_wrap=False, delim='\n', fillna=False, sources=None):
    """
    Helper function for visualize_alignment().

    Pretty formatting of a merged gene table, e.g. text-wrapping.
    """
    
    merged_table = merged_table.copy()
    
    # Infer which columns refer to gene function sources
    if sources is None:
        sources = [c for c in merged_table.columns if c not in ['accession', 'accession_representative', 'contig', 'ncontigs', 'ncontigs_gain', 'instances', 'order']]
        
    # The columns to do text wrapping
    columns_to_wrap = sources
    if isinstance(merged_table['contig'].iloc[0], str):
        columns_to_wrap.append('contig')
    
    for c in columns_to_wrap:
        if each_wrap:
            merged_table[c] = merged_table[c].apply(lambda x: '\n'.join(['\n'.join(textwrap.wrap(y, wrap)) for y in x]) if (wrap>0) else '|'.join(x))
        else:
            merged_table[c] = merged_table[c].apply(lambda x: '\n'.join(textwrap.wrap('|'.join(x), wrap)) if (wrap>0) else '|'.join(x))

        if fillna is not False:
            merged_table[c] = [fillna if (x=='') else x for x in merged_table[c]]

    merged_table = merged_table.sort_values(['order'], ascending=False)
    merged_table = utils.rearrange(merged_table, ['accession', 'accession_representative', 'ncontigs', 'ncontigs_gain', 'instances', 'contig'], [0, 1, 2, 3, 4, 5])
    merged_table = merged_table.drop(columns=['order'])
        
    return merged_table

def draw_ribbon(ax1, ax2, xmin, xmax, xmin2, xmax2, **kwargs):
    """
    INSPIRED FROM https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_zoom_effect.html
    
    Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both axes will
    be marked.

    Parameters
    ----------
    ax1
        The main axes.
    ax2
        The zoomed axes.
    xmin, xmax
        The limits of the colored area in both plot axes.
    **kwargs
        Arguments passed to the patch constructor.
    """

    from matplotlib.transforms import (
        Bbox, TransformedBbox, blended_transform_factory)
    from mpl_toolkits.axes_grid1.inset_locator import (
        BboxPatch, BboxConnector, BboxConnectorPatch)


    def connect_bbox(bbox1, bbox2,
                     loc1a, loc2a, loc1b, loc2b,
                     prop_lines, prop_patches=None):
        c1 = BboxConnector(
            bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
        c2 = BboxConnector(
            bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)
        
    #     bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    #     bbox_patch2 = BboxPatch(bbox2, **prop_patches)

        p = BboxConnectorPatch(bbox1, bbox2,
                               # loc1a=3, loc2a=2, loc1b=4, loc2b=1,
                               loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                               clip_on=False,
                               **prop_patches)
        return c1, c2, p

    #     return c1, c2, bbox_patch1, bbox_patch2, p

#        return p

    bbox = Bbox.from_extents(xmin, 0, xmax, 1)
    mybbox1 = TransformedBbox(bbox, ax1.get_xaxis_transform())
    bbox = Bbox.from_extents(xmin2, 0, xmax2, 1)
    mybbox2 = TransformedBbox(bbox, ax2.get_xaxis_transform())

#    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}
    prop_patches = {**kwargs, "ec": "none"}

#     c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
#    p = connect_bbox(
    c1, c2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines={'alpha' : 0.5, 'ec' : 'gray', 'linewidth' : 1, 'capstyle' : 'butt'}, prop_patches=prop_patches)

#     ax1.add_patch(bbox_patch1)
#     ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)


def slop(functions, slop, sizes=None):
    """
    Equivalent to bedtools' slop utility. Extends each function by a number of nucleotides

    functions : 

        anvi'o formatted gene functions table

    slop : 
    
        number of nucleotides

    sizes :
    
         Dictionary or pd.Series where keys are contigs, and values are the contig sizes.

         Necessary to make sure that slop does not go over contig length
    """

    functions = functions.copy()

    # display(functions.dtypes)
    # display(functions)

    functions['start'] = functions['start'] - slop
    functions['stop'] = functions['stop'] + slop

    if sizes is not None:
        # sizes = pd.Series(sizes).to_frame('size')
        # functions = functions.merge(sizes, left_on='contig', right_index=True, how='left')
        functions['size'] = sizes.loc[np.asarray(functions['contig'].values)].values
        assert functions['size'].notna().all(), 'Sizes specified are NA for some contigs, e.g. {}'.format(functions[functions['size'].isna()].sample(5))

        functions['stop'] = np.minimum(functions['stop'], functions['size'])
        functions.drop(columns=['size'], inplace=True)

    functions['start'] = np.maximum(0, functions['start'])

    # display(functions)

    return functions

def interval_merge_self(functions, apply=None):
    """Takes an anvio functions-format table (consists of columns ['contig', 'start', 'stop']),
    and merges sets of intervals that are intersecting (specifically, connected components of overlapping intervals).
    
    This is akin to bedtools' merge function
    
    apply : a function to apply on a group of rows to be merged together

    Returns
    --------
    
    New dataframe with ['contig', 'start', 'stop'] columns
    """
    
    
    def interval_merge_same_contig(functions):
        """Merges intervals in the same contig"""

        if len(functions) <= 1:
            if apply is None:
                return functions[['contig', 'start', 'stop']]
            return apply(functions)
        else:
            # Construct intervals
            intervals = pd.arrays.IntervalArray.from_tuples(list(zip(functions['start'], functions['stop'])), closed='left')
            
            # Construct interval-by-interval overlap matrix
            overlap_matrix = np.array([intervals.overlaps(x) for x in intervals])
            
            # Turn overlap matrix into igraph, and then compute connected components
            this_contig = functions['contig'].iloc[0]

            if apply is None:
                merged_intervals = []
                for comp in list(utils.create_ig(overlap_matrix, weighted=False, directed=False).components()):
                    start, stop = functions.iloc[comp]['start'].min(), functions.iloc[comp]['stop'].max()
                    merged_intervals.append( (this_contig, start, stop) )
                return pd.DataFrame(merged_intervals, columns=['contig', 'start', 'stop'])
            else:
                merged_intervals = []
                for comp in list(utils.create_ig(overlap_matrix, weighted=False, directed=False).components()):
                    merged_intervals.append(apply(functions.iloc[comp]))
                    # start, stop = functions.iloc[comp]['start'].min(), functions.iloc[comp]['stop'].max()
                    # merged_intervals.append( (this_contig, start, stop) )
                return pd.concat(merged_intervals)
        
    return functions.groupby('contig', as_index=False, group_keys=False).apply(interval_merge_same_contig).reset_index(drop=True)


def overlap(start1, stop1, start2, stop2):
    start1_isin_2 = ((start1 >= start2) and (start1 < stop2))
    start2_isin_1 = ((start2 >= start1) and (start2 < stop1))
    return start1_isin_2 or start2_isin_1

def interval_merge_same_contig(functions, functions2, suffix=None):
    """Merges intervals in the same contig"""

    # Construct intervals
    intervals = pd.arrays.IntervalArray.from_tuples(list(zip(functions['start'], functions['stop'])), closed='left')
    intervals2 = pd.arrays.IntervalArray.from_tuples(list(zip(functions2['start'], functions2['stop'])), closed='left')

    # print(intervals)
    # print(intervals2)
    # print('----------------------')
#    mask_list = [intervals2.overlaps(x) for x in intervals]

    # mask_list = []
    # for x in intervals:
    #     mask_list.append(intervals2.overlaps(x))

    # mask_list = []
    # for start1, stop1 in zip(functions['start'], functions['stop']):
    #     mask = [overlap(start1,stop1,start2,stop2) for start2, stop2 in zip(functions2['start'], functions2['stop'])]
    #     mask_list.append(np.where(mask)[0])

    start1, stop1 = functions['start'].values.reshape(-1,1), functions['stop'].values.reshape(-1,1)
    start2, stop2 = functions2['start'].values.reshape(1,-1), functions2['stop'].values.reshape(1,-1)
    mask = ((start1 >= start2) & (start1 < stop2)) | ((start2 >= start1) & (start2 < stop1))
    idx1, idx2 = mask.nonzero()

#    functions2_repeated = functions2.iloc[np.concatenate(mask_list)].reset_index(drop=True)
    functions2_repeated = functions2.iloc[idx2].reset_index(drop=True)

    functions_repeated = functions.iloc[idx1].reset_index(drop=True)

#    functions2_repeated = [functions2[m] for m in mask_list]

#    functions2_repeated = [functions2[intervals2.overlaps(x)] for x in intervals]
    if suffix is not None:
        functions2_repeated = functions2_repeated.rename(columns=lambda a: '{}_{}'.format(a, suffix))
#        functions2_repeated = [x.rename(columns=lambda a: '{}_{}'.format(a, suffix)) for x in functions2_repeated]

    # functions_merged = pd.concat([functions.iloc[np.repeat(np.arange(len(functions)), [len(x) for x in functions2_repeated])].reset_index(drop=True),
    #                               pd.concat(functions2_repeated).reset_index(drop=True)], 1)
    # functions_merged = pd.concat([functions.iloc[np.repeat(np.arange(len(functions)), [len(x) for x in mask_list])].reset_index(drop=True),
    #                               functions2_repeated], 1)
    functions_merged = pd.concat([functions_repeated, functions2_repeated], 1)

#    display(functions_merged)
    return functions_merged

def interval_merge(functions, functions2, suffix=None):
    """Takes two anvio functions-format table (consists of columns ['contig', 'start', 'stop']),
    and merges sets of intervals that are intersecting (specifically, connected components of overlapping intervals).

    This is akin to bedtools' merge function

    
    Returns
    --------

    New dataframe with ['contig', 'start', 'stop'] columns
    
    Rows in `functions` that overlap with rows in `functions2` are concatendated together on axis=1. Columns in `functions` are renamed with `suffix` (if specified)
    
    """

    # def interval_merge_same_contig(functions, functions2):
    #     """Merges intervals in the same contig"""

    #     # Construct intervals
    #     intervals = pd.arrays.IntervalArray.from_tuples(list(zip(functions['start'], functions['stop'])), closed='left')
    #     intervals2 = pd.arrays.IntervalArray.from_tuples(list(zip(functions2['start'], functions2['stop'])), closed='left')

    #     functions2_repeated = [functions2[intervals2.overlaps(x)] for x in intervals]
    #     if suffix is not None:
    #         functions2_repeated = [x.rename(columns=lambda a: '{}_{}'.format(a, suffix)) for x in functions2_repeated]
    #     functions_merged = pd.concat([functions.iloc[np.repeat(np.arange(len(functions)), [len(x) for x in functions2_repeated])].reset_index(drop=True),
    #                                   pd.concat(functions2_repeated).reset_index(drop=True)], 1)
    #     return functions_merged

    contig_2_functions = {k : v for k, v in functions.groupby('contig')}
    contig_2_functions2 = {k : v for k, v in functions2.groupby('contig')}
    
    merged_intervals = []
    for contig in set(contig_2_functions.keys()) & set(contig_2_functions2.keys()):
        merged_intervals.append(interval_merge_same_contig(contig_2_functions[contig], contig_2_functions2[contig], suffix=suffix))
    return pd.concat(merged_intervals)
    

def overlap_accessions(functions,
                       method='window_0',
                       single_counts=None,
                       cap_per_contig=None,
                       contigs_groupings=None,
                       **kwargs):
    """Calculates which pairs of accessions lie exactly on top of each other.

    ##################
    # Current method #

    This is done by seeing which accession are "0" genes away from each other, i.e lie on top of each other

    #######################
    ## Deprecated method

    This is done by converting loci, defined as a triplet (contig,
    start, stop), into virtual contigs. So only accessions that lie
    exactly on top of each other are on the same virtual contig

    contigs_groupings :

        If contigs_groupings was not specified (None), then multiple
        co-occcurrences in the same contig will be counted (because
        each of its virtual contigs will be counted once). However,
        you might want to actually count the unique number of contigs
        (not virtual contigs). If `identity_groupings`=True, then
        virtual contigs will be grouped together.

    """

    if single_counts is None:
        single_counts = False

    if 'norm' not in kwargs:
        kwargs['norm'] = False
    if kwargs['norm']:
        assert single_counts is True, 'single_counts must be True if `norm` is set to True'

    if method == 'window_0':
        sp, dic, sp_df = enrich_pair(functions,
                                     0,
                                     window_type='genes',
                                     cap_per_contig=cap_per_contig,
                                     contigs_groupings=contigs_groupings,
                                     single_counts=single_counts,
                                     overlap=False,
                                     **kwargs)
        return sp, dic, sp_df

    elif method == 'virtual_contigs':
        ###############
        ## Deprecated

        functions = functions.sort_values(['start', 'stop', 'contig'])

        if contigs_groupings is None:
            contigs_groupings = pd.Series(np.arange(functions['contig'].cat.categories.size),
                                          index=functions['contig'].cat.categories)

        utils.tprint('Merging virtual contig IDs')
        loci = functions[['start', 'stop', 'contig']].drop_duplicates()
        loci['locusID'] = np.arange(loci.shape[0])

        functions = pd.merge(functions, loci, on=['start', 'stop', 'contig'])[['accession', 'locusID']]
        functions['start'] = 0
        functions['stop'] = 1
        functions = functions.rename(columns={'locusID': 'contig'})
        functions = functions.astype({'contig': 'category'})

        # Change contigs_groupings, so that virtual contigs will be given the groupings of their parent contig
        if contigs_groupings is not None:
            contigs_groupings = pd.Series(contigs_groupings.loc[loci['contig']].values, loci['locusID'].values)

        # Note, window size doesn't matter
        sp, dic, _ = enrich_pair(functions,
                                   100000,
                                   contigs_groupings=contigs_groupings,
                                   single_counts=single_counts,
                                   overlap=False,
                                   **kwargs)
        return sp, dic

    else:
        raise Exception()

