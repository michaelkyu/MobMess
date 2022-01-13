import pandas as pd
import numpy as np
import scipy.sparse

def create_ig(X, weighted=True, directed=True,
              square=None, rownames=None, colnames=None,
              vertex_attrs=None,
              v1=None,
              v2=None):
    """
    Create a weighted, directed igraph.Graph object from a numpy array, sparse matrix, DataFrame
    """

    import igraph

    if square is None:
        square = True

    if isinstance(X, np.ndarray) or scipy.sparse.isspmatrix(X):
        if scipy.sparse.isspmatrix_coo(X):
            X = X.tocsr()

        edges = X.nonzero()
        if weighted:
            edge_attrs = {'weight' : np.array(X[edges]).flatten()}
        else:
            edge_attrs = {}

        if square:
            assert X.shape[0]==X.shape[1]
            edges = list(zip(*edges))

            if vertex_attrs is None:
                vertex_attrs = pd.DataFrame()

            if rownames is not None:
                assert colnames is None, "You should only specify `rownames`, if square=True"
                name_list = list(rownames)
                vertex_attrs = vertex_attrs.reindex(name_list)
                vertex_attrs['name'] = name_list

            vertex_attrs = {c : vertex_attrs[c].values.flatten() for c in vertex_attrs.columns}

            g = igraph.Graph(n=X.shape[0],
                             edges=edges,
                             directed=directed,
                             vertex_attrs=vertex_attrs,
                             edge_attrs=edge_attrs)
        else:
            edges_i, edges_j = edges
            
            # Number the nodes with the rows as 1,2,..,n, and then the
            # columns as n+1,n+2,...,m
            edges_j += X.shape[0]
            
            edges = list(zip(edges_i, edges_j))

            if vertex_attrs is None:
                vertex_attrs = pd.DataFrame()

            if rownames is not None and colnames is not None:
                name_list = list(rownames) + list(colnames)
                vertex_attrs = vertex_attrs.reindex(name_list)
                vertex_attrs['name'] = name_list

            vertex_attrs = {c : vertex_attrs[c].values.flatten() for c in vertex_attrs.columns}

            g = igraph.Graph(n=X.shape[0] + X.shape[1],
                             edges=edges,
                             directed=directed,
                             edge_attrs=edge_attrs,
                             vertex_attrs=vertex_attrs)
    elif isinstance(X, pd.DataFrame):
        assert (v1 is not None) and (v2 is not None)

        X = X.copy().astype({v1 : 'category', v2 : 'category'})

        names = X[v1].cat.categories.union(X[v2].cat.categories)
        X[v1] = X[v1].cat.set_categories(names)
        X[v2] = X[v2].cat.set_categories(names)
        
        edges = list(zip(X[v1].cat.codes, X[v2].cat.codes))

        edge_attrs = X[[c for c in X.columns if c not in [v1,v2]]].to_dict('list')

        if vertex_attrs is None:
            vertex_attrs = pd.DataFrame(index=names)

        if vertex_attrs is not None:
            vertex_attrs = vertex_attrs.reindex(names)

            # display(vertex_attrs.index.dtype)
            # display(vertex_attrs.dtypes)


            # Set the names to be strings (because if it is a mixed
            # type, e.g. mix of integers and strings, then it won't be
            # recognized as an attribute when exporting to graphml in
            # the potential future)
            names = [str(x) for x in names]

            vertex_attrs['name'] = names

            # # Create a redundant vertex attribute 'label', which is
            # # the same as the node names. This is needed when
            # # converting igraph to graphml and importing to cytoscape,
            # # because the vertex attribute 'name' won't get
            # # interpreted
            # vertex_attrs['label'] = names
            # vertex_attrs['name_copy'] = names
            # display(vertex_attrs)

#            vertex_attrs = {c : vertex_attrs[c].values.flatten() for c in vertex_attrs.columns}

            # Need to replace np.nans with None, in order to be recognized by igraph
#            vertex_attrs = {c : [None if pd.isna(x) else x for x in vertex_attrs[c].values.flatten()] for c in vertex_attrs.columns}
            vertex_attrs = {c : [None if pd.isna(x) else x for x in vertex_attrs[c].values] for c in vertex_attrs.columns}

#            vertex_attrs = {c : vertex_attrs[c].fillna(None).tolist() for c in vertex_attrs.columns}

        # else:
        #     vertex_attrs = {}

        g = igraph.Graph(n=len(names),
                         edges=edges,
                         directed=directed,
                         edge_attrs=edge_attrs,
                         vertex_attrs=vertex_attrs)
        
    else:
        raise Exception()
        
    return g

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

def reachability(X):
    """
    Given a child-to-parent adjacency matrix, calculate a child-to-ancestor reachability matrix
    """

    tmp = create_ig(X).neighborhood(mode='out', order=X.shape[0])
    indices = np.concatenate(tmp)
    X_paths = scipy.sparse.csr_matrix((np.ones(indices.size, np.bool_), indices, np.append(0, np.cumsum([len(x) for x in tmp]))))
    return X_paths
