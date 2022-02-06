###############################################
#
# Utility functions for manipulating graphs. 
#
# Includes interfaces with igraph package
#
###############################################

import os
import pathlib

import pandas as pd
import numpy as np
import scipy.sparse

from mobmess.sp_utils import *

def ig_to_df(G, node_names=None):
    """
    Converts an igraph.Graph into two dataframes of vertex and edge attributes
    """

    vertex_attrs = pd.DataFrame({attr: G.vs[attr] for attr in G.vertex_attributes()})
    if node_names is None:
        node_names = ['v1','v2']
    edges = pd.DataFrame([(e.source, e.target, G.vs[e.source]['name'], G.vs[e.target]['name']) for e in G.es], columns=node_names + [x+'_name' for x in node_names])
    edge_attrs = pd.concat([edges, 
                            pd.DataFrame({attr: G.es[attr] for attr in G.edge_attributes()})],
                           axis=1)
    return vertex_attrs, edge_attrs

def create_ig(X,
              weighted=True,
              directed=True,
              square=None,
              rownames=None,
              colnames=None,
              vertex_attrs=None,
              edge_attrs=None,
              default_edge_attrs=None,
              v1=None,
              v2=None,
              isolated_vertex_attrs=None,
              graph_name=None,
              output=None):
    """
    Create a weighted, directed igraph.Graph object from a numpy array, sparse matrix, DataFrame

    vertex_attrs : pd.DataFrame of vertex attributes. The index must be the vertex names.
    """

    import igraph

    if square is None:
        square = True

    if isolated_vertex_attrs is None:
        isolated_vertex_attrs = True

    if isinstance(X, np.ndarray) or scipy.sparse.isspmatrix(X):
        if scipy.sparse.isspmatrix_coo(X):
            X = X.tocsr()

        edges_i, edges_j = X.nonzero()

        if weighted or (edge_attrs is not None):
            ## Create a pd.DataFrame describing the edges
            edge_attrs_all = pd.DataFrame({'row' : edges_i, 'col': edges_j})

            # Add edge weights
            if weighted:
                edge_attrs_all['weight'] = np.array(X[edges_i, edges_j]).flatten()
                
            # Add user-specified edge attributes
            if edge_attrs is not None:
                assert all(c in edge_attrs.columns for c in ['row', 'col']),\
                    "`edge_attrs` must be a pandas DataFrame with columns 'row' and 'col'"        
                edge_attrs_all = edge_attrs_all.merge(edge_attrs, on=['row','col'], how='left', validate='1:1')

                # Fill NA values in edge attributes with predefined default values
                if default_edge_attrs is not None:
                    for c, v in default_edge_attrs.items():
                        edge_attrs_all[c] = edge_attrs_all[c].fillna(v)
                for c in edge_attrs_all.columns:
                    if (c not in ['row', 'col']) and (edge_attrs_all[c].isna().sum()>0):
                        print(f"WARNING: edge attribute {c} has NA values and won't be written into graphml")

            # Use the preferred variable name `edge_attrs`
            edge_attrs = edge_attrs_all
            
            # Turn this into a dictionary of {column : list of values}
            edge_attrs = edge_attrs.to_dict('list')

            # Need to replace np.nan's with None, in order to be recognized by igraph
            edge_attrs = {k : [None if pd.isna(x) else x for x in v] for k, v in edge_attrs.items()}

        else:
            edge_attrs = {}

        if vertex_attrs is None:
            vertex_attrs = pd.DataFrame()

        # This will be be set only if you supply `rownames` (and `colnames`, if square is False)
        names = None

        if square:
            # This code assumes that the rows and columns of X are the
            # same (and in the same order)
            assert X.shape[0]==X.shape[1]

            if rownames is not None:
                if colnames is not None:
                    assert len(rownames)==len(colnames) and np.all(np.array(rownames)==np.array(colnames)),\
                        'You set square=True, but `rownames` and `colnames` were inconsistent. Resolve this inconsistency or input only `rownames` or `colnames, but not both.'
                names = list(rownames)

            n_nodes = X.shape[0]
        else:
            # This code assumes that the rows and columns of X are
            # different (even if some rows and cols have the same
            # name, they'll be represented by different graph
            # vertices)

            # Number the nodes with the rows as 1,2,..,n, and then the
            # columns as n+1,n+2,...,m
            edges_j += X.shape[0]

            assert (rownames is not None) != (colnames is not None)
            
            if rownames is not None and colnames is not None:
                names = list(rownames) + list(colnames)
            
            n_nodes = X.shape[0] + X.shape[1]

        if names is not None:
            if isolated_vertex_attrs:
                # This adds the isolated vertices, i.e. those in `vertex_attrs` but don't have edges to them
                #vertex_attrs = vertex_attrs.reindex(vertex_attrs.index.union(names))
                vertex_attrs = vertex_attrs.reindex(names + [v for v in vertex_attrs.index if v not in names])
                names = vertex_attrs.index
            else:
                vertex_attrs = vertex_attrs.reindex(names)
            vertex_attrs['name'] = names

            n_nodes = len(names)

        vertex_attrs = {c : vertex_attrs[c].values.flatten() for c in vertex_attrs.columns}

        edges = list(zip(edges_i, edges_j))

        g = igraph.Graph(n=n_nodes,
                         edges=edges,
                         directed=directed,
                         vertex_attrs=vertex_attrs,
                         edge_attrs=edge_attrs)

    elif isinstance(X, pd.DataFrame):
        assert edge_attrs is None
        assert (v1 is not None) and (v2 is not None)

        X = X.copy().astype({v1 : 'category', v2 : 'category'})

        names = X[v1].cat.categories.union(X[v2].cat.categories)

        if vertex_attrs is None:
            vertex_attrs = pd.DataFrame(index=names)

        if isolated_vertex_attrs:
            # This adds the isolated vertices, i.e. those in `vertex_attrs` but don't have edges to them
            vertex_attrs = vertex_attrs.reindex(vertex_attrs.index.union(names))
            names = vertex_attrs.index
        else:
            vertex_attrs = vertex_attrs.reindex(names)

        X[v1] = X[v1].cat.set_categories(names)
        X[v2] = X[v2].cat.set_categories(names)
        
        edges = list(zip(X[v1].cat.codes, X[v2].cat.codes))

        edge_attrs = X[[c for c in X.columns if c not in [v1,v2]]].to_dict('list')

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

        # vertex_attrs = {c : vertex_attrs[c].values.flatten() for c in vertex_attrs.columns}

        # Need to replace np.nans with None, in order to be recognized by igraph
        vertex_attrs = {c : [None if pd.isna(x) else x for x in vertex_attrs[c].values] for c in vertex_attrs.columns}

        g = igraph.Graph(n=len(names),
                         edges=edges,
                         directed=directed,
                         edge_attrs=edge_attrs,
                         vertex_attrs=vertex_attrs)
    else:
        raise Exception()

    # Write graphml file
    if output is not None:
        # - Check that `output` is a string or pathlib.Path object
        # - Convert to str (because sending a Path object to igraph.write_graphml will cause segmentation fault!!!)
        assert isinstance(output, (str, pathlib.PosixPath))
        output = str(output)
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        g.write_graphml(output)

    if graph_name is not None:
        g['name'] = graph_name

    return g

def reachability(X):
    """
    Given a child-to-parent adjacency matrix, calculate a child-to-ancestor reachability matrix
    """

    tmp = create_ig(X).neighborhood(mode='out', order=X.shape[0])
    indices = np.concatenate(tmp)
    X_paths = scipy.sparse.csr_matrix((np.ones(indices.size, np.bool_), indices, np.append(0, np.cumsum([len(x) for x in tmp]))))
    return X_paths
