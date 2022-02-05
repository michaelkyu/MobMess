import plasx.pd_utils

from mobmess import utils

def sparse_pivot(*args, ig_kws=None, **kwargs):
    """
    Wrapper around plasx.pd_utils.sparse_pivot. This allows one to specify rettype='ig', which will produce a igraph.Graph object
    """

    if 'rettype' in kwargs and kwargs['rettype']=='ig':

        # Run sparse_pivot() with rettype='spmatrix'
        kwargs['rettype'] = 'spmatrix'
        sp, rownames, colnames = plasx.pd_utils.sparse_pivot(*args, **kwargs)
        
        # Convert to a igraph.Graph
        if ig_kws is None:
            ig_kws = {}
        return utils.create_ig(sp,
                               weighted=True,
                               directed=kwargs.get('directed', None),
                               rownames=rownames,
                               colnames=colnames,
                               square=kwargs.get('square', True),
                               **ig_kws)
    else:
        return plasx.pd_utils.sparse_pivot(*args, **kwargs)
