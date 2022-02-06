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


def merge_gene_calls_and_functions(functions,
                                   gene_calls,
                                   contig_column=None,
                                   annotation_column=None,
                                   add_dummy_annotations=None,
                                   verbose=None):
    """
    Merges two tables:
    (1) A gene calls table that contains (contig,start,stop,direction) information about each gene. Each gene is given a unique integer id, called 'gene_callers_id'. This table is created by `anvi-export-gene-calls` in anvi'o.
    (2) A gene functions table that contains a mapping of 'gene_callers_id' to an annotation. This table is created by `anvi-export-functions` in anvi'o

    Merging is done on 'gene_callers_id'.

    """
    
    
    # Set default names of contig/annotation-columns
    if contig_column is None:
        contig_column = 'contig'

    annotation_column_was_none = False
    if annotation_column is None:
        annotation_column_was_none = True
        annotation_column = 'accession'

    functions = utils.read_table(functions, verbose=verbose)

    if gene_calls is not None:
        gene_calls = utils.read_table(gene_calls, verbose=verbose)

    def map_gene_callers_id_to_contig(functions):
        """Maps gene_callers_id column to contig.

        Example use case: a table from anvi-export-functions
        contains only gene_callers_id. The mapping to contigs can
        be found from anvi-export-gene-calls
        """

        if (gene_calls is not None) and (contig_column not in functions.columns):
            assert 'gene_callers_id' in functions.columns, "Column 'contig' and 'gene_callers_id' were not in functions table."
            assert 'gene_callers_id' in gene_calls.columns, "Column 'gene_callers_id' was not in gene calls table."

            cols = [contig_column, 'gene_callers_id']
            for c in ['start','stop','direction']:
                if c in gene_calls.columns:
                    cols.append(c)

            if verbose:
                print('Attempting merge of gene calls and functions')

            if add_dummy_annotations:
                functions = functions.merge(gene_calls[cols], on=['gene_callers_id'], how='outer')
#                display(functions)

                assert 'source' in functions.columns
                functions['source'] = functions['source'].fillna('dummy_annotation')

                if 'descriptions' in functions.columns:
                    description_col = 'descriptions'
                elif 'function' in functions.columns:
                    description_col = 'function'
                else:
                    description_col = None
                    
                if description_col is not None:
                    functions['description'] = [f'{c}_{start}_{stop}_{direction}' if pd.isna(description) else description \
                                                for c, start, stop, direction, description in functions[['contig', 'start', 'stop', 'direction', description_col]].to_records(index=False)]
#                display(functions)

            else:
                functions = functions.merge(gene_calls[cols], on=['gene_callers_id'])
        else:
            print('Skipping merge of gene calls and functions')

        return functions

    try:
        C = utils.read_table(functions,
                             verbose=verbose,
                             post_apply=map_gene_callers_id_to_contig)
    except KeyError:
        if annotation_column_was_none:
            annotation_column = 'annotation'

        C = utils.read_table(functions,
                             verbose=verbose,
                             post_apply=map_gene_callers_id_to_contig)
    
    return C
