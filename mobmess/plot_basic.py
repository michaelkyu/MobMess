from collections import OrderedDict
import math

import numpy as np
import pandas as pd
import scipy

from mobmess import utils
from mobmess.dummy_module import DummyModule

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = DummyModule("matplotlib")

try:
    import seaborn as sns
except ImportError:
    sns = DummyModule("seaborn")


def histplot(*args,
             xlabel=None, ylabel=None, title=None, show=None, 
             **kwargs):
    """
    Wrapper around seaborn.histplot. Adds extra convenience parameters
    """

    ax = sns.histplot(*args, **kwargs)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    if show:
        plt.show()
        
    return ax

def linkage(data, metric='precompute', method='ward', fast=False, **kwargs):
    """Does the same thing with scipy.cluster.hierarchy.linkage, but it
    handles my desired default parameters, and easily handles the
    usage of a precomputed distance metric.

    fast :
    
        If True, use fastcluster package's implementation of linkage(). Otherwise, use scipy's (default).

        Note: not recommended to use fastcluster unless necessary. My
        empirical tests show that the fastcluster package is about 2x
        faster with method='ward' and precomputed distances for
        ~16K-by-16K.  However, in a direct comparison of Z created by
        scipy, it is not exactly the same -- although discrepancies
        may just be degenerate permutations of the
        leaves. Additionally, I'm not sure if fastcluster is computing
        optimal leaf ordering (this is done by default in scipy, which
        may also exlain why it's slower)
    """
    
    if metric=='precompute':
        data = scipy.spatial.distance.squareform(data, checks=False)
    # else:
    #     raise Exception()

    if fast:
        import fastcluster
        linkage = fastcluster.linkage
    else:
        linkage = scipy.cluster.hierarchy.linkage

    # print('metric', metric)
    # print(data)
    Z = linkage(data, metric=metric, method=method, **kwargs)
    
    return Z


def plot_colors(colors_df, ncols=None):
    """Visualizes the mapping of values to their colors as created by utils.make_category_colors()"""
    
    if ncols is None:
        ncols = 3

    # Remove unused categorical. These will attempted to be colored (which is wasteful of color space) and plotted (which results in an error)
    colors_df = utils.unused_cat(colors_df)

    # Map values in the dataframe to colors
    # - dataframe with colors
    columns = colors_df.columns
#    print(columns)
    colors_df = pd.concat([colors_df,
                           make_category_colors(None, colors_df, fmt='rgb').rename(columns=lambda x : f'{x}_color')],
                           axis=1)
#    display(colors_df)

    fig, axes = subplots(None, ncols, 3, 1, n=colors_df.shape[1], flatten=True)
    for ax, c in zip(axes, columns):
        data = colors_df[[c, f'{c}_color']].drop_duplicates().copy()
        if len(data)<=1:
            continue

        data['height'] = 1
        # display(data)
        # display(data.dtypes)
        # print(c)
        sns.barplot(x=c, y='height', data=data, ax=ax, palette=data.set_index(c)[f'{c}_color'].to_dict())
        ax.tick_params(labelleft=False, left=False, labelrotation=-45, direction='in')
        ax.set(xlabel=None, ylabel=None)
        for x in ['left', 'bottom', 'top', 'right']:
            ax.spines[x].set_visible(False)
#        ax.set_title(c)
        ax.set_ylabel(c)
    for ax in axes[len(columns) : ]:
        ax.remove()

def make_category_colors(names, assignments, palette_name='husl', fmt='hex', na_color=None):
    """Given a categorization of a set of objects (`names`), assign a
    color to each object by their category. Multiple types of
    categorizations are given as a dataframe
    
    assignments : dataframe of objects-by-categories
    
    names : the subset of object to be colored
    
    na_color : the color used for objects with no category (value of nan in `assignments`)

    """

    import seaborn as sns

    if isinstance(assignments, pd.Series):
        assignments = assignments.to_frame()
        
    if names is None:
        assignments = assignments.astype('category')
    else:
        assignments = assignments.loc[names].astype('category')

    colors = pd.DataFrame(index=assignments.index)
    n_classes_dict = OrderedDict()
    for c in assignments.columns:
        assignments[c] = assignments[c].cat.remove_unused_categories()
        n_classes = assignments[c].cat.categories.size
        n_classes_dict[c] = n_classes
        palette = sns.color_palette(palette_name, n_classes)

        if fmt=='hex':
            palette = palette.as_hex()
            if na_color is None:
                na_color = '#ffffff'
        elif fmt=='rgb':
            if na_color is None:
                na_color = (1.,1.,1.)

        colors[c] = [palette[code] if (code != -1) else na_color \
                        for code in assignments[c].cat.codes]

    return colors

def subplots(nrows, ncols, width, height, total_height=None, total_width=None,
             flatten=False, width_ratios=None, height_ratios=None, tight_layout=True, n=None, downsize=False, figsize=None, gridspec_kw=None, **kwargs):
    """Convenient wrapper around plt.subplots(). Calculates figsize
    based on (width, height) of each axes"""
    
    if gridspec_kw is None:
        gridspec_kw = {}

    if width_ratios is not None:
        assert 'width_ratios' not in gridspec_kw
        gridspec_kw['width_ratios'] = width_ratios
    if height_ratios is not None:
        assert 'height_ratios' not in gridspec_kw
        gridspec_kw['height_ratios'] = height_ratios

    if n is not None:
        if ncols is None:
            # Infer number of columns
            ncols = math.ceil(n / nrows)            
        else:
            # Infer number of rows
            nrows = math.ceil(n / ncols)            

            if downsize and (nrows==1):
                # Decrease number of columns becaues `n` is small
                ncols = min(n, ncols * nrows)

    if figsize is None:
        if total_width is None:
            total_width = ncols * width
        if total_height is None:
            total_height = nrows * height
        figsize = (total_width, total_height)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             tight_layout=tight_layout, squeeze=False, gridspec_kw=gridspec_kw,
                             **kwargs)
    if flatten:
        axes = axes.flatten()
    
    return fig, axes

def clustermap(data,
               Z=None,
               method=None,
               figsize=None,
               metric=None,
               colors_ratio=None,
               dendrogram_ratio=None,
               is_affinity=True,
               row_colors=None,
               col_colors=None,
               number_labels=False,
               show_cluster=None,
               show_row_cluster=None,
               show_col_cluster=None,
               verbose_colors=False,
               optimal_ordering=None,
               center=None, cmap=None,
               square=None,
               row_Z=None, col_Z=None,
               verbose=True,
               show=False,
               co_cluster=None, n_clusters=None, random_state=None,
               row_cluster=None, col_cluster=None,
               xticklabels=True,
               yticklabels=True,
               cbar_pos=None,
               convert_row_colors=None,
               **kwargs):
    """
    **** DEPRECATED : See clustermap() *************
    
    Plots a heatmap. Acts like sns.clustermap, but uses my desired
    default paramters, and does basic formatting I typically want

    Notes on runtime:

    ## Seems like biggest time is to plot pixels
    #
    ## Using distances as-is
    # -- 4m 36s, all plasmids, 16827-by-16827
         suggested figsize=(30,30)
    # -- 2m 20s, 12K-by-12K, ward, figsize=20-by-20, row/col_cluster default or False (this setting doesn't matter).
         But you want to let row/col_cluster=True to reorder the leaves 
         suggested figsize=(20,20)
    # -- 34s, 6000-by-6000, ward
    # -- 16s, 6000-by-6000, complete
    ## From scratch, default metric
    # -- 24s for 3000-by-3000
    # -- 3m 12s for 6000-by-6000
    """

    import seaborn as sns

    from scipy.cluster import hierarchy

    if optimal_ordering is None:
        optimal_ordering = False

    if square is None:
        square = False

    if metric is None:
        if square:
            metric = 'precompute'
        else:
            metric = 'euclidean'
        
    if figsize is None:
        if data.shape[0] > 15000:
            figsize = (32,30)
        elif data.shape[0] > 6000:
            figsize = (22,20)
        elif data.shape[0] > 3000:
            figsize = (17, 15)
        elif data.shape[0] > 100:
            figsize = (11,10)
        elif data.shape[0] > 40:
            figsize = (8.75, 8)
        else:
#            figsize = (5.5, 5)
            figsize = (7.5, 7)

    def rearrange_by_Z(data, data_arr, names, Z, row=True):
        """Rearranges the data rows or cols according to a linkage Z"""

        leaves = hierarchy.leaves_list(Z)
        if row:
            # This linkage represents rows
            data, data_arr, names = data.iloc[leaves,:], data_arr[leaves,:], names[leaves]
        else:
            # This linkage represents cols
            data, data_arr, names = data.iloc[:, leaves], data_arr[:,leaves], names[leaves]
            
        return data, data_arr, names


    if isinstance(data, pd.DataFrame):
        # Test if dataframe contains sparse datatypes. If so, then convert to dense
        if any('Sparse' in x.name for x in data.dtypes.values):
            data = data.sparse.to_dense()

        rownames, colnames = data.index.values, data.columns.values
        
        if square and not np.all(rownames==colnames):
            assert Z is None

            # Sync rownames and colnames to make square
            data = data.copy()
            union = data.index.union(data.columns)
            data = data.reindex(union).reindex(columns=union).fillna(0)
            rownames, colnames = data.index.values, data.columns.values

        data_arr = data.values
    else:
        rownames, colnames = np.arange(data.shape[0]), np.arange(data.shape[1])
        data_arr = data
        data = pd.DataFrame(data)

    if verbose:
        print('Plotting shape:', data.shape)

    if square:
        assert data.shape[0]==data.shape[1]
        if Z is not None:
            assert (row_Z is not None) and (col_Z is not None)
            row_Z, col_Z = Z, Z        

    if is_affinity:
        # The specified matrix is affinities/similarities (rather than
        # distances). So invert it, before giving it to linkage()
        data_linkage = 1 - data_arr
    else:
        data_linkage = data_arr

    if show_cluster:
        assert (show_row_cluster is None) and (show_col_cluster is None)
        show_row_cluster, show_col_cluster = True, True

    to_rearrange_rows, to_rearrange_cols = False, False
    if co_cluster:
        ### Do bi-clustering using sklearn's SpectralCoclustering() method
        if verbose:
            print('Calculating co-clustering')

        import sklearn.cluster
        if random_state is None:
            random_state = 0
        if n_clusters is None:
            n_clusters = 20
        model = sklearn.cluster.SpectralCoclustering(n_clusters=n_clusters, random_state=random_state)
        # print('wacka', np.isinf(data_arr).sum())
        # print('wacka', data_arr.dtype)

        # Normalize values
        data_arr = sklearn.preprocessing.StandardScaler().fit_transform(data_arr)

        data_arr = data_arr.astype(np.float64) + 1e-10 # Add a small number to avoid floating point errors (don't know why this happens)
#        return data_arr
        model.fit(data_arr)
        data = data.iloc[np.argsort(model.row_labels_), :].iloc[:, np.argsort(model.column_labels_)]
        data_arr = data_arr[np.argsort(model.row_labels_), :][:, np.argsort(model.column_labels_)]

        assert (row_Z is None) and (col_Z is None) and (not show_row_cluster) and (not show_col_cluster)
        show_row_cluster, show_col_cluster = False, False
    else:
        ### Cluster rows and cols separately
        
        # If either row_Z or col_Z is set, then make sure that
        # row/col_cluster is set to the default None or is not False
        if (row_Z is None) and (row_cluster is None):
            # Default is to cluster rows
            row_cluster = True
        elif row_Z is not None:
            if row_cluster is None:
                row_cluster = True
            else:
                assert row_cluster is not False, "If row_Z is specified, then row_cluster cannot be False"
        if (col_Z is None) and (col_cluster is None):
            # Default is to cluster cols
            col_cluster = True
        elif col_Z is not None:
            if col_cluster is None:
                col_cluster = True
            else:
                assert col_cluster is not False, "If col_Z is specified, then col_cluster cannot be False"

        # print(row_cluster, col_cluster)
        # print(show_row_cluster, show_col_cluster)

        if method is None:
            method = 'ward'

        if row_cluster:
            # Make sure a row linkage is calculated
            if row_Z is None:
                if verbose:
                    utils.tprint('Calculating row linkage with metric={}, method={}, optimal_ordering={}'.format(metric, method, optimal_ordering))
                row_Z = linkage(data_linkage, metric=metric, method=method, optimal_ordering=optimal_ordering)

            if show_row_cluster is None:
                # If row_cluster is True, then default is to show them
                show_row_cluster = True

            if show_row_cluster:
                to_rearrange_rows = True
            else:
                ## If not showing the row dendrogram, then need to internally reorder the rows
                data, data_arr, rownames = rearrange_by_Z(data, data_arr, rownames, row_Z, row=True)
        else:
            assert not show_row_cluster, "Cannot show a clustering when one isn't being calculated"
            show_row_cluster = False

        if col_cluster:
            # Make sure a col linkage is calculated
            if col_Z is None:
                if verbose:
                    utils.tprint('Calculating col linkage with metric={}, method={}, optimal_ordering={}'.format(metric, method, optimal_ordering))
                col_Z = linkage(data_linkage.T, metric=metric, method=method, optimal_ordering=optimal_ordering)

            if show_col_cluster is None:
                # If col_cluster is True, then default is to show them
                show_col_cluster = True

            if show_col_cluster:
                to_rearrange_cols = True
            else:
                ## If not showing the col dendrogram, then need to internally reorder the cols
                data, data_arr, colnames = rearrange_by_Z(data, data_arr, colnames, col_Z, row=False)
                to_rearrange_cols = False
        else:
            assert not show_col_cluster, "Cannot show a clustering when one isn't being calculated"
            show_col_cluster = False

    # Calculate colors for heatmap
    if row_colors is not None:
        row_colors = row_colors.loc[row_colors.index.intersection(data.index)]
        row_colors = utils.unused_cat(row_colors)

        if square:
            assert col_colors is None, "Cannot specify `row_colors` and `col_colors` simultaneously when square==True"
            col_colors = row_colors

        if (convert_row_colors is None) or convert_row_colors:
            plot_colors(row_colors)
            row_colors = make_category_colors(None, row_colors, fmt='rgb')

    if col_colors is not None:
        empty_col_colors = col_colors.reindex(colnames).isna().all()
        if verbose_colors:
            print('These col_colors are all NA across rows: {}'.format(empty_col_colors.index[empty_col_colors]))
            display(col_colors.loc[colnames,~empty_col_colors])

        col_colors = col_colors.loc[col_colors.index.intersection(data.columns)]
        col_colors = utils.unused_cat(col_colors)
        
#        display(col_colors)
        plot_colors(col_colors)
        # print(colnames)
        # display(col_colors)
        # display(empty_col_colors)
        col_colors = make_category_colors(colnames, col_colors.loc[:,~empty_col_colors], fmt='rgb')
        
    if colors_ratio is None:
        ncol_colors = 1 if (col_colors is None) else col_colors.shape[1]
        nrow_colors = 1 if (row_colors is None) else row_colors.shape[1]
        colors_ratio = (1/figsize[0] / nrow_colors, 1/figsize[1] / ncol_colors)
        #print('colors_ratio:', colors_ratio)
        
    if dendrogram_ratio is None:
        dendrogram_ratio = (min(0.1, 2 / figsize[0]), min(0.1, 1 / figsize[1]))

    if cbar_pos=='None':
        cbar_pos = None # A workaround to disable the cbar by setting it to 'None' (the string), rather than None
    elif cbar_pos is None:
        # Set the colorbar to a thin vertical bar
        height = min(0.8, 10/figsize[1])
#        cbar_pos=(0, 0.8 - height, min(2/figsize[0], 0.025), height)
        cbar_pos=(0.02, 0.02, min(2/figsize[0], 0.025), height)
        # print('figsize:', figsize)
        # print('cbar_pos:', cbar_pos)

    if verbose:
        utils.tprint('Figure size:', figsize)
        utils.tprint('colors_ratio:', colors_ratio)
        utils.tprint('dendrogram_ratio:', dendrogram_ratio)
        utils.tprint('cbar_pos:', cbar_pos)
        utils.tprint('row/col_cluster:', row_cluster, col_cluster)
        utils.tprint('show_row/col_cluster:', show_row_cluster, show_col_cluster)
        utils.tprint('row/col_Z is set:', row_Z is not None, col_Z is not None)
        utils.tprint('Plotting')

    fig = sns.clustermap(data,
                         row_cluster=show_row_cluster,
                         col_cluster=show_col_cluster,
                         row_linkage=row_Z,
                         col_linkage=col_Z,
                         figsize=figsize,
                         colors_ratio=colors_ratio,
                         dendrogram_ratio=dendrogram_ratio,
                         row_colors=row_colors,
                         col_colors=col_colors,
                         center=center,
                         cmap=cmap,
                         cbar_pos=cbar_pos,
                         **kwargs
    )
    fig.ax_heatmap.tick_params(axis='x', which='both', labelbottom=xticklabels)
    fig.ax_heatmap.tick_params(axis='y', which='both', labelbottom=yticklabels)

    # # Add a tick to every row
    # fig.ax_heatmap.set_yticks(np.arange(min(fig.ax_heatmap.get_ylim()) + 0.5, max(fig.ax_heatmap.get_ylim()) + 0.5, 1))

    if show:
        if verbose:
            utils.tprint('Showing Figure')
        plt.show()
    else:
        if verbose:
            utils.tprint('Done')

    if to_rearrange_rows:
        if verbose:
            utils.tprint('Rearranging rows')
        data, _, _ = rearrange_by_Z(data, data_arr, rownames, row_Z, row=True)
    if to_rearrange_cols:
        if verbose:
            utils.tprint('Rearranging cols')
        data, _, _ = rearrange_by_Z(data, data_arr, colnames, col_Z, row=False)

    return data, fig, row_Z, col_Z
