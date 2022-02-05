import time
import functools
import subprocess
import os
import tempfile
from statistics import median

import numpy as np
import pandas as pd
from numba import jit
import scipy, scipy.sparse

from mobmess import utils, nb_utils

def mash_sketch(inp, out, kmer=None, sketch=None, threads=None):
    """Runs mash sketch. 

    inp :

       Filename of fasta file

    out:
    
       File PREFIX for output
    
    ### Sketch size 2000
    # 7 sec on epona, sketch size 1000 or 2000
    # 53 sec for 9000 of the refseq genomes, size 2000
    # 36 sec for all refseq, size 2000

    ### Sketch size 100,000
    # 3000 genomes: 17.4s
    # All genomes: 1m 23s (18Gb storage)

    """
    
    if threads is None:
        threads = utils.get_max_threads()
    threads = '-p {}'.format(threads)
    if kmer is not None:
        kmer = '-k {}'.format(kmer)
    else:
        kmer = ''
    if sketch is not None:
        sketch = '-s {}'.format(sketch)
    else:
        sketch = ''

    # Write fasta to separate files
    with tempfile.NamedTemporaryFile('wt') as f:
        utils.write_fasta(inp)

    cmd = """mash sketch {threads} {sketch} -o {out} -l {inp} 2> {out}.log""".format(
        threads=threads, sketch=sketch, kmer=kmer, out=out, inp=inp)        
        
    print(cmd)
#    utils.run_cmd(cmd)

def mash_dist(inp, out=None, inp2=None, pivot=False):
    """
    Runs `mash dist` on a precomputed mash inp.

    ### Inp size 2000
    # 8000 random genomes: 30 s
    # All genomes: 8 min, -d 0.5
    # All genomes: 1 hr 2 min, -d 1.0 (default, where calculates all distances)

    ### Inp size 100000
    # 1000 genomes: 12.8s
    # 3000 genomes: 1m 30s
    # All genomes: 2 hr 30 min
    """

    if threads is None:
        threads = utils.get_max_threads()
    threads = '-p {}'.format(threads)

    if inp2 is None:  inp2 = inp

    # Read stdin directly, and then format output with pandas.read_csv #
    with open(str(inp) + '.log', 'wb') as f:
        cmd = 'mash dist {threads} {inp}.msh {inp}.msh'.format(inp, threads=threads)
        print(cmd)

        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=f)

        mash_dist_long = pd.read_csv(p.stdout,
                                     sep='\t',
                                     nrows=None,
                                     chunksize=1000000,
                                     names=['s1', 's2', 'dist', 'p_value', 'kmers'],
                                     usecols=['s1', 's2', 'dist', 'p_value', 'kmers'],
                                     dtype={'s1' : 'category',
                                            's2' : 'category',
                                            'dist' : np.float32,
                                            'p_value' : np.float32,
                                            'kmers' : 'category'})
        mash_dist_long = utils.better_pd_concat(list(mash_dist_long), ignore_index=True)

        # Change fasta paths into contig names
        mash_dist_long['s1'].cat.categories = [os.path.basename(x).rstrip('.fa').replace('.','_') for x in mash_dist_long['s1'].cat.categories]
        mash_dist_long['s2'].cat.categories = [os.path.basename(x).rstrip('.fa').replace('.','_') for x in mash_dist_long['s2'].cat.categories]    

        # Change distance to similarity
        mash_dist_long['sim'] = 1 - mash_dist_long['dist']
        mash_dist_long.drop(columns='dist', inplace=True)

        p.terminate()

    if pivot:
        mash_dist = utils.sparse_pivot(mash_dist_long, index='s1', columns='s2', values='sim', rettype='sparse', square=True)
    else:
        mash_dist = None

    if out is not None:
        utils.pickle(mash_dist_long, '{}.long.dist.pkl.blp'.format(inp))
        if pivot: 
            utils.pickle(mash_dist, '{}.dist.pkl.blp'.format(inp))

    return mash_dist_long, mash_dist

# def read_fastani(path, fmt='long', rettype=None, nrows=None):
#     """
#     Read either the 'output' or 'output.matrix' files produced by the FastANI tool
#     """

#     if fmt=='long':
#         df = pd.read_csv(path,
#                          sep='\t',
#                          header=None,
#                          nrows=nrows,
#                          names=['s1', 's2', 'ANI', 'matches', 'total'],
#                          dtype={'s1' : 'category',
#                                 's2' : 'category',
#                                 'ANI' : np.float32,
#                                 'matches' : np.int32,
#                                 'total' : np.int32})

#         df['s1'].cat.categories = [os.path.basename(x).replace('.fa', '').replace('.','_') for x in df['s1'].cat.categories]
#         df['s2'].cat.categories = [os.path.basename(x).replace('.fa', '').replace('.','_') for x in df['s2'].cat.categories]
#         df['s1'].cat.reorder_categories(np.sort(np.asarray(df['s1'].cat.categories)), inplace=True)
#         df['s2'].cat.reorder_categories(np.sort(np.asarray(df['s2'].cat.categories)), inplace=True)

#         if rettype=='wide':
#             df = sparse_pivot(df, index='s1', columns='s2', values='ANI', square=True)
#     else:
#         with open(path) as f:
#             seq_names = []
#             for i, line in enumerate(f.readlines()):
#                 if i==0:
#                     n = int(line)
#                     matrix = np.zeros((n,n), np.float64)
#                 else:
#                     line = line.rstrip().split('\t')
#                     seq_names.append(line[0])
#                     matrix[i-1,:len(line[1:])] = [0.0 if x=='NA' else float(x) for x in line[1:]]
#                 if (i%100)==0:
#                     print(i, end=',')

#         df = pd.DataFrame(matrix, index=seq_names, columns=seq_names)

#     return df


def read_fastANI(similarities_path, file_2_name=None, names=None, nrows=None):
    """
    Read table of pairwise similarities from fastANI
    """

    # Parse ANI results from fastANI    
    similarities = pd.read_csv(similarities_path, sep='\t', header=None, nrows=nrows,
                         names=['query','reference','ANI','fragments_matched','total_query_fragments'])
    similarities = similarities.astype({'query' : 'category', 'reference' : 'category'})

    if file_2_name is None:
        def rename(x):
            return os.path.basename(x.split('.fa')[0])
    else:
        file_2_name = {str(x) : y for x, y in file_2_name.items()}

        def rename(x):
            return file_2_name[x]

    similarities['query'].cat.categories = [rename(x) for x in similarities['query'].cat.categories]
    similarities['reference'].cat.categories = [rename(x) for x in similarities['reference'].cat.categories]

    contig_categ = similarities['query'].cat.categories.union(similarities['reference'].cat.categories)
    if names is not None:
        assert contig_categ.isin(names).all()
        contig_categ = names

    similarities['query'].cat.set_categories(contig_categ, inplace=True)
    similarities['reference'].cat.set_categories(contig_categ, inplace=True)

    return similarities

def threshold_ANI(similarities, similarity_threshold, min_alignment_fraction,
                      both_directions_alignment_fraction=None, 
                      calculate_coverage=None,
                      do_remove_unused_categories=None):
        
    if do_remove_unused_categories is None:
        do_remove_unused_categories = False

    if both_directions_alignment_fraction is None:
        both_directions_alignment_fraction = True

    print('ANI pairs before threshold: {:n}'.format(len(similarities)))

    if (calculate_coverage is True) or ((calculate_coverage is None) and ('coverage' not in similarities.columns)):
        # Compute coverage as the fraction of fragments mapped by the total of query fragments (note that this is asymmetrical, depending on what is the query vs reference)
        similarities['coverage'] = similarities['fragments_matched'] / similarities['total_query_fragments']
    
    # Filter based on min_alignment_fraction
    similarities = similarities[(similarities['coverage'] >= min_alignment_fraction) & (similarities['ANI'] >= similarity_threshold)]

    if both_directions_alignment_fraction:
        # Only keep pairs of contigs, where min_alignment_fraction is satisfied in BOTH directions A-->B and B-->A
        # similarities = similarities.merge(similarities.rename(columns={'query':'reference', 'reference':'query'})[['query','reference']],
        #                                   how='inner', on=['query','reference'])
        similarities = utils.better_merge(similarities,
                                          similarities[['query','reference']].rename(columns={'query':'reference', 'reference':'query'}),
                                          how='inner', on=['query','reference'])

    if do_remove_unused_categories:
        similarities = remove_unused_categories(similarities)

    print('ANI pairs after threshold: {:n}'.format(len(similarities)))

    return similarities


def derep_sources(similarities, similarity_threshold, min_alignment_fraction,
                  verbose=False, circular_df=None, lengths=None, fasta_dict=None, output=None,
                  plasmid_scores=None):
    """
    Given the similarities between contigs (a directed graph), de-replicate a set of contigs by

    (1) Calculate the strongly connected components, i.e. a set of vertices where every pair of vertices A and B has BOTH paths A-->B and B-->A
    (2) Create a meta-graph where nodes are strongly connected components, and directed edges between components are made if there were directed edges between the components' individual vertices
    (3) Calculate the "source" nodes in this meta-graph, i.e. nodes that have no parents (this includes nodes with no neighbors at all)
    
    Params
    ------

    similarities :

        Dataframe of contig similarities. Already thresholded for ANI and coverage

    Returns
    --------

    circular_df :
    
        Dataframe describing the detection and circularity of each contig (when recruiting a metagenome to its own assembly)

                   contig      sample    detect         cov  circular
        0          AST0002_000000012700     AST0002  1.000000  744.743600      True
        1          AST0002_000000015498     AST0002  1.000000  264.243470      True
        2          AST0002_000000014415     AST0002  1.000000  141.373380      True
        3          AST0002_000000010621     AST0002  1.000000   87.924160      True
        4          AST0002_000000008030     AST0002  1.000000  102.475750      True

    """

    # Remove pairs of sequences that don't meet the ANI similarity thresholds
    similarities = threshold_ANI(similarities, similarity_threshold, min_alignment_fraction, both_directions_alignment_fraction=False)

    # Create a scipy.sparse matrix to represent the contig-2-contig network. Matrix entries are set to the 'full_ANI' column,
    # which is the global version of ANI.
    # 
    # - query is "child" (or smaller) contig
    # - reference is the "parent" (or larger) contig
    sp_fullANI, rownames, colnames = utils.sparse_pivot(similarities, index='query', columns='reference', values='full_ANI', square=True, rettype='spmatrix', directed=True)
    sp = sp_fullANI > 0

    # (Update 1/31/22): Make sure the matrix diagonal is set to 1
    #
    # It's possible that a sequence does not have an edge to itself if it has large repetitive regions. Such repeats won't be
    # aligned to anything after applying MUMmer's delta-filter, and so the coverage/identity will be less than 100%. This can
    # happen in reference plasmids that have repeat regions. However, all metagenomic-assembled predicted plasmids that we've
    # examined are always 100% aligned to themselves (i.e. there's an edge), maybe because it's hard to assemble sequences with
    # big repeats (a typical assembly algorithm would break that sequence into smaller contigs).
    utils.setdiag(sp, 1)    

    # Create a igraph.Graph object that represents the contig-by-contig network
    g = utils.sparse_pivot(similarities, index='query', columns='reference', square=True, rettype='ig', directed=True)

    # Do some sanity checks on correctness of the graph
    assert np.all(rownames == g.vs['name']).all()
    if verbose:
        print('igraph.Graph object:')
        print(g.summary())

    # Identify the strongly connected components 
    clusters = list(g.components(mode='strong'))
    if verbose:
        print('Strongly connected components:', len(clusters))

    # Calculate the in-degree of every contig in its cluster (i.e. how many children it has in the cluster)
    clusters_degrees = pd.Series(utils.as_flat(utils.sp_mask(sp.astype(np.int), blocks=clusters).T.dot(np.ones(sp.shape[0], dtype=np.int))), rownames)
    clusters_sumANI = pd.Series(utils.as_flat(utils.sp_mask(sp_fullANI, blocks=clusters).T.dot(np.ones(sp.shape[0]))), rownames)

    # Create contigs-by-clusters network (i.e. cluster membership matrix)
    row, col = zip(*[(v, i) for i, v_list in enumerate(clusters) for v in v_list])
    contig_2_clusters_sp = scipy.sparse.coo_matrix((np.ones(len(row), np.float32), (row, col)))

    # Create clusters-by-clusters directed network (with child-to-parent orientation)
    clusters_2_clusters = contig_2_clusters_sp.T.dot(sp.astype(np.float32)).dot(contig_2_clusters_sp).astype(np.int32)

    # Identify the 'source' nodes in the cluster-by-cluster network. A source node is defined as having no outgoing edges. Note:
    # I got confused by graph theory terminology... in retrospect, I should have called these 'sink' instead of 'source' nodes.
    # 
    # -- Source nodes represent clusters of 'maximal' contig. A maximal contig is the largest version of that sequence in the
    # input data, i.e. it is not contained within any larger contig.
    is_source = utils.as_flat((clusters_2_clusters > 0).sum(1)==1)
    sources = is_source.nonzero()[0]

    # Create pd.Series of contigs' nucleotide lengths
    if lengths is None:
        assert fasta_dict is not None
        lengths = pd.Series({x : len(seq) for x, seq in fasta_dict.items()})
    lengths = pd.Series(lengths)
    
    ##################################
    # Format circularity information #

    if circular_df is None:
        # Assume everything is circular
        circular_df = pd.Series(True, rownames)
    # If not already boolean, verify that values are integers 0 or 1, and then cast to boolean
    if not pd.api.types.is_bool_dtype(circular_df):
        assert all(x in [0,1] for x in circular_df.drop_duplicates().values), "Values for circularity were not boolean (True/False) or boolean integers (0/1)"
        circular_df = circular_df.astype(bool)
    # Create a boolean indicator vector for whether a contig is circular
    tmp = circular_df.loc[rownames].values.astype(np.int32)
    # Calculate number of member contigs that are circular
    clusters_ncirc = utils.as_flat(contig_2_clusters_sp.T.dot(tmp.reshape(-1,1))).astype(np.int32)

    ###########################################
    # Get a dataframe describing each cluster #
    # (i.e. strongly connected components)    #

    # Calculate the reachability from every component other components
    clusters_2_clusters_paths = utils.reachability(clusters_2_clusters)
    # tmp = utils.create_ig(clusters_2_clusters).neighborhood(mode='out', order=clusters_2_clusters.shape[0])
    # indices = np.concatenate(tmp)
    # clusters_2_clusters_paths = scipy.sparse.csr_matrix((np.ones(indices.size, np.bool_), indices, np.append(0, np.cumsum([len(x) for x in tmp]))))

    # Calculate weak components ("meta clusters")
    meta_clusters = list(utils.create_ig(clusters_2_clusters, directed=True).components('weak'))
    meta_clusters_assignments = np.empty(clusters_2_clusters.shape[0], np.int)
    for i, x in enumerate(meta_clusters):
        meta_clusters_assignments[x] = i

    # Create `clusters_df`: a pd.DataFrame describing each cluster
    clusters_df = pd.DataFrame({
        'meta_cluster' : meta_clusters_assignments,
        'sources_parents' : utils.as_flat((clusters_2_clusters[:,sources] > 0).sum(1)),
        'sources_ancestors' : utils.as_flat((clusters_2_clusters_paths[:, sources]).sum(1)),

        # Boolean indicator if this is a source cluster
        'is_source' : is_source,
        # Number of contigs in this cluster that are circular
        'members_circ' : clusters_ncirc,

        'sources_circ' : utils.as_flat((clusters_2_clusters_paths[:,sources[clusters_ncirc[sources] > 0]] > 0).sum(1)),

        # Classifies every cluster as 'source', 'backbone', or 'fragment'
        'cluster_type' : ['source' if is_source else ('backbone' if (circ > 0) else 'fragment') for is_source, circ in zip(is_source, clusters_ncirc)],

        # Tuple of contigs in this cluster
        'members' : map(tuple, np.split(rownames[np.concatenate(list(clusters))].values, np.cumsum([len(c2) for c2 in clusters])[:-1])),
        # Tuple of lengths of contigs in this cluster
        'members_lengths' : map(tuple, np.split(lengths.loc[rownames[np.concatenate(list(clusters))]].values, np.cumsum([len(c2) for c2 in clusters])[:-1]))
    })
    # Reduce tuple of contig lengths to a triple of statistics (min, median, max)
    clusters_df['members_lengths'] = [(min(x), median(x), max(x)) for x in clusters_df['members_lengths']]
    clusters_df['members_lengths_min'] = [x[0] for x in clusters_df['members_lengths']]
    clusters_df['members_lengths_max'] = [x[-1] for x in clusters_df['members_lengths']]

    # Annotate the min/median/max model score of plasmids in a cluster
    if plasmid_scores is not None:
        clusters_df['members_scores'] = list(map(tuple, np.split(plasmid_scores.loc[rownames[np.concatenate(list(clusters))]].values, np.cumsum([len(c2) for c2 in clusters])[:-1])))
        clusters_df['members_scores'] = [(min(x), median(x), max(x)) for x in clusters_df['members_scores']]
        clusters_df['members_scores_min'] = [x[0] for x in clusters_df['members_scores']]
        clusters_df['members_scores_max'] = [x[-1] for x in clusters_df['members_scores']]

    if verbose:
        print('Cluster types')
        utils.display(clusters_df['cluster_type'].value_counts())

    # Create `contigs_df`: a pd.DataFrame describing each contig
    contigs_df = pd.Series({rownames[v] : i for i, v_list in enumerate(clusters) for v in v_list}).to_frame('cluster')
    contigs_df['idx'] = np.arange(len(contigs_df))
    contigs_df = functools.reduce(lambda x, y: x.merge(y, left_index=True, right_index=True, how='outer'),
        [contigs_df,
         lengths.to_frame('length'),
         clusters_degrees.to_frame('nchildren_in_cluster'),
         clusters_sumANI.to_frame('avgANI'),
         circular_df.reindex(rownames).to_frame('circular').fillna(False).rename_axis('contig')])
    contigs_df = contigs_df.rename_axis('contig').reset_index()

    # For each contig, add info about the cluster it is in
    clusters_df = functools.reduce(lambda x, y: x.merge(y, on='cluster', how='outer'),
                 [   # Cluster name
                     clusters_df.rename_axis('cluster').reset_index(),
                     # Cluster size
                     contigs_df['cluster'].value_counts().to_frame('cluster_size').rename_axis('cluster').reset_index(),
                     # Representative: longest contig in a cluster
                     contigs_df.sort_values(['cluster', 'length', 'nchildren_in_cluster'], ascending=False).drop_duplicates(['cluster'])[['contig','cluster']].rename(columns={'contig':'rep_longest'}),
                     # Representative: contig with the most children contigs in a cluster
                     contigs_df.sort_values(['cluster', 'nchildren_in_cluster', 'length'], ascending=False).drop_duplicates(['cluster'])[['contig','cluster']].rename(columns={'contig':'rep_most_children'}),
                     # Representative: contig with the highest average ANI to other contigs in a cluster
                     contigs_df.sort_values(['cluster', 'avgANI', 'nchildren_in_cluster', 'length'], ascending=False).drop_duplicates(['cluster'])[['contig','cluster']].rename(columns={'contig':'rep_central'}),
                     # Representative: contig that is circular AND the highest average ANI to other contigs in a cluster
                     contigs_df[contigs_df['circular']].sort_values(['cluster', 'avgANI', 'nchildren_in_cluster', 'length'], ascending=False).drop_duplicates(['cluster'])[['contig','cluster']].rename(columns={'contig':'rep_circular_central'})
                 ])

    # Get mapping of a cluster to all of its backbone clusters (including itself, if it is a backbone)
    backbone = clusters_df.loc[clusters_df['cluster_type']=='backbone', 'cluster'].values
    i, j = clusters_2_clusters_paths[backbone, :].nonzero()
    clusters_2_backbones = pd.DataFrame({'backbone' : backbone[i], 'cluster' : j})
    clusters_2_backbones = clusters_2_backbones.merge(clusters_df[['cluster', 'rep_central']].rename(columns={'rep_central' : 'backbone_rep_central'}), on='cluster')
    clusters_2_backbones = clusters_2_backbones.groupby('cluster').agg(
        backbones=pd.NamedAgg('backbone', tuple),
        backbones_rep_central=pd.NamedAgg('backbone_rep_central', tuple))
    # Append dataframe of empty tuples for clusters with no backbones
    empty = pd.DataFrame(index=clusters_df.index.difference(clusters_2_backbones.index))
    for c in clusters_2_backbones.columns:
        empty[c] = [tuple([]) for x in empty.index]
    clusters_2_backbones = pd.concat([clusters_2_backbones, empty]).sort_index().rename_axis('cluster').reset_index()
    assert len(clusters_df) == len(clusters_2_backbones)
    # Merge this backbone info back into clusters_df
    clusters_df = clusters_df.merge(clusters_2_backbones, on='cluster')

    # Create a new cluster type that has four categories:
    # -- fragment and backbone stay the same.
    # -- source clusters are renamed "compound", if it is part of a system (i.e. has a backbone),
    # -- or "maximal_not_in_system" otherwise
    clusters_df['cluster_four_types'] = [('maximal_not_in_system' if len(b)==0 else 'compound') if (c=='source') else c for c, b in clusters_df[['cluster_type', 'backbones']].to_records(index=False)]

    # Merge this aggregate info back into contigs_df
    contigs_df = contigs_df.merge(clusters_df[['meta_cluster', 'cluster_size', 'cluster_type', 'cluster_four_types', 'rep_longest','rep_most_children','rep_central', 'rep_circular_central', 'backbones', 'backbones_rep_central']],
                                  left_on='cluster', right_index=True).set_index('contig')

    contigs_df = contigs_df.sort_values('idx')

    contigs_df['avgANI'] = contigs_df['avgANI'] / contigs_df['cluster_size']


    #############
    # Save data #

    if output is not None:
        # Save info about every contig
        contigs_df.to_csv('{}.contigs.txt'.format(output), sep='\t', header=True, index=True)
        utils.pickle(contigs_df, '{}.contigs.pkl.blp'.format(output))

        # Save info about every cluster
        tmp = clusters_df.copy()
        tmp['members'] = tmp['members'].apply(lambda x: '|'.join(x))
        tmp['members_lengths'] = tmp['members_lengths'].apply(lambda x: '|'.join(map(str,x)))
        if 'members_scores' in tmp.columns:
            tmp['members_scores'] = tmp['members_scores'].apply(lambda x: '|'.join(map(str,x)))
        clusters_df.to_csv('{}.clusters.txt'.format(output), sep='\t', header=True, index=True)
        utils.pickle(clusters_df, '{}.clusters.pkl.blp'.format(output))

        # # Write fasta of representative contigs for all non-source and backbone clusters
        # assert fasta_dict is not None
        # reps = clusters_df.loc[clusters_df['cluster_type'].isin(['source', 'backbone']), 'rep_central'].values
        # if verbose:
        #     print('Representative contigs:', len(reps))
        # utils.write_fasta(utils.subset_dict(fasta_dict, reps), '{}.reps.fa'.format(output))
        
        # # Write fasta of representative CIRCULAR contigs for all backbone clusters
        # reps = clusters_df.loc[clusters_df['cluster_type'].isin(['backbone']), 'rep_central'].values
        # utils.write_fasta(utils.subset_dict(fasta_dict, reps), '{}.backbones.reps_circular.fa'.format(output))

        # Save cluster_2_clusters
        utils.pickle(clusters_2_clusters, '{}.clusters_2_clusters.pkl.blp'.format(output))

        # Save cluster_2_clusters_paths
        utils.pickle(clusters_2_clusters_paths, '{}.clusters_2_clusters_paths.pkl.blp'.format(output))

    return contigs_df, contig_2_clusters_sp, clusters_2_clusters, clusters_2_clusters_paths, sources, clusters_df, similarities

def view_ani_heatmap(similarities, values,
                     contigs=None, cmap=None, contig_2_clusters=None, cluster=True, center=None, figsize=None):
    """Visualize the ANI between contigs.

    similarities :

        long format dataframe of similarities

    """

    if contigs is not None:
        # Get the ANI between the source contigs
        tmp = utils.remove_unused_categories(similarities[utils.categorical_isin(similarities['query'], contigs) & utils.categorical_isin(similarities['reference'], contigs)])
    else:
        tmp = utils.remove_unused_categories(similarities)

    if values=='min_ani_cov':
        tmp['min_ani_cov'] = np.minimum(tmp['ANI'] / 100, tmp['coverage'].values)
    # elif values=='full_ANI':
    #     tmp['full_ANI'] = tmp['full_ANI'] / 100
    else:
        assert values in similarities.columns
#        raise Exception()

    utils.tprint('Pivoting long to wide')
    tmp = utils.sparse_pivot(tmp, index='query', columns='reference', values=values, rettype='dense_df')

    if contig_2_clusters is not None:
        row_colors = utils.make_category_colors(None, contig_2_clusters.loc[tmp.index, ['cluster', 'cluster_is_source', 'circular']])
    else:
        row_colors = None

    if center is not None:
        cmap = 'bwr'
    else:
        cmap = None

    utils.pheatmap(tmp, row_colors=row_colors, center=center, cmap=cmap, square=False, is_affinity=True,
                   show_linkage=False)

    # if cluster:
    #     utils.tprint('Calculating hierarchical clustering')
    #     row_linkage = scipy.cluster.hierarchy.linkage(tmp.values)
    #     if (len(tmp.index)==len(tmp.columns)) and (tmp.index==tmp.columns).all():
    #         col_linkage = row_linkage
    #     else:
    #         col_linkage = scipy.cluster.hierarchy.linkage(tmp.values.T)
    # else:
    #     row_linkage, col_linkage = None, None

    # utils.tprint('Drawing heatmap')
    # sns.clustermap(tmp, row_colors=row_colors, row_cluster=cluster, col_cluster=cluster, row_linkage=row_linkage, col_linkage=col_linkage, center=center, cmap=cmap, figsize=figsize)

def derep_fastANI(similarities, lengths=None, fasta_dict=None):
    # Convert similarity matrix to sparse representation
    sp, rownames, colnames = utils.sparse_pivot(similarities, index='query', columns='reference', values='ANI', rettype='spmatrix', square=True)
    assert np.all(rownames == colnames)

    # Calculate the connected components ("weak" components so directionality does not matter)
    comp = utils.create_ig(sp, weighted=False, directed=True, square=True).components(mode='weak')
    print(comp.summary())
    comp = [rownames[x] for x in comp]

    # Take the longest sequence in each component
    if lengths is None:
        assert fasta_dict is not None
        lengths = pd.Series({x : len(seq) for x, seq in fasta_dict.items()})
    comp_reps = pd.concat([lengths[c].nlargest(1) for c in comp])

    # Write table of components (2-column format of (representative, member) pairs)
    comp = pd.DataFrame([(r, c) for c_list, r in zip(comp, comp_reps.index) for c in c_list], columns=['representative', 'member'])
    print('{} representatives out of {} plasmids'.format(comp['representative'].nunique(), len(comp)))

    return comp, comp_reps

def parse_and_derep_fastANI(similarities, similarity_threshold, min_alignment_fraction,
                            both_directions_alignment_fraction=None,
                            calculate_coverage=None,
                            output=None, output_dir=None, lengths=None, fasta_dict=None,
                            file_2_name=None, names=None):
    """
    both_directions_alignment_fraction :

        If True, only keep pairs of contigs, where min_alignment_fraction is satisfied in BOTH directions A-->B and B-->A
    """

    if (output is None) and (output_dir is not None):
        output = output_dir / 'components_sim_{}_align_frac_{}'.format(similarity_threshold, min_alignment_fraction)

    if os.path.exists(str(similarities)):
        similarities = read_fastANI(similarities, file_2_name=file_2_name, names=names)

    similarities = threshold_ANI(similarities, similarity_threshold, min_alignment_fraction,
                                 calculate_coverage=calculate_coverage,
                                 both_directions_alignment_fraction=both_directions_alignment_fraction,
                                 do_remove_unused_categories=True)

    comp, comp_reps = derep_fastANI(similarities, lengths=lengths, fasta_dict=fasta_dict)

    # Write table mapping contigs to their representatives
    if output is not None:
        comp.to_csv('{}.txt'.format(output), sep='\t', header=True, index=False)

    # Write fasta file of just the component representatives
    if output is not None:
        assert fasta_dict is not None    
        utils.write_fasta({x : fasta_dict[x] for x in comp_reps.index}, '{}.reps.fa'.format(output))

    return similarities, comp

def read_derep(comp_path):
    """Reads the table of contigs-->representatives mapping produced by parse_and_derep_fastANI"""

    grouping_df = pd.read_table(comp_path)
    grouping = grouping_df.set_index('member')['representative'].astype('category').cat.codes
    reps = grouping_df['representative'].unique()
    return grouping_df, grouping, reps

def fastani_set_full_ani(similarities):
    """Computes the "full" sequence similarity, which takes into account the coverage"""

    similarities['ANI_full'] = similarities['ANI'] * similarities['fragments_matched'] / similarities['total_query_fragments']    

def fastani_long_2_wide(similarities, values=None, rettype='sparse'):
    """Convert long-format fastANI similarities to wide-format"""
    
    if values is None:
        values = 'ANI'
    
    return utils.sparse_pivot(similarities, index='query', columns='reference', values='ANI', square=True, rettype=rettype, directed=True)

def run_fastANI(fasta_df, output_dir, k=None, fragLen=None, minFraction=None, matrix=None, threads=None):
    # Create fasta.txt that is a 2-column table of fasta files
    fasta_df[['path']].to_csv(output_dir / 'fasta.txt', sep='\t', header=False, index=False)

    if k is None:
        k = 16
    if fragLen is None:
        fragLen = 1000
    if minFraction is None:
        minFraction = 0.25
    if threads is None:
        threads = get_max_threads()

    prefix = 'fastANI_k{:d}_l{:d}_f{}'.format(k, fragLen, minFraction)

    ## Run fastANI
    ## Note: running the command doesn't seem to work with utils.run_cmd(). Try running in a shell
#    cmd = """cd {dir} && time fastANI --ql fasta.txt --rl fasta.txt -k {kmer:d} --fragLen {fragLen:d} --minFraction {minFraction} -t {t:d} -o fastANI_similarities --matrix > fastani.log 2>&1""".format(
    cmd = """cd {dir} && time fastANI --ql fasta.txt --rl fasta.txt -k {kmer:d} --fragLen {fragLen:d} --minFraction {minFraction} -t {t:d} -o {prefix}_similarities {matrix} > {prefix}.log 2>&1""".format(
        dir=output_dir, kmer=k, fragLen=fragLen, minFraction=minFraction, t=threads, prefix=prefix, matrix='--matrix' if matrix else '')
    print(utils.make_cmd(cmd, env='anvio-6.2'))

    utils.run_cmd(cmd, tee=False, env='anvio-6.2')




##########
# MUMMER #

def run_mummer(fasta, verbose=False, minmatch=None):
    """
    Runs MUMMER. This is so far only used for genome view.

    It's not used for dereplication. Dereplication is used with custom scripts in Run_plasmid_classifier_Dereplicate_predictions

    TODO : does not run mummer_fast_deltafilter() yet
    """

    if minmatch is None:
        minmatch = 16
    
#    with tempfile.NamedTemporaryFile('wt') as infile, tempfile.NamedTemporaryFile('wt') as outfile:
#    with tempfile.NamedTemporaryFile('wt', delete=False) as infile, tempfile.NamedTemporaryFile('wt', delete=False) as outfile:

#    with tempfile.TemporaryDirectory() as d:
    with utils.TemporaryDirectory() as d:
        infile = os.path.join(d, 'in.fa')
        outfile = os.path.join(d, 'out')
        utils.write_fasta(fasta, infile)        

        # print(infile)
        # print(outfile)
        utils.run_cmd("""nucmer --maxmatch --minmatch={minmatch} -p {outfile} -t 252 {infile} {infile}""".format(
            infile=infile, outfile=outfile, minmatch=int(minmatch)),
                      env='mummer', debug=False, verbose=verbose)
        aln_blocks = read_mummer_aln_blocks(outfile + '.delta')

    return aln_blocks

def run_nucmer(fasta, output, threads=None):
    """
    Run nucmer, use `maxmatch` which allows for mapping of non-unique "anchors"
    NOTE: the uniqueness of an anchor seems to be based on its presence across all sequences in a mult-sequence fasta file.
          i.e. strings that occur in different plasmids (even if those plasmids are nearly identical) will be considered non-unique, and so won't be matched at all
          -- this is why you should use `maxmatch`. It's slower, but it seems to be faster than separating plasmids in a script to do a bunch of one-vs-one comparisons without --maxmatch
             (which is what dRep does, and I'm guessing pyani does too). This is very slow because of the overhead, whereas letting nucmer handle multithreading internally
             seems to be faster even with `maxmatch`

    ~49 min on all predicted plasmids with 252 cores
    """

    if threads is None:
        threads = utils.get_max_threads()

    cmd = f"nucmer --maxmatch --minmatch=16 -p {output} -t {threads} {fasta} {fasta}"
    utils.run_cmd(cmd, verbose=True)

def mummer_fast_deltafilter(mummer_delta_file, outfile, n_jobs=None, delta_cmd=None, tmp=None):
    """
    Runs MUMMER's delta-filter on every pair of alignments separately.
    
    Does this by writing separate files to /dev/shm

    ~1 hr 45 min on all predicted plasmids using all cores
    """
    
    if n_jobs is None:
        n_jobs = utils.get_max_threads()

    utils.tprint('Reading delta file')
    with open(mummer_delta_file) as f:
        aln_blocks = f.read().split('>')
        header, aln_blocks = aln_blocks[0], aln_blocks[1:]

        # Replace the first line with dummy  line 'a a'
        old_header = header
        header = header.splitlines()
        header[0] = 'a a'
        header = '\n'.join(header) + '\n'

    if delta_cmd is None:
        delta_cmd = 'delta-filter'

    utils.tprint('Breaking up delta file and filtering')
    with utils.TemporaryDirectory(tmp) as f:
        new_blocks, p_list, outfile_chunk_list = [], [], []
        for chunk_i, chunk_list in enumerate(utils.to_chunks(aln_blocks, k=n_jobs)):
            utils.tprint(chunk_i, end=',')
            outfile_chunk = os.path.join(f, '{}_filter'.format(chunk_i))

            # Create a bash script with a bunch of commands
            cmd_list = []
            for i, block in enumerate(chunk_list):
                infile = os.path.join(f, '{}_{}'.format(chunk_i, i))
                with open(infile, 'wt') as g:
                    g.write(header + '>' + block)
                cmd_list.append('{0} -q -r {1} >> {2}'.format(delta_cmd, infile, outfile_chunk))
            infile_script = os.path.join(f, '{}.sh'.format(chunk_i))
            with open(infile_script, 'wt') as g:
                g.write('#!/bin/bash\n')
                for cmd in cmd_list:
                    g.write(cmd + '\n')
            p = utils.run_cmd('bash {}'.format(infile_script), shell=True, tee=False, wait=False)        

            p_list.append(p)
            outfile_chunk_list.append(outfile_chunk)

        utils.tprint(f'Waiting for {len(p_list)} chunks:')
        for chunk_i, p in enumerate(p_list):
            utils.tprint(chunk_i, end=',')
            p.wait()

        utils.tprint('Reading and concatenating output files')
        with open(outfile, 'wt') as outfile_f:
            outfile_f.write(old_header)
            for outfile_chunk in outfile_chunk_list:
                with open(outfile_chunk, 'rt') as g:
                    x = g.read().replace(header, '')
                    outfile_f.write(x)
    utils.tprint('Done')


def read_mummer_aln_blocks(maxmatch_file, output=None, tmp=None, delete_tmp=None, filename_prefix=None):
    """ Alternative way to parse .delta files from MUMMER
     - Strategy: De-interleave the '>' headers and alignment block info into two files.
     -           Read each file with pd.read_csv, then concatenate them together.

    MUMMER returns 1-based closed intervals. Convert this to 0-based half-open (,] intervals

    ~ 9 min on all predicted plasmids (Consider loading pre-saved results)
    """

    with utils.TemporaryDirectory(tmp, post_delete=delete_tmp) as tmp:

        if filename_prefix is None:
            filename_prefix = 'mummer_align.delta'

        fname = tmp / f'{filename_prefix}_core'
        fname_headers = tmp / f'{filename_prefix}_headers'
        fname_blocks = tmp / f'{filename_prefix}_blocks'

        # Removes the trailing info for each alignment block. And adds line numbers. E.g. convert this:
        # 
        # >ISR0835_000000001116 ISR0091_000000003656 8458 8481
        # 1 8458 23 8481 3 3 0
        # -1190
        # 0
        # >ITA0001_000000000070 ISR0091_000000003656 8475 8481
        # 1 8475 3 8481 5 5 0
        # -1210
        # -4238
        # -1
        # -1
        # 0
        #
        # to this:
        # 
        # 1  >ISR0835_000000001116 ISR0091_000000003656 8458 8481
        # 2  1 8458 23 8481 3 3 0
        # 3  >ITA0001_000000000070 ISR0091_000000003656 8475 8481
        # 4  1 8475 3 8481 5 5 0
        utils.run_cmd("awk -F' ' '{if ((NF==7) || ($1 ~ /^>/)) print $0}' %s | nl > %s" % (maxmatch_file, fname), verbose=True)

        # Separate the alignment headers (starts with '>') and the alignment info into separate files,
        # in order to read them faster as homogenous dtype tables with pd.read_csv(). Finally, merge them back together.
        utils.run_cmd(f"grep '>' {fname} > {fname_headers}", verbose=True)
        utils.run_cmd(f"grep -v '>' {fname} > {fname_blocks}", verbose=True)

        headers = pd.read_csv(fname_headers, delim_whitespace=True, header=None, names=['nl', 'ref','query','ref_len','query_len'])
        blocks = pd.read_csv(fname_blocks, delim_whitespace=True, header=None, names=['nl', 'ref_start', 'ref_end', 'query_start', 'query_end', 'errors', 'mismatches', 'nonalpha'])
        headers = headers.astype({'nl' : int})
        blocks = blocks.astype({'nl' : int})
        aln_blocks = pd.merge_asof(blocks, headers, on='nl', direction='backward').drop(columns=['nl'])

        aln_blocks['ref'] = aln_blocks['ref'].str[1:]
        aln_blocks = aln_blocks.astype({'query':'category', 'ref':'category',
                                        **{k : np.int32 for k in ['query_start', 'query_end', 'ref_start', 'ref_end', 'errors', 'mismatches', 'nonalpha', 'query_len', 'ref_len']}})    

    # print("awk -F' ' '{if ((NF==7) || ($1 ~ /^>/)) print $0}' out_maxmatch.delta > %s" % fname)

    # aln_blocks['query_start'] -= 1
    # aln_blocks['ref_start'] -= 1

    if output is not None:
        utils.pickle(aln_blocks, output)

    return aln_blocks


@jit(nopython=True)
def get_query_coverage_buggy(start, end):
    """
    *** DEPRECATED AND BUGGY:  THIS HAS A BUG  ***
    *** If there is any nucleotide position where 3 or more (start,end) intervals cover, then this function does not appropriately count the number of unique positions covered. ***
    *** New function is get_query_coverage() ***

    Helper function to get_query_coverage_multiquery.

    Handles the alignment blocks for only a single pair of (query, reference) sequences
    """

#    print(start, end)

    # Total number of unique bases covered
    cov_uniq = 0
    
    # Total number of bases along all alignment blocks (might repeat over same coordinates on genomes)
    cov = 0
    
    # # Iterate over alignment blocks
    # prev_s, prev_e = 0, 0
    # for s, e in zip(start, end):

        # cov_add = (e - s + 1)
        # cov += cov_add
        # # Account for situation when start is before the end of the last alignment block
        # cov_add += (s - prev_e - 1) * (s <= prev_e)
        # # Account for situation when this block is subsumed completely within the last block
        # cov_add = max(0, cov_add)
        # cov_uniq += cov_add
        # prev_s, prev_e = s, e

    # Iterate over alignment blocks
    for i in range(len(start)):
        s, e = start[i], end[i]

        cov_add = e - s
        cov += cov_add

#        print(s, e, cov_add)

        # Scan forward for overlap
        for j in range(i+1, len(start)):
            next_s, next_e = start[j], end[j]
            if next_s >= e:
                break
            # Subtract overlap
#            print('Subtracting', next_s, next_e, '|', e - next_s, e - next_e, '|', (e - next_s) - max(0, e - next_e))
            cov_add -= (e - next_s) - max(0, e - next_e)
#            print('cov_add:', cov_add)

        cov_uniq += cov_add
        
    return cov, cov_uniq

@jit(nopython=True)
def get_query_coverage(start, end):
    pos = np.append(start, end)
    idx = np.argsort(pos)
    pos = pos[idx]

    # Boolean indicator: True if position represents the start of a read. False if the end of a read
    is_start = idx < start.size

    length = max(end)

    cov, _, _ = nb_utils.get_coverage_nb(pos, is_start, length, get_dense=True, dtype=start.dtype)

    cov_uniq = (cov > 0).sum()
    cov = cov.sum()

    return cov, cov_uniq

@jit(nopython=True)
def get_query_coverage_multiquery(ref, query, start, end, mismatches):
    """Helper function to get_query_coverage_multiquery_py
    
    Calculates coverage, assuming that the starts and ends have been sorted properly
    """

    prev_r, prev_q = ref[0], query[0]
    prev_i = 0
    
    # start, end = np.minimum(start, end), np.maximum(start, end)

    # ## This is no longer needed, because I pre-convert to 0-based indices in read_mummer_aln_blocks()
    # MUMmer uses 1-based closed coordinates. Change this to 0-based half open coordinates
    start = start.astype(np.int32) - 1

    end = end.astype(np.int32)
    ref = ref.astype(np.int64)
    query = query.astype(np.int64)

#    cov_i, cov_arr = 0, np.zeros((10000, 6), np.int32)
    cov_i, cov_arr = 0, np.zeros((10000, 6), np.int64)
    for i in range(len(ref)):
        r, q = ref[i], query[i]
#        print(r, q)
        if (r != prev_r) or (q != prev_q):
#            print('new', r, q, prev_i, i)
            cov, cov_uniq = get_query_coverage(start[prev_i : i], end[prev_i : i])
            if cov_i >= len(cov_arr):
                cov_arr = nb_utils.extend_2d_arr(cov_arr)
#            cov_arr[cov_i, :] = prev_r, prev_q, cov, cov_uniq, mismatches[prev_i : i].sum(), i-prev_i

            cov_arr[cov_i, 0] = prev_r
            cov_arr[cov_i, 1] = prev_q
            cov_arr[cov_i, 2] = cov
            cov_arr[cov_i, 3] = cov_uniq
            cov_arr[cov_i, 4] = mismatches[prev_i : i].sum()
            cov_arr[cov_i, 5] = i-prev_i

            cov_i += 1
            prev_i, prev_r, prev_q = i, r, q

    # Wrap it up
    i = len(ref)
    cov, cov_uniq = get_query_coverage(start[prev_i : i], end[prev_i : i])
#    print('cov_uniq:', cov, cov_uniq)
    if cov_i >= len(cov_arr):
        cov_arr = nb_utils.extend_2d_arr(cov_arr)
#    cov_arr[cov_i, :] = prev_r, prev_q, cov, cov_uniq, mismatches[prev_i : i].sum(), i-prev_i

    cov_arr[cov_i, 0] = prev_r
    cov_arr[cov_i, 1] = prev_q
    cov_arr[cov_i, 2] = cov
    cov_arr[cov_i, 3] = cov_uniq
    cov_arr[cov_i, 4] = mismatches[prev_i : i].sum()
    cov_arr[cov_i, 5] = i-prev_i

    cov_i += 1

#    0 / asdf
            
    cov_arr = cov_arr[:cov_i, :]
    return cov_arr

def get_mummer_query_coverage_py(aln_blocks, output=None):
    """Calculates the overall query coverage and ANI from a set of MUMmer alignment blocks.

    Multiple blocks are potentially collapsed.

    If blocks are overlapping, then the coverage of the unique base
    pairs is calculated. Also, ANI is calculated by dividing the total
    number of matches by the total number of base pairs (including
    repetition over the same base pairs).

    Input is from read_mummer_aln_blocks()

    """

    utils.tprint('Sorting')
    # Set start, end to be min/max
    aln_blocks = aln_blocks.copy()
    start, end = aln_blocks['query_start'].values, aln_blocks['query_end'].values

    # # Before taking the min/max, must convert indices to 1-index closed.
    # # After taking min/max, then revert to 0-indices
    # aln_blocks['query_start'] += 1
    # aln_blocks['query_start'], aln_blocks['query_end'] = nb_utils.nb_minimum_st(start, end), nb_utils.nb_maximum_st(start, end)
    # aln_blocks['query_start'] -= 1

    aln_blocks['query_start'], aln_blocks['query_end'] = nb_utils.nb_minimum_st(start, end), nb_utils.nb_maximum_st(start, end)

    aln_blocks = aln_blocks.sort_values(['ref', 'query', 'query_start'])

    utils.tprint('Condensing')
    df = get_query_coverage_multiquery(aln_blocks['ref'].cat.codes.values, aln_blocks['query'].cat.codes.values, aln_blocks['query_start'].values, aln_blocks['query_end'].values, aln_blocks['mismatches'].values)    
    
    utils.tprint('Processing')
    df = pd.DataFrame(df, columns=['ref', 'query', 'query_cov', 'query_cov_uniq', 'mismatches', 'nblocks'])
    df['ref'] = pd.Categorical.from_codes(df['ref'].values, aln_blocks['ref'].cat.categories)
    df['query'] = pd.Categorical.from_codes(df['query'].values, aln_blocks['query'].cat.categories)
    
    # Calculate ANI by subtracting the number of mismatches
    # -- NOTE that this ANI counts base pairs multiple times if there are overlapping subalignments. There is no easy way to get around this.
    # -- -- Thus, the ANI is calculated by dividing the total length of subalignments, potentially counting base pairs multiple times.
    df['ANI'] = (df['query_cov'] - df['mismatches']) / df['query_cov']
        
    # Calculate coverage as a percentage
    query_len = utils.fast_drop_duplicates(aln_blocks[['query', 'query_len']]).set_index('query')['query_len']
    df['query_cov'] = df['query_cov'].values / utils.fast_loc(query_len, df['query'])
    df['query_cov_uniq'] = df['query_cov_uniq'].values / utils.fast_loc(query_len, df['query'])    

    # print('max:', df['query_cov_uniq'].max())
    assert 0 <= df['query_cov_uniq'].max() <= 1
    assert 0 <= df['query_cov_uniq'].min() <= 1
    assert 0 <= df['ANI'].max() <= 1
    assert 0 <= df['ANI'].min() <= 1

    # Calculate the full ANI
    # -- Assumes 0% identity in unaligned regions.
    # -- Uses 'query_uniq_cov' to know how many UNIQUE base pairs are covered
    df['full_ANI'] = df['ANI'] * df['query_cov_uniq']

    if output is not None:
        utils.pickle(df, output)

    return df


def depropagate_long(graph, row_header='row', col_header='col', value_header='weight'):
    """
    Same as depropagate() but the input graph is a long-format dataframe
    """

    # Convert graph to sparse matrix
    graph, rownames, colnames = utils.sparse_pivot(
        graph, index=row_header, columns=col_header, rettype='spmatrix', values=value_header, square=True)
    graph = graph.tocsr()
    graph_paths = utils.reachability(graph)
    graph = utils.depropagate(graph, graph_paths)
    # Remove self-edges
    utils.setdiag(graph, 0)
    # Convert graph back to long-format dataframe
    graph = utils.sp_to_df(graph, value_header=value_header, rownames=rownames, colnames=colnames,
                           row_header=row_header, col_header=col_header)

    return graph

def depropagate(X, X_paths=None):
    """
    Get the transitive reduction of a directed graph.
    
    X is a child-to-parent adjacency matrix.
    
    X_paths is a child-to-ancestor reachability matrix.
    """
    
    X = X.copy()
    utils.setdiag(X, 0)

    if X_paths is None:
        X_paths = utils.reachability(X)
    else:
        X_paths = X_paths.copy()
    utils.setdiag(X_paths, 0)
    
    # Paths of length 2
    X2 = X_paths.dot(X_paths)
    
    X_deprop = utils.sp_mask(X, X2, invert=True)
    X_deprop = scipy.sparse.csr_matrix(X_deprop)
    return X_deprop

def create_clusters_2_clusters_visualization(        
        clusters_2_clusters_paths,
        comp_df, 
        cluster_list=None,
        output=None,
        sample_2_clusters=None, 
        remove_fragment_plasmids=None,
        country_df=None,
        gene_2_clusters=None,
        vertex_attrs=None,
        edge_attrs=None,
        default_edge_attrs=None,
        extra_edges=None):
    """
    From a list of clusters, create the cluster-to-cluster subnetwork.

    -- Edges are de-transitivized using function depropagate()

    Optionally, write to a graphml for cytoscape visualization.
    """

    if remove_fragment_plasmids:
        # Restrict to backbone and cargo plasmids (remove fragments)
        cluster_list = np.array(cluster_list)[comp_df.loc[cluster_list, 'cluster_type'].isin(['source', 'backbone'])]

#    if norm_edges:
    if False:
        # Normalize number of edges between components by the product of the components' sizes
        clusters_2_clusters_norm = nb_utils.norm(clusters_2_clusters_paths, marginals=comp_df['cluster_size'].values)
    else:
        clusters_2_clusters_norm = clusters_2_clusters_paths        

    # ## Use the depropagated edges, for a more parsimonious network
    # clusters_2_clusters_norm = enrich.norm(clusters_2_clusters_deprop, marginals=comp_df['cluster_size'].values)

    subgraph = clusters_2_clusters_norm[cluster_list, :][:, cluster_list]
    subgraph_paths = subgraph > 0

    # Use the depropagated edges, for a more parsimonious network
    subgraph = depropagate(subgraph, subgraph_paths)

    # Remove self-edges
    utils.setdiag(subgraph, 0)
    print('Number of edges:', subgraph.nonzero()[0].size)

#    if non_source_magnify:
    if False:
        # Magnify the edges between non-source nodes by 10x (to make them clump more)
        is_source = comp_df.loc[cluster_list, 'is_source'].values
        subgraph = subgraph.tocoo()
        mask = utils.isin_int(subgraph.row, np.where(~ is_source)[0]) & utils.isin_int(subgraph.col, np.where(~ is_source)[0])
        subgraph.data[mask] = 10 * subgraph.data[mask]
        subgraph = subgraph.tocsr()
        
    subgraph = utils.sp_to_df(subgraph, value_header='weight', rownames=cluster_list, colnames=cluster_list)

#    if magnify:
    if False:
        # Set all the weights to 1
        subgraph['weight'] = 1
        mask = (~ comp_df.loc[subgraph['row'].values, 'is_source'].values) & (~ comp_df.loc[subgraph['col'].values, 'is_source'].values)
        multiplier = np.array([1, 10])[mask.astype(int)]
        subgraph['weight'] *= multiplier

    pre_vertex_attrs = vertex_attrs

    # Create node attributes of plasmid clusters
    vertex_attrs = comp_df.loc[cluster_list]
    vertex_attrs['members_has_circ'] = vertex_attrs['members_circ'] > 0
    vertex_attrs['node_type'] = 'plasmid'

    # Set the label and shape and color
    # node_colors = pd.concat([comp_df.loc[cluster_list, 'cluster_type'].map({'backbone' : '#FFFF00', 'fragment' : '#FF3366', 'source' : '#33CCFF'}),
    #                          pd.Series('#FF3366', sample_list)]).to_frame('node_colors')

    #node_colors = comp_df.loc[cluster_list, 'cluster_type'].rename_axis('node_colors').map({'backbone' : '#FFFF00', 'fragment' : '#FF3366', 'source' : '#33CCFF'})

    if sample_2_clusters is not None:
        if isinstance(sample_2_clusters, pd.DataFrame) and ('cluster' in sample_2_clusters.columns) and ('sample' in sample_2_clusters.columns):
            sample_2_clusters = utils.subset(sample_2_clusters, dict(cluster=cluster_list))
            sample_2_clusters = sample_2_clusters.rename(columns={'cluster':'row', 'sample':'col'})
            sample_2_clusters['weight'] = 1
            subgraph = pd.concat([subgraph, sample_2_clusters])

            subgraph = depropagate_long(subgraph)

            # # Set the weight of edges between clusters to be 10, and everything else to 1
            subgraph['weight'] = 1
            subgraph.loc[subgraph['row'].isin(cluster_list) & subgraph['col'].isin(cluster_list), 'weight'] = 10
            subgraph['edgetype'] = 'sample-plasmid'
            subgraph.loc[subgraph['row'].isin(cluster_list) & subgraph['col'].isin(cluster_list), 'edgetype'] = 'plasmid-plasmid'

            # Add node attribute
            vertex_attrs['node_type'] = 'plasmid'

            sample_list = sample_2_clusters['col'].unique()
            vertex_attrs = vertex_attrs.append(pd.Series('sample', sample_list).to_frame('node_type'))

            # Check that this is true, because I will use color for fragment plasmids (red) for samples
            assert remove_fragment_plasmids
            assert country_df is not None

        else:
            raise Exception('Unsupported')
    
    if extra_edges is not None:
        subgraph = pd.concat([subgraph, extra_edges], sort=True)

    subgraph = depropagate_long(subgraph)

    if gene_2_clusters is not None:
        subgraph = pd.concat([subgraph, gene_2_clusters.assign(weight=1)])

    # Set all node sizes to 1
    vertex_attrs['node_size'] = 1

    # Update vertex_attrs with pre-specified values
    if pre_vertex_attrs is not None:
        vertex_attrs = vertex_attrs.reindex(vertex_attrs.index.union(pre_vertex_attrs.index))
        # Need to check that to-update columns already exist (due to how pd.DataFrame.update works)
        for c in pre_vertex_attrs.columns:
            if c not in vertex_attrs.columns:
                vertex_attrs[c] = np.nan
        vertex_attrs.update(pre_vertex_attrs)

    G = utils.create_ig(subgraph, directed=True, vertex_attrs=vertex_attrs, v1='row', v2='col',
                        edge_attrs=edge_attrs,
                        default_edge_attrs=default_edge_attrs,
                        output=output)
    print(G.summary())

    return G, vertex_attrs, subgraph


def get_minimal_maximal_backbones(backbone_list, assembly_plasmid_clusters_2_clusters_paths):
    """Given a list of backbones, return the set of maximal and minimal backbones relative to this list"""

    backbone_list = np.array(list(backbone_list))

#    assert len(backbone_list)>0

    if len(backbone_list)>0:
        # Get the maximal backbone among a list of backbones
        tmp = assembly_plasmid_clusters_2_clusters_paths[backbone_list,:][:,backbone_list]
        utils.setdiag(tmp, 0)
        maximal_backbone_list = backbone_list[utils.as_flat(tmp.sum(0)==0)]
        minimal_backbone_list = backbone_list[utils.as_flat(tmp.sum(1)==0)]
    else:
        minimal_backbone_list, maximal_backbone_list = np.array([]), np.array([])

    return minimal_backbone_list, maximal_backbone_list


def separate_cargo_func(func,
                        cov_detect=None,
                        self_assembly=None,
                        strict=True,
                        strict_sources=None,
                        plasmid_grouping_df=None,
                        plasmid_grouping_comp_df=None):
    utils.tprint('Merging plasmid clustering and backbone info')

    cluster_info = plasmid_grouping_df[['cluster']].merge(
        plasmid_grouping_comp_df[['cluster_type', 'backbones', 'rep_central', 'minimal_backbone', 'minimal_system', 'is_minimal_backbone', 'is_maximal_backbone']].explode('backbones').rename(columns={'backbones':'backbone'}),
#        plasmid_grouping_comp_df[['cluster_type', 'backbones', 'rep_central']].explode('backbones').rename(columns={'backbones':'backbone'}),
                left_on='cluster', right_index=True).reset_index()
    cluster_info['contig'] = pd.Categorical(cluster_info['contig'], categories=func['contig'].cat.categories)
    cluster_info = cluster_info.convert_dtypes().astype({'rep_central':'category', 'cluster_type':'category'})
    all_func = utils.better_merge(func, cluster_info, on='contig', how='left')

    if self_assembly:
        # In this mode, assume that each contig is detected in only the sample it was assembled from
        all_func['sample'] = utils.categorical_from_nonunique(all_func['contig'].cat.codes,
                              pd.Series([x.rsplit('_', 1)[0] for x in assembly_plasmid_sample_func['contig'].cat.categories]))
        all_sample_func = utils.better_merge(all_func, country_df.reset_index(), on='sample', how='left')
        return all_func, all_sample_func
        
    # For each backbone, get a list of accessions contained in the backbone
    utils.tprint('Calculating cargo functions')
#     backbone_2_accessions = utils.subset(all_func, dict(cluster_type='backbone')).groupby('backbone')['accession'].unique().explode().reset_index()    
    backbone_2_accessions = utils.subset(all_func, cluster_type='backbone', columns=['cluster','accession']).drop_duplicates().rename(columns={'cluster':'backbone'})
    backbone_2_accessions['is_backbone_accession'] = True
    # Label which rows represent an accession that occurs somewhere in the associated backbone
    all_func = all_func.merge(backbone_2_accessions, on=['backbone', 'accession'], how='left').astype({'accession':'category'})
    all_func['is_backbone_accession'].fillna(False, inplace=True)
    # Define the cargo annotations as those that occur in a contig where (1) the contig has a backbone, and (2) the annotation is not on the backbone
    in_system_func = all_func.dropna(subset=['backbone'])
    cargo_func = utils.subset(in_system_func, is_backbone_accession=False)

    # Define a strict condition of cargo genes as those that have NONE of the annotations that are found in the backbone genes. E.g. if a gene shares a Pfam (but none of its COGs) with the backbone genes,
    # then that is not a cargo gene
    cargo_genes_not_strict = utils.subset(in_system_func, is_backbone_accession=True)
    if strict_sources is not None:
        cargo_genes_not_strict = utils.subset(cargo_genes_not_strict, source=strict_sources)
    cargo_genes_not_strict = cargo_genes_not_strict[['contig', 'start', 'stop', 'backbone']].drop_duplicates()

    # Annotate which are strict cargo genes
    cargo_func = cargo_func.merge(cargo_genes_not_strict, on=['contig', 'start', 'stop', 'backbone'], how='left', indicator=True)
    cargo_func = cargo_func.astype({'contig':'category'})
    cargo_func['strict_cargo'] = cargo_func['_merge'] == 'left_only'
    cargo_func = cargo_func.drop(columns=['_merge'])
    x, y = len(cargo_func.drop_duplicates(['contig','start','stop'])), len(utils.subset(cargo_func, strict_cargo=True).drop_duplicates(['contig','start','stop']))
    print('Cargo genes:', x)
    print('--Strict cargo genes: {} ({:.3f})'.format(y, y/x))
    x, y = len(cargo_func), len(utils.subset(cargo_func, strict_cargo=True))
    print('Cargo functions:', x)
    print('--In strict cargo genes: {} ({:.3f})'.format(y, y/x))

#    return cargo_func, in_system_func, cargo_genes_not_strict

    if strict:        
        # Return functions ONLY IN strict cargo genes
        cargo_func = utils.subset(cargo_func, strict_cargo=True)

    if cov_detect is not None:        
        utils.tprint('Merging sample detection')
        cov_detect = cov_detect[['contig', 'sample', 'country', 'industrial']].drop_duplicates()        
        all_sample_func = utils.better_merge(all_func, cov_detect, on='contig')
        cargo_sample_func = utils.better_merge(cargo_func, cov_detect, on='contig')
    else:
        all_sample_func, cargo_sample_func = None, None
        
    return cargo_func, cargo_sample_func, all_func, all_sample_func


def format_derep_output(contigs_df, clusters_df, clusters_2_clusters, output=None):
    """Formats the output of derep_sources()"""

    # def rename_cluster_type(x):
    #     if x=='source':
    #         return 'maximal'
    #     else:
    #         return x

    def format_clusters(clusters_df, systems_df):
        tmp = clusters_df[['cluster', 'cluster_four_types', 'cluster_size', 'members_circ', 'members', 'rep_central', 'backbones']].copy()
        tmp = tmp.rename(columns={'rep_central' : 'representative_contig'})
        tmp['members'] = tmp['members'].apply(lambda x: '|'.join(sorted(x)))
        tmp = tmp.rename(columns={'members':'contigs', 'cluster_size':'number_of_contigs', 'members_circ':'number_of_circular_contigs'})
        # tmp['cluster_type'] = tmp['cluster_type'].apply(rename_cluster_type)
        tmp = tmp.rename(columns={'cluster_four_types':'cluster_type'})

        # Add system names
        backbone_2_system = systems_df.set_index('backbone_cluster')['system_name'].to_dict()
        tmp['backbones'] = tmp['backbones'].apply(lambda backbone_list: '|'.join([backbone_2_system[b] for b in sorted(backbone_list)]))
        tmp = tmp.rename(columns={'backbones':'systems'})
        
        return tmp

    def format_contigs(contigs_df, clusters_df, systems_df):
        tmp = contigs_df[['cluster', 'cluster_four_types', 'circular', 'rep_central', 'backbones']].copy()
        tmp = tmp.rename(columns={'rep_central' : 'representative_contig'})
        # tmp['cluster_type'] = tmp['cluster_type'].apply(rename_cluster_type)
        tmp = tmp.rename(columns={'cluster_four_types':'cluster_type'})
        tmp['circular'] = tmp['circular'].astype(int)

        # Add system names
        backbone_2_system = systems_df.set_index('backbone_cluster')['system_name'].to_dict()
        tmp['backbones'] = tmp['backbones'].apply(lambda backbone_list: '|'.join([backbone_2_system[b] for b in sorted(backbone_list)]))
        tmp = tmp.rename(columns={'backbones':'systems'})

        return tmp

    def format_systems(contigs_df, clusters_df, clusters_2_clusters):
        nonfragments = utils.subset(contigs_df, cluster_type='fragment', invert=True)[['cluster', 'backbones', 'cluster_type']]\
                        .rename(columns={'backbones':'backbone'}).explode('backbone').dropna().reset_index()
        compound_tmp = nonfragments[nonfragments['cluster'] != nonfragments['backbone']]
        backbone_tmp = utils.subset(nonfragments, cluster_type='backbone')

        # display(backbone_tmp)
        # display(compound_tmp)

        backbone_2_compound_plasmid = compound_tmp.groupby('backbone')['contig'].apply(sorted).to_frame('compound_plasmids')
        backbone_2_compound_plasmid['number_of_compound_plasmids'] = backbone_2_compound_plasmid['compound_plasmids'].apply(len)
#        backbone_2_compound_plasmid['compound_plasmids'] = backbone_2_compound_plasmid['compound_plasmids'].apply(lambda x: '|'.join(x))

        backbone_2_compound_cluster = compound_tmp.drop_duplicates(['backbone','cluster']).groupby('backbone')['cluster'].apply(sorted).to_frame('compound_clusters')
        backbone_2_compound_cluster['number_of_compound_clusters'] = backbone_2_compound_cluster['compound_clusters'].apply(len)
#        backbone_2_compound_cluster['compound_clusters'] = backbone_2_compound_cluster['compound_clusters'].apply(lambda x: '|'.join(map(str,x)))

        backbone_2_backbone_plasmid = backbone_tmp.groupby('cluster')['contig'].apply(sorted).to_frame('backbone_plasmids')
        backbone_2_backbone_plasmid['number_of_backbone_plasmids'] = backbone_2_backbone_plasmid['backbone_plasmids'].apply(len)
#        backbone_2_backbone_plasmid['backbone_plasmids'] = backbone_2_backbone_plasmid['backbone_plasmids'].apply(lambda x: '|'.join(x))

        backbone_2_compound_plasmid['compound_plasmids'] = backbone_2_compound_plasmid['compound_plasmids'].apply(lambda x: '|'.join(x))
        backbone_2_compound_cluster['compound_clusters'] = backbone_2_compound_cluster['compound_clusters'].apply(lambda x: '|'.join(map(str,x)))
        backbone_2_backbone_plasmid['backbone_plasmids'] = backbone_2_backbone_plasmid['backbone_plasmids'].apply(lambda x: '|'.join(x))

        # Rename plasmid systems by using the argsort index order of the backbone's cluster numbers
        tmp = utils.subset(clusters_df, cluster_type='backbone').copy()
        assert np.all(tmp['cluster'].values == np.sort(tmp['cluster'].values)), 'Clusters are not sorted'
        tmp['system_name'] = ['PS'+str(i+1) for i in range(len(tmp))]
        system_names = tmp[['system_name', 'cluster']].rename(columns={'cluster':'backbone_cluster'})

        # Add list of backbone and compound clusters/plasmids (exclude fragment plasmids)
        system_names['backbone_cluster'] = system_names.index
        system_names = system_names.join(backbone_2_backbone_plasmid)
        system_names = system_names.join(backbone_2_compound_cluster)
        system_names = system_names.join(backbone_2_compound_plasmid)
        system_names = system_names.convert_dtypes()

        assert len(system_names.columns) == 8
        system_names = system_names[['system_name', 'backbone_cluster', 'number_of_backbone_plasmids', 'backbone_plasmids',
                                     'compound_clusters', 'number_of_compound_clusters', 'number_of_compound_plasmids', 'compound_plasmids']]

        return system_names

    systems_df = format_systems(contigs_df, clusters_df, clusters_2_clusters)
    clusters_df = format_clusters(clusters_df, systems_df)
    contigs_df = format_contigs(contigs_df, clusters_df, systems_df)
    clusters_2_clusters = pd.DataFrame(clusters_2_clusters.nonzero(), index=['shorter_cluster', 'longer_cluster']).T

    # display(contigs_df)
    # display(clusters_df)
    # display(systems_df)

    assert systems_df['compound_plasmids'].notna().all(), "Some of your plasmid systems have no compound plasmids. This doesn't make sense and probably is a bug in the code"
    
    if output is not None:
        utils.write_table(contigs_df.reset_index(),
                          str(output) + '_contigs.txt',
                          txt=True, txt_kws=dict(index=False))
        utils.write_table(contigs_df.reset_index(),
                          str(output) + '_clusters.txt',
                          txt=True, txt_kws=dict(index=False))
        utils.write_table(systems_df,
                          str(output) + '_systems.txt',
                          txt=True, txt_kws=dict(index=False))
        utils.write_table(clusters_2_clusters,
                          str(output) + '_cluster_graph.txt',
                          txt=True, txt_kws=dict(index=False))
        
    return contigs_df, clusters_df, systems_df, clusters_2_clusters


def create_contig_2_contig_igraph(similarities, contigs_df, output=None):
    """
    similarities : long-format pd.DataFrame of ANI similarities between sequences. Only pairs that pass the similarity threshold (e.g. >=90% identity/coverage) should be included

    contigs_df : pd.DataFrame describing the contigs
    """
    
    tmp_contigs = contigs_df.copy()

    # Create 'label' attribute for Cytoscape styling
    tmp_contigs['label'] = [f"contig:{x}\ncluster:{y}\nsystem:{z}" \
                            for x, y, z in contigs_df[['cluster','systems']].to_records(index=True)]

    # # Add a dummy isolated node, for testing
    # tmp_contigs = pd.concat([tmp_contigs, tmp_contigs.iloc[0].copy().rename('asdf').to_frame().T])

    # Remove self-edges
    tmp = similarities[similarities['query'] != similarities['reference']]

    print('output:', output)

    return utils.create_ig(tmp,
                           v1='query', v2='reference',
                           vertex_attrs=tmp_contigs,
                           output=output)

def create_cluster_2_cluster_igraph(clusters_2_clusters_sp, clusters_df, output=None):
    """
    clusters_2_clusters_sp : scipy.sparse adjacency matrix describing the child-to-parent edges

    contigs_df : pd.DataFrame describing the clusters
    """

    tmp = depropagate(clusters_2_clusters_sp)

    # Create 'label' attribute for Cytoscape styling
    tmp_clusters = clusters_df.copy()
    tmp_clusters['label'] = [f"cluster:{x}\nrep:{y}\nsystem:{z}" for x, y, z in clusters_df[['cluster', 'representative_contig','systems']].to_records(index=False)]
    tmp_clusters.index = tmp_clusters.index.astype(str)

    # # Add a dummy isolated node, for testing
    # tmp_clusters = pd.concat([tmp_clusters, tmp_clusters.iloc[0].copy().rename('asdf').to_frame().T.assign(label='asdf')])

    return utils.create_ig(tmp,
                           vertex_attrs=tmp_clusters,
                           rownames=[str(x) for x in range(tmp.shape[0])],
                           output=output)
