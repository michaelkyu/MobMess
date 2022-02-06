import argparse
import textwrap

import pandas as pd

from mobmess import utils, ani_utils

def infer_systems(args):

    similarity_threshold = args.min_similarity * 100
    min_alignment_fraction = args.min_coverage
    threads = args.threads

    utils.tprint(f'Reading fasta file of sequences at {args.sequences}')
    fasta_dict = utils.get_fasta_dict(args.sequences)
    sequence_lengths = {x : len(y) for x, y in fasta_dict.items()}

    utils.tprint(f'Reading circularity/completeness information at {args.complete}')
    circular_df = utils.read_table(args.complete, read_table_kws=dict(header=None)).set_index(0)[1]

    with utils.TemporaryDirectory(name=args.tmp, verbose=True) as tmp:

        if args.ani is None:

            utils.tprint('Calculating pairwise alignments and ANI')

            nucmer_first_output = tmp / 'mummer_align'
            nucmer_delta_output = tmp / 'mummer_align.qr_filter'
            nucmer_delta_pkl_output = tmp / 'mummer_align.qr_filter.pkl.blp'
            nucmer_summary_output = tmp / 'mummer_align.delta.qr_filter.summary.pkl.blp'

            # Run nucmer (from the MUMmer package) to align sequences
            ani_utils.run_nucmer(args.sequences, nucmer_first_output, threads=threads)

            # Use MUMmer's delta-filter tool to remove overlapping alignment blocks, to create a set of blocks that represents a
            # 1-to-1 alignment between very pair of sequences
            ani_utils.mummer_fast_deltafilter(
                str(nucmer_first_output) + '.delta',
                nucmer_delta_output,
                n_jobs=threads,
                tmp=tmp / 'mummer_align_delta_scripts')

            # Parse the filtered alignment blocks
            aln_blocks = ani_utils.read_mummer_aln_blocks(
                nucmer_delta_output, 
                output=nucmer_delta_pkl_output,
                tmp=tmp,
                delete_tmp=False,
                filename_prefix='mummer_align.delta')

            similarities = ani_utils.get_mummer_query_coverage_py(aln_blocks, output=nucmer_summary_output)

            # ref query query_cov query_cov_uniq mismatches nblocks ANI full_ANI
            similarities = utils.unpickle(nucmer_summary_output)

            # Save the alignment summary into a tab-separated text file
            utils.tprint('Columns:', similarities.columns)
            similarities_tmp = similarities.drop(columns=['query_cov', 'mismatches', 'nblocks'])\
                              .rename(columns={'ref':'reference',
                                               'query_cov_uniq':'C',
                                               'ANI':'I_local',
                                               'full_ANI':'I_global'})
            utils.write_table(similarities_tmp, 
                              str(args.output) + '_ani.txt',
                              txt=True, txt_kws=dict(index=False))

            # Save the alignment blocks into a tab-separated text file
            utils.write_table(aln_blocks,
                              str(args.output) + '_ani_blocks.txt',
                              txt=True, txt_kws=dict(index=False))
            
        else:
            utils.tprint(f'Reading prespecified ANI table at {args.ani}')
            similarities = utils.read_table(args.ani, read_table_kws=dict(comment='#'))

            # Reference_Sequence Query_Sequence C I_local I_global
            similarities = similarities.rename(columns={
                'Reference_Sequence':'ref',
                'reference':'ref',
                'Query_Sequence':'query',
                'C':'query_cov_uniq',
                'I_local':'ANI',
                'I_global':'full_ANI'})

        similarities = similarities.rename(columns={'ref':'reference', 'query_cov_uniq':'coverage'})
        similarities['ANI'] = similarities['ANI'] * 100

        utils.tprint('Dereplicating and inferring systems')
        contigs_df, _, clusters_2_clusters_sp, _, _, clusters_df, similarities = \
            ani_utils.derep_sources(similarities, similarity_threshold, min_alignment_fraction, 
                                    circular_df=circular_df, lengths=sequence_lengths,
                                    output=tmp / 'mobmess',
                                    fasta_dict=fasta_dict,
                                    plasmid_scores=None,
                                    verbose=True)
    
    utils.tprint('Formatting and saving output')
    
    # Write tables describing contigs, clusters, systems
    contigs_df, clusters_df, systems_df, clusters_2_clusters = ani_utils.format_derep_output(
        contigs_df, clusters_df, clusters_2_clusters_sp,
        output=args.output)

    # Create *.graphml files for contigs-2-contigs and clusters-2-clusters networks
    contigs_2_contigs_G = ani_utils.create_contig_2_contig_igraph(
        similarities,
        contigs_df,
        output=str(args.output) + '_contig_graph.graphml')
    clusters_2_clusters_G = ani_utils.create_cluster_2_cluster_igraph(
        clusters_2_clusters_sp,
        clusters_df,
        output=str(args.output) + '_cluster_graph.graphml')
    
def visualize(args):

    from mobmess import plot

    if args.contigs is not None:
        contigs = [x.strip() for x in args.contigs.split(',')]
    else:
        contigs = None
        
    print('Plotting contigs:', ' '.join(contigs))

    # Need to set the matplotlib backend to a non-interactive one "Agg", otherwise it will try to create new windows of plots
    import matplotlib as mpl
    mpl.use('Agg')

    plot.visualize_alignment(
        args.annotations,
        gene_calls=args.gene_calls,
        contigs=contigs,
        fasta=args.sequences,
        aln_blocks=args.align,
        output=args.output,
        most_common_gene=True,
        dist='jacc',
        neighborhood=args.neighborhood,
        show=False,
        threads=args.threads,
        width=args.width,
        aln_blocks_height=args.align_blocks_height
        )

def get_parser():   
    description = textwrap.dedent(\
"""Runs MobMess algorithm to infer plasmid systems.""")
 
    parser = argparse.ArgumentParser(
        description=description)
    parser.set_defaults(func=lambda args: parser.print_help())
    
    # Add subparsers
    subparsers = parser.add_subparsers()

    ###### Infer plasmid systems ########

    systems_parser = subparsers.add_parser('systems', help='Infer plasmid systems.')
    systems_parser.set_defaults(func=infer_systems)

    required = systems_parser.add_argument_group('required arguments')
    required.add_argument(
        '-s', '--sequences', dest='sequences', type=str, default=None, required=True,
        help="""Fasta file of the sequences to align and cluster""")
    required.add_argument(
        '-c', '--complete', dest='complete', required=True,
        help="""Table indicating which sequences are assembled complete/circular""")
    required.add_argument(
        '-o', '--output', dest='output', required=True,
        help="""Filename prefix for output files""")

    optional = systems_parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-a', '--ani', dest='ani', default=None,
        help="""Precomputed file of MUMmer alignment results. Specifying this will skip computing MUMmer alignments, saving time.""")
    optional.add_argument(
        '-T', '--threads', dest='threads', type=int, default=1,
        help="""Number of threads to do pairwise MUMmer alignments. Default: 1 thread""")
    optional.add_argument(
        '--min-similarity', dest='min_similarity', type=float, default=0.9,
        help="""Minimum alignment identity. Default: 0.9""")
    optional.add_argument(
        '--min-coverage', dest='min_coverage', type=float, default=0.9,
        help="""Minimum alignment coverage. Default: 0.9""")
    optional.add_argument(
        '--tmp', dest='tmp', default=None,
        help="""Directory to save intermediate files, including ones created by nucmer. Default: a temporary directory that is deleted upon termination""")


    ###### Visualize the alignment of multiple plasmids or sequences ########

    visualize_parser = subparsers.add_parser('visualize', help='Visualize the alignment of plasmids in a system. Useful for seeing shared backbone content.')
    visualize_parser.set_defaults(func=visualize)

    required = visualize_parser.add_argument_group('required arguments')
    optional = visualize_parser.add_argument_group('optional arguments')
    required.add_argument(
        '-s', '--sequences', dest='sequences', type=str, default=None, required=True,
        help="""Fasta file of the sequences to align and cluster""")
    required.add_argument(
        '-a', '--annotations', dest='annotations', nargs='+', required=True,
        help='Table of gene annotations to COGs, Pfams, and de novo families')
    required.add_argument(
        '-g', '--gene-calls', dest='gene_calls', default=None,
        help='Table of gene calls, mapping gene_callers_id to contig')
    required.add_argument(
        '-o', '--output', dest='output', required=True,
        help="""PDF file to save visualization.""")
    optional.add_argument(
        '--contigs', dest='contigs',
        help="""A comma-separated list of contigs that you want to visualize. E.g. 'contig1,contig2,contig3'. Default: all contigs in the fasta file `--sequences` will be visualized.""")
    optional.add_argument(
        '--align', dest='align',
        help="""Table of alignment blocks produced by MUMmer. If you ran `mobmess systems` and saved intermediate files"""\
             """with the `--tmp` flag, then use the file 'mummer_align.qr_filter.pkl.blp'."""\
             """Default: if you don't specify this file, then MUMmer alignments will be computed on the fly.""")
    optional.add_argument(
        '-T', '--threads', dest='threads', type=int, default=1,
        help="""Number of threads to do pairwise MUMmer alignments (this only happens if you don't specify `--align`. Default: 1 thread""")
    optional.add_argument(
        '--neighborhood', dest='neighborhood', type=int, default=20000,
        help="""Only a neighborhood around each anchor gene will be visualized. This specifies the size of the neighborhood upstream and downstream of each anchor. Default: Show 20kb. Setting this to zero "0" will visualize the entire contigs.""")
    optional.add_argument(
        '--width', dest='width', type=int, default=40,
        help="""The width (inches) of the PDF page that shows the sequence alignment. You probably want to increase this value if your sequences are very long.""")
    optional.add_argument(
        '--align-blocks-height', dest='align_blocks_height', type=float, default=0.5,
        help="""The vertical spacing (inches) inbetween sequences. Increase this value to more clearly show the ribbons aligning sequences, especially for very long sequences. The default is 0.5. Set this to 1.0 to double the height, 2.0 to quadruple the height, etc.""")

    return parser

def run(args=None):
    parser = get_parser()

    args = parser.parse_args(args=args)
    args.func(args)

if __name__=='__main__':
    run()
