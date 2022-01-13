import argparse
import textwrap

import pandas as pd

from plasx import utils
from mobmess import ani_utils

def infer_systems(args):

    similarity_threshold = args.min_similarity * 100
    min_alignment_fraction = args.min_coverage
    threads = args.threads

    utils.tprint(f'Reading fasta file of sequences at {args.sequences}')
    fasta_dict = utils.get_fasta_dict(args.sequences)
    sequence_lengths = {x : len(y) for x, y in fasta_dict.items()}

    utils.tprint(f'Reading circularity/completeness information at {args.complete}')
    circular_df = utils.read_table(args.complete, read_table_kws=dict(header=None)).set_index(0)[1]

    if args.ani is None:

        with utils.TemporaryDirectory(name=args.tmp) as tmp:

            utils.tprint('Calculating pairwise alignments and ANI')

            nucmer_first_output = tmp / 'out_maxmatch_k16'
            nucmer_delta_output = tmp / 'out_maxmatch_k16.qr_filter'
            nucmer_delta_pkl_output = tmp / 'out_maxmatch_k16.qr_filter.pkl.blp'
            nucmer_summary_output = tmp / 'out_maxmatch_k16.delta.qr_filter.summary.pkl.blp'

            # Run NUCMER
            ani_utils.run_nucmer(args.sequences, nucmer_first_output, threads=threads)

            # Use MUMmer's delta-filter tool to remove overlapping
            # alignment blocks, to create a set of blocks that
            # represents a 1-to-1 alignment between very pair of
            # sequences
            ani_utils.mummer_fast_deltafilter(str(nucmer_first_output) + '.delta', nucmer_delta_output, n_jobs=threads, tmp=tmp / 'delta')

            # Parse the filtered alignment blocks
            aln_blocks = ani_utils.read_mummer_aln_blocks(nucmer_delta_output, output=nucmer_delta_pkl_output, tmp=tmp, delete_tmp=False)
            # aln_blocks = utils.unpickle(nucmer_delta_pkl_output)

            similarities = ani_utils.get_mummer_query_coverage_py(aln_blocks, output=nucmer_summary_output)

            # ref query query_cov query_cov_uniq mismatches nblocks ANI full_ANI
            similarities = utils.unpickle(nucmer_summary_output)

            # Save the alignment summary into a tab-separated text file
            print('Columns:', similarities.columns)
            tmp = similarities.drop(columns=['query_cov', 'mismatches', 'nblocks'])\
                              .rename(columns={'ref':'reference',
                                               'query_cov_uniq':'C',
                                               'ANI':'I_local',
                                               'full_ANI':'I_global'})
            utils.write_table(tmp, 
                              str(args.output) + '_ani.txt',
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


    # utils.display(circular_df)
    # utils.display(circular_df.loc['ISR0117_000000003149'])
    # print('---')
    # print('similarity_threshold / min_alignment_fraction:', similarity_threshold, min_alignment_fraction)

    utils.tprint('Dereplicating and inferring systems')
    contigs_df, _, clusters_2_clusters, _, _, clusters_df = \
        ani_utils.derep_sources(similarities, similarity_threshold, min_alignment_fraction, 
                                circular_df=circular_df, lengths=sequence_lengths,
                                output=None,
                                fasta_dict=fasta_dict,
                                plasmid_scores=None,
                                verbose=True)

    utils.tprint('Formatting and saving output')
    contigs_df, systems_df, clusters_2_clusters = ani_utils.format_derep_output(
        contigs_df, clusters_df, clusters_2_clusters,
        output=args.output)


def get_parser():   
    description = textwrap.dedent(\
"""Runs MobMess algorithm to infer plasmid systems.""")
 
    parser = argparse.ArgumentParser(
        description=description)
    parser.set_defaults(func=lambda args: parser.print_help())
    
    # Add subparsers
    subparsers = parser.add_subparsers()

    systems_parser = subparsers.add_parser('systems', help='Infer plasmid systems.')
    systems_parser.set_defaults(func=infer_systems)
    required = systems_parser.add_argument_group('required arguments')
    required.add_argument(
        '-s', '--sequences', dest='sequences', type=str, default=None, required=True,
        help="""The sequences to align and cluster""")
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

    return parser

def run(args=None):
    parser = get_parser()

    args = parser.parse_args(args=args)
    args.func(args)

if __name__=='__main__':
    run()
