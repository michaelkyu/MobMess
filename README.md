MobMess is a tool for inferring evolutionary relations among plasmid sequences. MobMess performs two functions:

1. Identify a non-redundant subset of plasmids from an input set of plasmid sequences. MobMess aligns every pair of sequences using [MUMmer4](https://mummer4.github.io/), and then clusters sequences that are highly similar along their entire lengths. MobMess then chooses one sequence to represent each cluster.

2. Infer "plasmid systems", which is an evolutionary phenomenon in which a "backbone plasmid" with core genes acquires accessory genes to form "compound plasmids".

Here's a toy diagram of a plasmid system.

<p align="center">
  <img src="docs/plasmid_system_diagram.png" width="250" class="center"/>
</p>

This README includes a tutorial for inferring plasmid systems. It also includes instructions for reproducing the 1,169 plasmid systems from our study "The Genetic and Ecological Landscape of Plasmids in the Human Gut" by Michael Yu, Emily Fogarty, A. Murat Eren. For a technical explanation of the MobMess algorithm, please see the supplementary methods of this study.

# Installation

MobMess requires `python >=3.7`. We recommend installing MobMess in a new virtual environment using Anaconda, to make installing dependencies easier. **We have only tested MobMess on Linux machines. In the future, we will provide support for running on Mac OS and Windows.**

```
# Create virtual environment named "mobmess" (you can change the name to whatever you like)
conda create --name mobmess
# Install dependencies
conda install --name mobmess -y -c anaconda -c conda-forge -c bioconda --override-channels --strict-channel-priority  numpy pandas scipy scikit-learn igraph numba python-blosc mummer4
```

Alternatively, create the environment and install dependencies in a single command

```
conda create --name mobmess -y -c anaconda -c conda-forge -c bioconda --override-channels --strict-channel-priority  numpy pandas scipy scikit-learn igraph numba python-blosc mummer4
```

Then, activate the new environment
```
conda activate mobmess
```

Next, download and install [PlasX](https://github.com/michaelkyu/plasx), which has some utility functions that MobMess uses. You do not need to run the conda commands at https://github.com/michaelkyu/PlasX#installation, as the above conda commands already installs the relevant dependencies.
```
# Download the PlasX repository
# - If you've previously downloaded the repository, skip this command and
#   just change into the parent directory of where you downloaded it.
git clone https://github.com/michaelkyu/PlasX.git

# Install PlasX
pip install ./PlasX
```

Finally, download and install MobMess
```
git clone https://github.com/michaelkyu/MobMess.git
pip install ./MobMess
```

# Tutorial for inferring plasmid systems using MobMess

In this tutorial, we will use MobMess to organize plasmids into plasmid systems. We will use an example set of plasmid sequences in `test/test-contigs.fa`, but you can repeat these steps with your own sequences. 

**Note that the output files of this tutorial are already in the `test` directory. Running the code in this tutorial will recreate the same files.**

### Preliminary setup of command line


```bash
# Change into the directory where MobMess was downloaded (e.g. where `git clone` downloaded to)
cd /path/to/MobMess

# Change into the `test` subdirectory that contains test-contigs.fa
cd test

# Input and output filename will start with this prefix
PREFIX='test-contigs'

# The number of CPU cores that will be used to align plasmids MUMmer. We recommend you setting it to a high number, to speed up the processing of many contigs
THREADS=4
```



### Understanding the format of the input files


We are going to run MobMess on this fasta file of 123 plasmid sequences


```bash
head test-contigs.fa
```

```
>MON0062_000000008770
TTGGTCACCCATAATATAAGGAACAATTCAATTTGTACGCTGTGCTTTAACATTTTGATACGATTTTAAGAGAATGAAATAAGTTTAAATCACTTCAATTTCAATTTTTGAATTATGTATTTCAGCAGACCTTGGACTTCTATCTAAGTGATAGGCAGTCCTGAAAATTCCAATCTCTTTTAGGTCAAGACAATTAGTCT

>AST0016_000000004532
AACAACAAGAAAACAAGTTTCTTTTCGTTTACGTGAGGATTTATTAATGGCTTTAAGGGAAGAAGCGAGAAAGGCTAACAAAAGCCTGAACGGTTTCGTAGAAAGCATTCTGGCAGATGCAATGCTGAAGAGAACCAATGAGGGTAATAACGTACTCATAAAAAACGATACAGAGGTTTAGTAAGGTTTTTTGGCATTTT

>CAN0004_000000004460
AACGACTTTAACAAATAAAAGTCGTAGATTGTGTAATTAGTATTCAAAAAGTCGTAGTTAAAGTTTGAAATAGTATTCAAAAAGTCGTAGTTAGTATTCAAAAAGTCGTAGTTAGATTCTTGTTGTTGCTTGTAATTTATTGTTTATCAGATTACTAAAACGTGTTTTTCGAAAGTCGATAAGCATATAACAAGTAATGC

>CAN0005_000000003481
CGGTAAGCCCTTCCAGCCGGGAGCTGGAGAAAATGGGCAAGACAGAGAAGGAACAGGCTGAAGCCATGAGAAGGTATGTCCGTGATGATGTGATGCAGCACTATGCGGAAGGGTTCGGAAAAGGCCTGAACAAAGAGGATATCGAGTATTACGGAAAGATCCATTTCGAGAGGAAGGGAGCCGACCGGTACGACATGCAC

>CAN0015_000000007026
TTTAGAGGATAAGGAACGACAGATGTTCCAGATAGTCCGGTTAATGGATGAACAACAATCTATTAACAAGAAGATAGCCAATCAAATTCCGGTTATTGTACAGAAAAGTGTGCAGGAACAGTCCAAAAAGCCAAAACGAAAAGGTTTCTTAGGCATATTCGGCAAAAAAAAGGAAGTAACTCCAGCAGTATCAACCACTA
```

MobMess also requires annotations of which sequences were assembled completely (as a circular DNA element), as opposed to being assembly fragments. MobMess uses this information to identify backbone plasmids, as it requires backbones to be complete.


```bash
# This is a two-column table, where the first column is the contig name and the second column indicates which sequences are complete.
# - The second column must contain the strings True/False or the integers 1/0, to represent complete/incomplete, respectively.
head test-contigs-circular.txt
```

```
MON0062_000000008770	1
AST0016_000000004532	1
CAN0004_000000004460	1
CAN0005_000000003481	1
CAN0015_000000007026	0
CHI0033_000000001004	0
CHI0132_000000005259	1
DEN0022_000000006299	0
DEN0056_000000000137	1
DEN0078_000000004762	1
```

### Infer plasmid systems

Activate the conda environment where MobMess is installed


```bash
conda deactivate
conda activate mobmess
```

MobMess can be run using the command line. Currently, one subcommand `mobmess systems` is implemented.


```bash
mobmess systems -h
```

```
usage: mobmess systems [-h] -s SEQUENCES -c COMPLETE -o OUTPUT [-a ANI] [-T THREADS] [--min-similarity MIN_SIMILARITY] [--min-coverage MIN_COVERAGE] [--tmp TMP]

optional arguments:
  -h, --help            show this help message and exit

required arguments:
  -s SEQUENCES, --sequences SEQUENCES
                        The sequences to align and cluster
  -c COMPLETE, --complete COMPLETE
                        Table indicating which sequences are assembled complete/circular
  -o OUTPUT, --output OUTPUT
                        Filename prefix for output files

optional arguments:
  -a ANI, --ani ANI     Precomputed file of MUMmer alignment results. Specifying this will skip computing MUMmer alignments, saving time.
  -T THREADS, --threads THREADS
                        Number of threads to do pairwise MUMmer alignments. Default: 1 thread
  --min-similarity MIN_SIMILARITY
                        Minimum alignment identity. Default: 0.9
  --min-coverage MIN_COVERAGE
                        Minimum alignment coverage. Default: 0.9
  --tmp TMP             Directory to save intermediate files, including ones created by nucmer. Default: a temporary directory that is deleted upon termination

```                        


```bash
# Infer plasmid systems (run `mobmess systems -h` to get help on parameters)
mobmess systems \
    --sequences $PREFIX.fa \
    --complete $PREFIX-circular.txt \
    --output $PREFIX-mobmess \
    --threads $THREADS
```

### Understanding the output files

The above call to `mobmess systems` generated four files with names that start with $PREFIX-mobmess (i.e. `test-contigs-mobmess`). Here we'll see the formats of these files.


```bash
# Table that summarizes the pairwise sequence alignments generated by MUMMER.
# - The alignment is specified asymmetrically, such that one contig is the "reference" and the other contig is the "query"
# - C : the fraction of the query contig that is covered by alignment
# - I_local : the average nucleotide identity, calculated with respect to the aligned subregions
# - I_global : the average nucleotide identity, by considering the full length of the query. That is, subregions of the query sequence that wasn't aligned is still factored into this calculation.
head $PREFIX-mobmess_ani.txt
```

```
reference	query	C	I_local	I_global
AST0016_000000004532	AST0016_000000004532	1.0	1.0	1.0
AST0016_000000004532	AST0151_000000002645	0.7538680138932744	0.9606282722513089	0.7241869276918219
AST0016_000000004532	CAN0004_000000004460	0.9805754077331867	0.9527191179218838	0.9342129375114532
AST0016_000000004532	CAN0005_000000003481	0.9805147058823529	0.952755905511811	0.9341911764705881
AST0016_000000004532	CAN0015_000000007026	0.8440320962888666	0.9544464250346604	0.8055834169174189
AST0016_000000004532	CHI0011_000000000792	0.688564110290794	0.9606126914660832	0.6614434232333886
AST0016_000000004532	CHI0033_000000001004	0.7208333333333333	0.9438496831986017	0.6803583133056588
AST0016_000000004532	CHI0047_000000002374	0.7312	0.9592997811816193	0.70144
AST0016_000000004532	CHI0062_000000002655	0.7285953177257525	0.9674087675005738	0.7048494983277592
```


```bash
# Table that summarizes the assignment of contigs to clusters.
# - Clusters are numbered using integers, starting with 0.
# - Cluster are categorized into three types: "backbone", "fragment", "maximal"
# - One contig from every cluster is designated as the representative.
head $PREFIX-mobmess_contigs.txt
```

```
contig	cluster	cluster_type	representative_contig
MON0062_000000008770	0	backbone	MON0062_000000008770
AST0016_000000004532	1	backbone	ISR0183_000000005642
CAN0004_000000004460	1	backbone	ISR0183_000000005642
CAN0005_000000003481	1	backbone	ISR0183_000000005642
CAN0015_000000007026	1	backbone	ISR0183_000000005642
CHI0033_000000001004	1	backbone	ISR0183_000000005642
CHI0132_000000005259	1	backbone	ISR0183_000000005642
DEN0022_000000006299	1	backbone	ISR0183_000000005642
DEN0056_000000000137	1	backbone	ISR0183_000000005642
```


```bash
# Table that summarizes the plasmid systems
# - There is a one-to-one correspondence between plasmid systems and backbone clusters. As an easy naming, the systems are renamed 'PS1' ... 'PSn' where n is the number of systems
# - Multiple values are concatenated with the '|' separator. This applies to columns 'backbone_cluster', 'backbone_plasmids', 'compound_clusters', and 'compound_plasmids'  (not printed in this example for conciseness)
cat $PREFIX-mobmess_systems.txt | cut -f1,2,3,6,7
```

```
system_name	backbone_cluster	number_of_backbone_plasmids	number_of_compound_clusters	number_of_compound_plasmids
PS1	0	1	10	122
PS2	1	146	9	49
PS3	4	6	1	9
PS4	9	60	1	1
```


```bash
# Two-column table of plasmid clusters that align to each other within the sequence similarity threshold (e.g. >=90% identity within aligned region, >=90% coverage of smaller cluster)
# - First column is the shorter cluster, and second column is the longer cluster.
# - Each row means that one or more contigs in the shorter cluster aligned to one or more contigs in the longer cluster, within the alignment similarity threshold.
head $PREFIX-mobmess_cluster_graph.txt
```

```
shorter_cluster	longer_cluster
0	0
0	1
1	1
1	2
1	3
1	4
1	5
1	6
1	7
```


# Reproduce plasmid systems from the study "The Genetic and Ecological Landscape of Plasmids in the Human Gut" by Michael Yu, Emily Fogarty, A. Murat Eren.
In this study, we predicted 226,194 plasmid contigs and then organized them into 1,169 plasmid systems. You can download a precomputed table that describes these plasmid systems.


```bash
# Download a precomputed table with information about plasmid systems in our study (~1 min on fast network)
wget https://zenodo.org/record/5819401/files/predicted_plasmids_systems.txt.gz
gunzip predicted_plasmids_systems.txt.gz

grep -v '#' predicted_plasmids_systems.txt | cut -f1-8 | head
```

```
system_name	backbone_cluster	number_of_backbone_plasmids	backbone_plasmids	compound_clusters	number_of_compound_clusters	number_of_compound_plasmids	compound_plasmids
PS1	1180	1	USA0047_01_000000006782	104257|104258|104259|104260|104261|104262|104263	7	26	AST0150_000000003279|AUS0014_000000004448|CHI0135_000000002208|CHI0150_000000001301|DEN0045_000000004247|ENG0024_000000002559|ENG0090_000000002209|ISR0003_000000003078|ISR0120_000000003329|ISR0222_000000004060|ISR0252_000000001013|ISR0259_000000001698|ISR0287_000000001305|ISR0343_000000001718|ISR0348_000000001086|ISR0376_000000002233|ISR0389_000000002379|ISR0802_000000001466|ISR0844_000000001897|ISR0852_000000000971|SPA0059_000000002718|SPA0107_000000003979|SPA0122_000000004474|USA0010_01_000000003640|USA0021_01_000000002558|USA0133_01_000000004011
PS2	1246	1	USA0044_01_000000001094	57453|62694|70008|71967|76572|122960|127080|127085	8	8	AST0019_000000000008|ENG0019_000000000010|ENG0060_000000000020|ENG0067_000000000023|ENG0174_000000000089|SPA0020_000000000102|SPA0027_000000000007|SPA0149_000000000166
PS3	1276	1	USA0041_01_000000014280	98822|105103|105106|105108	4	4	CHI0181_000000000733|DEN0089_000000000821|SPA0025_000000000866|SPA0146_000000000557
PS4	1425	1	USA0035_01_000000009056	113847	1	1	AST0059_000000003008
PS5	1660	1	USA0025_01_000000004006	127265|127280	2	5	AST0087_000000004237|CAN0006_000000006873|ISR0001_000000004559|ISR0201_000000004449|ISR0296_000000003088
PS6	1866	2	USA0018_01_000000008482|USA0039_01_000000008032	121213	1	2	USA0020_01_000000001444|USA0051_01_000000001114
PS7	2373	1	TAN0021_000000013580	21653|52208|69013	3	3	DEN0085_000000004815|ISR0812_000000018617|USA0031_01_000000004587
PS8	2793	1	TAN0007_000000033080	68390|68605	2	2	DEN0094_000000007794|ISR0810_000000006748
PS9	2940	1	TAN0005_000000004026	2941	1	1	TAN0007_000000005447
```

### Reproduce the table of plasmid systems

First, download a fasta file of the 226,194 predicted plasmid sequences


```bash
wget https://zenodo.org/record/5843600/files/predicted_plasmids.fa.gz
gunzip predicted_plasmids.fa.gz
```

Next, download information about their circularity (~1 min on fast network)


```bash
# Download a table that describe the plasmid sequences
wget https://zenodo.org/record/5819401/files/predicted_plasmids_summary.txt.gz
gunzip predicted_plasmids_summary.txt.gz
```


```bash
# Extract two columns from this table: the contig name and whether it is circular
grep -v '#' predicted_plasmids_summary.txt | cut -f1,18 | tail -n +2 > predicted_plasmids_circular.txt
```

```bash
head predicted_plasmids_circular.txt
```

```
MAD0004_000000000035	False
FIJ0168_000000000073	False
ISR0080_000000000116	False
SPA0003_000000000262	False
ISR0062_000000000346	True
MAD0056_000000000121	True
MAD0050_000000000091	False
AST0153_000000000532	False
ISR0063_000000000258	False
ENG0031_000000000248	False
```

Note that calculating pairwise alignments for this large set of plasmids will take a very long time (~10 days on 1 thread). You can shorten this time by increasing the number of threads.

Here, we will skip this intensive step by specifying a precomputed table of alignments using the `--ani` flag in `mobmess systems`.


```bash
# Download a precomputed table of alignment information (~5 min on fast network)
wget https://zenodo.org/record/5819401/files/predicted_plasmids_pairwise_alignments.txt.gz
gunzip predicted_plasmids_pairwise_alignments.txt.gz
```


```bash
# Infer plasmid systems (run `mobmess systems -h` to get help on parameters)
# (~2 min)
mobmess systems \
    --sequences predicted_plasmids.fa \
    --complete predicted_plasmids_circular.txt \
    --ani predicted_plasmids_pairwise_alignments.txt \
    --output predicted_plasmids_mobmess \
    --threads $THREADS
```


```bash
# Show table of plasmid systems
# - Compare this with the precomputed table, `predicted_plasmids_systems.txt`, that was downloaded above. They should be very similar.
# - Note that some of the numbering might differ slightly, due to randomness in how clusters and systems are numbered.
head predicted_plasmids_mobmess_systems.txt
```

```
system_name	backbone_cluster	number_of_backbone_plasmids	backbone_plasmids	compound_clusters	number_of_compound_clusters	number_of_compound_plasmids	compound_plasmids
PS1	1180	1	USA0047_01_000000006782	104243|104244|104245|104246|104247|104248|104249	7	26	AST0150_000000003279|AUS0014_000000004448|CHI0135_000000002208|CHI0150_000000001301|DEN0045_000000004247|ENG0024_000000002559|ENG0090_000000002209|ISR0003_000000003078|ISR0120_000000003329|ISR0222_000000004060|ISR0252_000000001013|ISR0259_000000001698|ISR0287_000000001305|ISR0343_000000001718|ISR0348_000000001086|ISR0376_000000002233|ISR0389_000000002379|ISR0802_000000001466|ISR0844_000000001897|ISR0852_000000000971|SPA0059_000000002718|SPA0107_000000003979|SPA0122_000000004474|USA0010_01_000000003640|USA0021_01_000000002558|USA0133_01_000000004011
PS2	1246	1	USA0044_01_000000001094	57444|62684|69998|71957|76558|122943|127062|127067	8	8	AST0019_000000000008|ENG0019_000000000010|ENG0060_000000000020|ENG0067_000000000023|ENG0174_000000000089|SPA0020_000000000102|SPA0027_000000000007|SPA0149_000000000166
PS3	1276	1	USA0041_01_000000014280	98807|105089|105092|105094	4	4	CHI0181_000000000733|DEN0089_000000000821|SPA0025_000000000866|SPA0146_000000000557
PS4	1425	1	USA0035_01_000000009056	113830	1	1	AST0059_000000003008
PS5	1660	1	USA0025_01_000000004006	127247|127262	2	5	AST0087_000000004237|CAN0006_000000006873|ISR0001_000000004559|ISR0201_000000004449|ISR0296_000000003088
PS6	1866	2	USA0018_01_000000008482|USA0039_01_000000008032	121195	1	2	USA0020_01_000000001444|USA0051_01_000000001114
PS7	2373	1	TAN0021_000000013580	21651|52204|69003	3	3	DEN0085_000000004815|ISR0812_000000018617|USA0031_01_000000004587
PS8	2793	1	TAN0007_000000033080	68380|68595	2	2	DEN0094_000000007794|ISR0810_000000006748
PS9	2940	1	TAN0005_000000004026	2941	1	1	TAN0007_000000005447
```
