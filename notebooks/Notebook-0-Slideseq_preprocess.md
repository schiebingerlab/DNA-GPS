# Preprocessing and downsampling Slide-seq reads

## Reconstituting FASTQ files from a BAM file

Since the Slide-seq datasets are published in the BAM format, first we converted the BAM file into FASTQ format:

```bash
samtools fastq 180528_23_kidney.bam | gzip -c > kidney_R2.fastq.gz
```

We next extracted bead barcodes and UMIs from the same BAM file and stored them in the R1 of the FASTQ files. Since the quality scores for these segments are not recoreded in the BAM file, we set the scores = 37 (F).

```bash
samtools fastq -T XC,XM 180528_23_kidney.bam | \
seqkit fx2tab | \
cut -f1,2,3 | \
awk 'BEGIN{OFS="\t"}{sub("XC:Z:","",$2);sub("XM:Z:","",$3);print $1,$2$3,"FFFFFFFFFFFFFFFFFFFFF"}' | \
seqkit tab2fx -o kidney_R1.fastq.gz
```



## Read structure conversion for 10X CellRanger using INTERSTELLAR

As our goal was to obtain a UMI count matrix, we thought it's easier to analyze the Slide-seq reads using a user-friendly all-in-one pipeline such as 10X CellRanger. Thus, we converted the Slide-seq read structure into 10X Chromium's one using [INTERSTELLAR](https://github.com/yachielab/Interstellar):

```bash
Interstellar -conf kidney.conf
```

Here is the content of ``kidney.conf``

```bash
[general]
# Working directory
PROJECT_DIR= Interstellar_to_CR/kidney

# Path to template shellscript that
SET_SHELL_ENV= ~/work/Interstellar_test/set_shell_env

[value_extraction]
# Input file path
READ1_PATH= ~/work/COAST/slide-seq/dataset/kidney/R1/
READ2_PATH= ~/work/COAST/slide-seq/dataset/kidney/R2/

# Read structure patterns by regular expression
READ1_STRUCTURE=^(?P<raw_cb>.{13})(?P<raw_umi>.{8})$
READ2_STRUCTURE=^(?P<raw_bio>.*)$

# Sequence filtering and correction for each segment
segment1.filtered.corrected.value =ALLOWLIST_CORRECT(source: raw_cb, levenshtein_distance:0,path:slide_kidney_bc.txt) >> SEQ2VALUE()

[value_translation]
#Value to destination sequence conversion
dest_segment1=VALUE2SEQ(source:segment1.filtered.corrected.value, allowlist_path:10x_3M-february-2018.txt)
dest_segment2=raw_umi
dest_segment3=raw_bio
dest_segment4="ATGC"

#Read structure configuration
READ1_STRUCTURE=dest_segment1+dest_segment2+dest_segment4
READ2_STRUCTURE=dest_segment3
```



## Downsampling

We used ``seqkit`` to downsample the converted reads. The following shell script was used:

```bash
#!/bin/bash

seed=$1
fq_read1=$2
fq_read2=$3
samp_rate=$4
outdir=$5
outname=$6

seqkit sample \
-s $1 \
-p $samp_rate \
-o $outdir/${outname}_rate${samp_rate}_s${seed}_R1.fastq.gz \
$fq_read1

seqkit sample \
-s $1 \
-p $samp_rate \
-o $outdir/${outname}_rate${samp_rate}_s${seed}_R2.fastq.gz \
$fq_read2
```

We downsampled the reads to 1/2, 1/4, ..., 1/32 (Conditions 1) and reads-per-beads (rpb) = 90, 70, 50, 30, 10 (Conditions 2). The commands below were used to perform the processes. 

```bash
# Conditions 1
x=0; \
for i in 0.5 0.25 0.125 0.0625 0.03125 0.021428571 0.016666667 0.011904762 0.007142857 0.002380952; \
do \
	x=$(expr $x + 1); \
	bash sampling.sh \
	$x \
	kidney/sample1/value_translation/out/translated_R1.fastq.gz \
	kidney/sample1/value_translation/out/translated_R2.fastq.gz \
	$i \
	downsampling/kidney/ \
	kidney; \
done

# Conditions 2
rpbs=(rpb10 rpb30 rpb50 rpb70 rpb90); \
rate=(0.002380952 0.007142857 0.011904762 0.016666667 0.021428571); \
for x in 0 1 2 3 4; \
do \
	bash sampling.sh \
	$x \
	kidney/sample1/value_translation/out/translated_R1.fastq.gz \
	kidney/sample1/value_translation/out/translated_R2.fastq.gz \
	${rate[x]} \
	downsampling/kidney/ \
	kidney_${rpbs}; \
done
```



## CellRanger analysis

Downsampled reads were analyized with 10X CellRanger v3.0.1.

```bash
cellranger count \
--id=$dirname \
--transcriptome=refdata-cellranger-mm10-3.0.0 \
--fastqs=downsampled_fastq_dir \
--localcores=8 \
--localmem=120 \
--chemistry SC3Pv3
```



## Generating gene expression count data

Following R script was used to collect the UMI count information.

```R
v0 <- c()
v1 <- c()
v2 <- c()

sampleing_condition <- c(
  "rpb10",
  "rpb30", 
  "rpb50", 
  "rpb70", 
  "rpb90"
  "kidney_rate003125",
  "kidney_rate00625", 
  "kidney_rate0125", 
  "kidney_rate025", 
  "kidney_rate05", 
  "kidney_rate1")

for(i in sampleing_condition){
  seu <- Read10X(paste0("CR/",i,"/outs/raw_feature_bc_matrix")) 
  seu <- seu[,colSums(seu)>0]
  csum <- colSums(seu)
  csum <- sort(csum,decreasing=T) %>% unname
  v0 <- c(v0,1:length(csum))
  v1 <- c(v1,csum)
  v2 <- c(v2,rep(i,length(csum)))}

df <- data.frame(rank=v0,count=v1,sample=v2)
```

