# DNA-GPS: A theoretical framework for large-scale optics-free spatial transcriptomics
![Reconstructions](https://github.com/schiebingerlab/GPS-seq/blob/master/aux_files/header.png)

DNA-GPS is a theoretical framework for large-scale optics-free spatial transcriptomics that combines high-throughput sequencing with manifold learning. Similar to technologies like Slide-seq [1] and 10X Visium [2], tissue samples are stamped on a surface of DNA-barcoded "anchors", such as spots or beads. Instead of relying on depositing known barcodes or retrospectively identifying barcodes through imaging, our approach localizes beads through the use of "satellite" barcodes (sBCs). sBCs diffuse locally resulting in each bead having a sBC transcriptome determined by its spatial position. We show that the transcriptome profiles of the anchors form a 2D manifold in the high dimensional sBC space, which can be recovered using manifold learning. Our simulations show that beads can be localized to within 10-30 $\mu m$ of their ground truth positions, depending on sequencing depth. For further details, see our [preprint](https://www.biorxiv.org/content/10.1101/2022.03.22.485380v2).


Here we provide a walk-through of the data simulation and reconstruction process, as well as provide precomputed data allowing you to recreate the sweep figures from the preprint. Each notebook can be run individually or they can be run in series to progress from data generation to a summary of what spatial resolution can be achieved across a wide range of parameters.

### Citing
Please cite the [preprint](https://www.biorxiv.org/content/10.1101/2022.03.22.485380v2).

The DNA-based global positioning system — a theoretical framework for large-scale spatial genomics

Laura Greenstreet, Anton Afanassiev, Yusuke Kijima, Matthieu Heitz, Soh Ishiguro, Samuel King, Nozomu Yachie, Geoffrey Schiebinger

### References
[1] Rodriques, S.G. et al. Slide-seq: A scalable technology for measuring genome-wide expression at high spatial resolution. Science, 2019. https://doi:10.1126/science.aaw1219.

[2] Stahl, P.L. et al. Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. Science, 2016. https://doi:10.1126/science.aaf2403.
