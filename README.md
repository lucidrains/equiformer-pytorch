## Equiformer - Pytorch (wip)

Implementation of the <a href="https://arxiv.org/abs/2206.11990">Equiformer</a>, SE3/E3 equivariant attention network that reaches new SOTA, and adopted for use by <a href="https://www.biorxiv.org/content/10.1101/2022.10.07.511322v1">EquiFold (Prescient Design)</a> for protein folding

This repository may eventually contain an implementation of EquiFold as well, with the added FAPE loss + structural violation checks

## Citations

```bibtex
@article{Liao2022EquiformerEG,
    title   = {Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs},
    author  = {Yi Liao and Tess E. Smidt},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2206.11990}
}
```

```bibtex
@article {Lee2022.10.07.511322,
    author  = {Lee, Jae Hyeon and Yadollahpour, Payman and Watkins, Andrew and Frey, Nathan C. and Leaver-Fay, Andrew and Ra, Stephen and Cho, Kyunghyun and Gligorijevic, Vladimir and Regev, Aviv and Bonneau, Richard},
    title   = {EquiFold: Protein Structure Prediction with a Novel Coarse-Grained Structure Representation},
    elocation-id = {2022.10.07.511322},
    year    = {2022},
    doi     = {10.1101/2022.10.07.511322},
    publisher = {Cold Spring Harbor Laboratory},
    URL     = {https://www.biorxiv.org/content/early/2022/10/08/2022.10.07.511322},
    eprint  = {https://www.biorxiv.org/content/early/2022/10/08/2022.10.07.511322.full.pdf},
    journal = {bioRxiv}
}
```
