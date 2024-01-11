<img src="./equiformer.png" width="450px"></img>

## Equiformer - Pytorch (wip)

Implementation of the <a href="https://arxiv.org/abs/2206.11990">Equiformer</a>, SE3/E3 equivariant attention network that reaches new SOTA, and adopted for use by <a href="https://www.biorxiv.org/content/10.1101/2022.10.07.511322v1">EquiFold (Prescient Design)</a> for protein folding

The design of this seems to build off of <a href="https://arxiv.org/abs/2006.10503">SE3 Transformers</a>, with the dot product attention replaced with MLP Attention and non-linear message passing from <a href="https://arxiv.org/abs/2105.14491">GATv2</a>. It also does a depthwise tensor product for a bit more efficiency. If you think I am mistakened, please feel free to email me.

Update: There has been a new development that makes scaling the number of degrees for SE3 equivariant networks dramatically better! <a href="https://arxiv.org/abs/2206.14331">This paper</a> first noted that by aligning the representations along the z-axis (or y-axis by some other convention), the spherical harmonics become sparse. This removes the m<sub>f</sub> dimension from the equation. <a href="https://arxiv.org/abs/2302.03655">A follow up paper</a> from Passaro et al. noted the Clebsch Gordan matrix has also become sparse, leading to removal of m<sub>i</sub> and l<sub>f</sub>. They also made the connection that the problem has been reduced from SO(3) to SO(2) after aligning the reps to one axis. <a href="https://arxiv.org/abs/2306.12059">Equiformer v2</a> (<a href="https://github.com/atomicarchitects/equiformer_v2">Official repository</a>) leverages this in a transformer-like framework to reach new SOTA.

Will definitely be putting more work / exploration into this. For now, I've incorporated the tricks from the first two paper for Equiformer v1, save for complete conversion into SO(2).

## Install

```bash
$ pip install equiformer-pytorch
```

## Usage

```python
import torch
from equiformer_pytorch import Equiformer

model = Equiformer(
    num_tokens = 24,
    dim = (4, 4, 2),               # dimensions per type, ascending, length must match number of degrees (num_degrees)
    dim_head = (4, 4, 4),          # dimension per attention head
    heads = (2, 2, 2),             # number of attention heads
    num_linear_attn_heads = 0,     # number of global linear attention heads, can see all the neighbors
    num_degrees = 3,               # number of degrees
    depth = 4,                     # depth of equivariant transformer
    attend_self = True,            # attending to self or not
    reduce_dim_out = True,         # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
    l2_dist_attention = False      # set to False to try out MLP attention
).cuda()

feats = torch.randint(0, 24, (1, 128)).cuda()
coors = torch.randn(1, 128, 3).cuda()
mask  = torch.ones(1, 128).bool().cuda()

out = model(feats, coors, mask) # (1, 128)

out.type0 # invariant type 0    - (1, 128)
out.type1 # equivariant type 1  - (1, 128, 3)
```

This repository also includes a way to decouple memory usage from depth using <a href="https://arxiv.org/abs/1707.04585">reversible networks</a>. In other words, if you increase depth, the memory cost will stay constant at the usage of one equiformer transformer block (attention and feedforward).

```python
import torch
from equiformer_pytorch import Equiformer

model = Equiformer(
    num_tokens = 24,
    dim = (4, 4, 2),
    dim_head = (4, 4, 4),
    heads = (2, 2, 2),
    num_degrees = 3,
    depth = 48,          # depth of 48 - just to show that it runs - in reality, seems to be quite unstable at higher depths, so architecture stil needs more work
    reversible = True,   # just set this to True to use https://arxiv.org/abs/1707.04585
).cuda()

feats = torch.randint(0, 24, (1, 128)).cuda()
coors = torch.randn(1, 128, 3).cuda()
mask  = torch.ones(1, 128).bool().cuda()

out = model(feats, coors, mask)

out.type0.sum().backward()
```

## Edges

with edges, ex. atomic bonds

```python
import torch
from equiformer_pytorch import Equiformer

model = Equiformer(
    num_tokens = 28,
    dim = 64,
    num_edge_tokens = 4,       # number of edge type, say 4 bond types
    edge_dim = 16,             # dimension of edge embedding
    depth = 2,
    input_degrees = 1,
    num_degrees = 3,
    reduce_dim_out = True
)

atoms = torch.randint(0, 28, (2, 32))
bonds = torch.randint(0, 4, (2, 32, 32))
coors = torch.randn(2, 32, 3)
mask  = torch.ones(2, 32).bool()

out = model(atoms, coors, mask, edges = bonds)

out.type0 # (2, 32)
out.type1 # (2, 32, 3)
```

with adjacency matrix

```python
import torch
from equiformer_pytorch import Equiformer

model = Equiformer(
    dim = 32,
    heads = 8,
    depth = 1,
    dim_head = 64,
    num_degrees = 2,
    valid_radius = 10,
    reduce_dim_out = True,
    attend_sparse_neighbors = True,  # this must be set to true, in which case it will assert that you pass in the adjacency matrix
    num_neighbors = 0,               # if you set this to 0, it will only consider the connected neighbors as defined by the adjacency matrix. but if you set a value greater than 0, it will continue to fetch the closest points up to this many, excluding the ones already specified by the adjacency matrix
    num_adj_degrees_embed = 2,       # this will derive the second degree connections and embed it correctly
    max_sparse_neighbors = 8         # you can cap the number of neighbors, sampled from within your sparse set of neighbors as defined by the adjacency matrix, if specified
)

feats = torch.randn(1, 128, 32)
coors = torch.randn(1, 128, 3)
mask  = torch.ones(1, 128).bool()

# placeholder adjacency matrix
# naively assuming the sequence is one long chain (128, 128)

i = torch.arange(128)
adj_mat = (i[:, None] <= (i[None, :] + 1)) & (i[:, None] >= (i[None, :] - 1))

out = model(feats, coors, mask, adj_mat = adj_mat)

out.type0 # (1, 128)
out.type1 # (1, 128, 3)
```

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> for the generous sponsorship, as well as my other sponsors out there

## Testing

Tests for equivariance etc

```bash
$ python setup.py test
```

## Example

First install `sidechainnet`

```bash
$ pip install sidechainnet
```

Then run the protein backbone denoising task

```bash
$ python denoise.py
```

## Todo

- [x] move xi and xj separate project and sum logic into Conv class
- [x] move self interacting key / value production into Conv, fix no pooling in conv with self interaction
- [x] go with a naive way to split up contribution from input degrees for DTP
- [x] for dot product attention in higher types, try euclidean distance
- [x] consider a all-neighbors attention layer just for type0, using linear attention

- [ ] integrate the new finding from spherical channels paper, followed up by so(3) -> so(2) paper, which reduces the computation from O(L^6) -> O(L^3)!
    - [x] add rotation matrix -> ZYZ euler angles
    - [x] function for deriving rotation matrix for r_ij -> (0, 1, 0)
    - [x] prepare get_basis to return D for rotating representations to (0, 1, 0) to greatly simplify spherical harmonics
    - [x] add tests for batch rotating vectors to align with another - handle edge cases (0, 0, 0)?
    - [x] redo get_basis to only calculate spherical harmonics Y for (0, 1, 0) and cache
    - [x] do the further optimization to remove clebsch gordan (since m_i only depends on m_o), as noted in eSCN paper
    - [x] validate one can train at higher degrees
    - [x] figure out the whole linear bijection argument in appendix of eSCN and why parameterized lf can be removed
    - [x] figure out why training NaNs with float32
    - [ ] refactor into full so3 -> so2 linear layer, as proposed in eSCN paper
    - [ ] add equiformer v2, and start looking into equivariant protein backbone diffusion again

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

```bibtex
@article{Shazeer2019FastTD,
    title   = {Fast Transformer Decoding: One Write-Head is All You Need},
    author  = {Noam M. Shazeer},
    journal = {ArXiv},
    year    = {2019},
    volume  = {abs/1911.02150}
}
```

```bibtex
@misc{ding2021cogview,
    title   = {CogView: Mastering Text-to-Image Generation via Transformers},
    author  = {Ming Ding and Zhuoyi Yang and Wenyi Hong and Wendi Zheng and Chang Zhou and Da Yin and Junyang Lin and Xu Zou and Zhou Shao and Hongxia Yang and Jie Tang},
    year    = {2021},
    eprint  = {2105.13290},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@inproceedings{Kim2020TheLC,
    title   = {The Lipschitz Constant of Self-Attention},
    author  = {Hyunjik Kim and George Papamakarios and Andriy Mnih},
    booktitle = {International Conference on Machine Learning},
    year    = {2020}
}
```

```bibtex
@article{Zitnick2022SphericalCF,
    title   = {Spherical Channels for Modeling Atomic Interactions},
    author  = {C. Lawrence Zitnick and Abhishek Das and Adeesh Kolluru and Janice Lan and Muhammed Shuaibi and Anuroop Sriram and Zachary W. Ulissi and Brandon C. Wood},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2206.14331}
}
```

```bibtex
@article{Passaro2023ReducingSC,
  title     = {Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs},
  author    = {Saro Passaro and C. Lawrence Zitnick},
  journal   = {ArXiv},
  year      = {2023},
  volume    = {abs/2302.03655}
}
```

```bibtex
@inproceedings{Gomez2017TheRR,
    title   = {The Reversible Residual Network: Backpropagation Without Storing Activations},
    author  = {Aidan N. Gomez and Mengye Ren and Raquel Urtasun and Roger Baker Grosse},
    booktitle = {NIPS},
    year    = {2017}
}
```

```bibtex
@article{Bondarenko2023QuantizableTR,
    title   = {Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing},
    author  = {Yelysei Bondarenko and Markus Nagel and Tijmen Blankevoort},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2306.12929},
    url     = {https://api.semanticscholar.org/CorpusID:259224568}
}
```

```bibtex
@inproceedings{Arora2023ZoologyMA,
  title   = {Zoology: Measuring and Improving Recall in Efficient Language Models},
  author  = {Simran Arora and Sabri Eyuboglu and Aman Timalsina and Isys Johnson and Michael Poli and James Zou and Atri Rudra and Christopher R'e},
  year    = {2023},
  url     = {https://api.semanticscholar.org/CorpusID:266149332}
}
```

