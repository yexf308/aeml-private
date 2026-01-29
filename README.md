# AutoEncoder Manifold Learning
This repository implements autoencoders using neural networks for manifold learning from point cloud data. It contains modules for generating synthetic data, training autoencoders, and running experiments to measure their comparative performance based on different training objectives/penalties.

All the code is written in python and uses standard libraries: numpy, scipy, pandas, matplotlib, and torch for the neural networks, and backpropagation. 

You can clone the repo with:

```bash
git clone https://github.com/shill1729/aeml.git
cd aeml
```

Create a new python environment (macOS/Linux):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

To run the penalty ablation experiment:
```bash
python run_experiments.py
```

add command line arguments (see penalty_ablation.py for full list):
```bash
python run_experiments.py --epochs 1000 --hidden_dim 32
```


## Mathematical overview
Let $(M, g)$ be a $d$-dimensional Riemannian manifold embedded in $\mathbb{R}^D$. We identify $M$ with its embedding. We observe a uniformly distributed point cloud: $X_1,\dotsc, X_n$ where $X_i$ are IID $\mathcal{U}(M)$ RVs given as points in the ambient space $\mathbb{R}^D$. We let $\pi_\theta: \mathbb{R}^D\to \mathbb{R}^d$ denote the encoder and
$\phi_\theta: \mathbb{R}^d\to \mathbb{R}^D$ denote the decoder. The parameter $\theta$ represents the joint collection of parameters. Here, both the encoder and decoder are parameterized as feed-forward neural networks. For example, for $L=2$ layers, the encoder has the form

$$\pi(x) =\tanh(W_2 \tanh(W_1 x+b_1)+b_2)$$

and the decoder has the form

$$\phi(z) = W_2'\tanh(W_1' z+b_1')+b_2',$$

where $\tanh(\cdot)$ is applied component-wise to vectors. The encoder is given a non-identity final activation layer so that, in the case of hyperbolic tangent, $\text{rng } \pi = (-1,1)^d$. If the weights are tied, then the decoder has the transpose of the weights of the encoder in reverse order: $W_1'=W_2^\top$ and $W_2'=W_1^\top$. 



Training a single auto-encoder means solving empirical risk minimization problem:

$$\min_{\theta} R_{n,0}(\theta)$$

where

$$R_{n,0}(\theta) = \frac{1}{n}\sum_{i=1}^{n} \lVert r_\theta (X_i)-X_i\rVert_2^2$$

where $r_\theta(x)=\phi_\theta \circ \pi_\theta(x)$ is the reconstructed point. In practice, this is done with stochastic gradient descent and backpropagation.

### Data generation
We generate data on manifolds in parameterized form $\phi(u,v)=(u,v, f(u,v))$ using importance-sampling. We use sympy to compute all the differential-geometry objects (e.g. the volume measure) and then use that to weight the importance-sampling to get uniformly distributed points in the local-coordinates. We then use the map 

to obtain ambient point cloud samples.
Similarly, we use sympy to compute the orthogonal projection onto $T_pM$ for a given point $p\in M$ and evaluate this at each point in the point cloud.




