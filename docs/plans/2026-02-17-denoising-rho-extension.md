# Denoising Extension of the $\rho$-Metric Generalization Theory

**Date:** 2026-02-17
**Status:** Draft sketch
**References:**
- Current paper: `Autoencoder-Paper/paper.tex`, Section 5 (Generalization Theory)
- Liu et al. (2303.09863): "Deep Nonparametric Estimation of Intrinsic Data Structures by Chart Autoencoders"
- New base-point Lipschitz lemma: `Autoencoder-Paper/main.tex`, lines 269--296
- Boissonnat et al. (2019): "The reach, metric distortion, geodesic convexity and the variation of tangent spaces"
- Federer (1959): "Curvature Measures", Theorem 4.8(8)

---

## Motivation

The current generalization theorem (Theorem 5.3 in `paper.tex`) assumes:
- Data is sampled in chart coordinates $x_i \sim \mathrm{Unif}(U)$, i.e., **on** the manifold
- The true tangent-space projector $P_\star(x_i)$ is known exactly (from population-level $\Lambda$)
- The encoder does not appear in the theory

In practice:
- Observations may be slightly off-manifold (finite timescale separation $\epsilon > 0$)
- The geometric labels $\hat{P}$ are estimated from finite short bursts with statistical error
- The covariance may be estimated at landmarks $z_j \neq x_i$, introducing displacement bias

This document sketches a denoising extension that:
1. Handles noisy off-manifold inputs (à la Liu et al.)
2. Handles noisy geometric labels (practical ATLAS setting)
3. Gives the new base-point Lipschitz lemma a central role
4. Introduces the reach $\tau(M)$ as a quantitative parameter in the generalization theory

---

## Background: Why Liu et al.'s Setting 3.1 Does Not Directly Apply

Liu et al. assume paired clean/noisy data $(x_i, v_i)$ where $v_i \in M$ is a clean label and $x_i = v_i + w_i$ is noisy. Their loss is pure reconstruction:

$$\frac{1}{m}\sum_i \|v_i - \mathcal{D} \circ \mathcal{E}(x_i)\|^2$$

This does not fit the AEML paper because:
- We do not have clean labels $v_i$ separate from the observations
- Our loss includes geometric penalties (tangent-bundle, curvature), not just reconstruction
- Our data comes from SDE trajectories, not a static point cloud

However, the key ingredients from Liu et al. are reusable:
- The reach $\tau(M)$ controls the Lipschitz constant of the nearest-point projection $\pi_M$
- The oracle encoder is $\mathcal{E}_\star = \phi_\star^{-1} \circ \pi_M$
- Manifold-adapted network architectures achieve rates depending on intrinsic $d$, not ambient $D$

---

## Setting

Let $M$ be a compact, smooth, $d$-dimensional Riemannian submanifold of $\mathbb{R}^D$ with positive reach $\tau := \tau(M) > 0$. Let $\phi_\star : U \to \mathbb{R}^D$ be the ground-truth chart map with inverse $\pi_\star := \phi_\star^{-1} : M \to U$.

### Data model

Training data consist of $m$ triples $\{(\tilde{x}_i, \tilde{v}_i, \hat{P}_i)\}_{i=1}^m$ generated as follows:

1. **Observations:** $\tilde{x}_i$ are i.i.d. draws from a probability measure $\mu$ supported on the $q$-tubular neighborhood $\mathcal{M}(q)$ with $q < \tau$.

2. **On-manifold projections:** $v_i := \pi_M(\tilde{x}_i)$. Since $q < \tau$, the nearest-point projection is unique. Write $\tilde{x}_i = v_i + w_i^x$ where $w_i^x := \tilde{x}_i - v_i \in (T_{v_i}M)^\perp$ with $\|w_i^x\| \leq q$. The induced distribution $\gamma := (\pi_M)_\# \mu$ is the pushforward measure on $M$. (Off-manifold displacement from finite timescale separation or measurement noise.)

3. **Noisy reconstruction labels:** $\tilde{v}_i = v_i + w_i^v$ where $w_i^v = \beta(v_i) + \eta_i$ with:
   - Bias: $\beta : M \to \mathbb{R}^D$ with $\|\beta\|_\infty \leq \epsilon_{\mathrm{bias}}$
   - Mean-zero residual: $\mathbb{E}[\eta_i \mid v_i, w_i^x] = 0$, $\|\eta_i\| \leq \sigma_v$

   (In the ATLAS setting, $\tilde{v}_i = v_i$ if the landmark is used directly, i.e., $\sigma_v = 0$ and $\epsilon_{\mathrm{bias}} = 0$.)

4. **Noisy geometric labels:** In the ATLAS pipeline, the covariance $\Lambda$ is estimated from short bursts launched at landmarks $z_j \in \mathcal{M}(q)$ (which may themselves be off-manifold). Let $\hat{\Lambda}_j$ denote the estimated covariance at $z_j$, and $\hat{P}_i := \mathrm{proj}_d(\hat{\Lambda}_{j(i)})$ the rank-$d$ projector onto its leading eigenspace, where $j(i)$ is the nearest landmark to $\tilde{x}_i$. Then:

   $$\|\hat{P}_i - P_\star(v_i)\|_F \leq \underbrace{\|\hat{P}_i - P_\star(\pi_M(z_j))\|_F}_{\text{statistical + off-manifold at landmark}} + \underbrace{\|P_\star(\pi_M(z_j)) - P_\star(v_i)\|_F}_{\text{on-manifold displacement}}$$

   We write $\|\hat{P}_i - P_\star(v_i)\|_F \leq \epsilon_P$ with the understanding that $\epsilon_P$ absorbs three noise sources: (a) finite-burst covariance estimation error, (b) the landmark being off-manifold ($z_j \notin M$), and (c) spatial displacement between $\pi_M(z_j)$ and $v_i$.

### Bounded loss (needed for oracle inequality)

The noisy per-sample loss $\hat{\ell}_\rho$ is uniformly bounded: $M$ is compact, $U$ is bounded, $\phi_\theta$ and $\mathcal{E}_\theta$ are Lipschitz-bounded (Assumption D3), $\|\tilde{v}_i - v_i\| \leq \sigma_v + \epsilon_{\mathrm{bias}}$, and $\hat{P}_i$ is a projector ($\|\hat{P}_i\|_F = \sqrt{d}$). Hence there exists $L_{\max} = L_{\max}(B, s_0, \sigma_v, \epsilon_{\mathrm{bias}}, d, \lambda) < \infty$ such that $\hat{\ell}_\rho \leq L_{\max}$ a.s.

### Special cases

| Case | $q$ | $\sigma_v$ | $\epsilon_{\mathrm{bias}}$ | $\epsilon_P$ | Description |
|------|-----|-----------|-------------------------|-------------|-------------|
| Current paper | $0$ | $0$ | $0$ | $0$ | On-manifold, exact labels |
| Liu et al. | $> 0$ | $0$ | $0$ | N/A | Noisy input, clean labels, no geometric penalty |
| Noisy geometry only | $0$ | $0$ | $0$ | $> 0$ | On-manifold, noisy $\hat{P}$ from finite bursts |
| Full denoising | $> 0$ | $> 0$ | small | $> 0$ | Most general |

---

## Assumptions

### Assumption D1 (Target chart regularity)

$\phi_\star \in W^{n,\infty}(U)$ with $\|\phi_\star\|_{W^{n,\infty}} \leq 1$ for some $n > 1$, and $\sigma_{\min}(D\phi_\star(u)) \geq s_0 > 0$ for all $u \in U$.

*Same as Assumption 4.1 in the current paper.*

### Assumption D2 (Manifold geometry)

$M$ has reach $\tau > 0$. The nearest-point projection $\pi_M : \mathcal{M}(q) \to M$ is well-defined on the $q$-tubular neighborhood $\mathcal{M}(q) := \{x \in \mathbb{R}^D : d(x, M) \leq q\}$ with:

1. **Regularity:** $\pi_M$ is $C^{n-1}$ on $\mathcal{M}(q)$ (since $M$ is $C^n$ and $q < \tau$).

2. **Lipschitz constant:**
$$\|\pi_M\|_{\mathrm{Lip}(\mathcal{M}(q))} \leq \frac{\tau}{\tau - q}$$
(Federer 1959, Theorem 4.8(8)).

3. **Derivative formula:** For $x \in \mathcal{M}(q) \setminus M$ with $t = \|x - \pi_M(x)\|$ and unit normal $\nu = (x - \pi_M(x))/t$,
$$D\pi_M(x) = \big(\mathrm{id}_{T_{\pi_M(x)}M} - t\, L_{\pi_M(x),\nu}\big)^{-1} P_\star(\pi_M(x))$$
where $L_{\cdot,\nu}$ is the Weingarten map (shape operator) in direction $\nu$.

4. **Base-point Lipschitz property of the tangent-space projector** (new lemma): For $x, y \in M$,
$$\|P_\star(x) - P_\star(y)\|_F \leq \frac{\sqrt{2d}}{\tau}\,\|x - y\|_2.$$

*This assumption is new. The reach $\tau$ and property (4) are the entry points for the new base-point Lipschitz lemma.*

### Assumption D3 (Hypothesis class)

The encoder $\mathcal{E}_\theta : \mathbb{R}^D \to \mathbb{R}^d$ and decoder $\phi_\theta : \mathbb{R}^d \to \mathbb{R}^D$ belong to constrained network classes:

- **(Decoder)** $\phi_\theta \in \mathcal{H}_{W,B,s}^{\mathrm{dec}}$: a DeNN class with $\|\phi_\theta\|_{W^{1,\infty}(U)} \leq B$ and $\sigma_{\min}(D\phi_\theta) \geq s_0/2$ a.e. on $U$.

- **(Encoder)** $\mathcal{E}_\theta \in \mathcal{H}^{\mathrm{enc}}$: a network class with $\|\mathcal{E}_\theta\|_{\mathrm{Lip}(\mathcal{M}(q))} \leq L_{\mathrm{enc}}$ and $\mathcal{E}_\theta(\mathcal{M}(q)) \subseteq U$.

*The decoder class is the same as Assumption 4.2 in the current paper. The encoder class is new.*

### Assumption D4 (Approximation)

There exist $\theta^\sharp$ with $(\mathcal{E}_{\theta^\sharp}, \phi_{\theta^\sharp}) \in \mathcal{H}^{\mathrm{enc}} \times \mathcal{H}^{\mathrm{dec}}$ such that:

1. **(Decoder approximation)**
$$\|\phi_{\theta^\sharp} - \phi_\star\|_{W^{1,\infty}(U)} \leq \delta_{\mathrm{dec}}$$
with $\delta_{\mathrm{dec}} \leq s_0/2$.

2. **(Encoder approximation)**
$$\|\mathcal{E}_{\theta^\sharp} - \mathcal{E}_\star\|_{L^\infty(\mathcal{M}(q))} \leq \delta_{\mathrm{enc}}$$
where $\mathcal{E}_\star := \pi_\star \circ \pi_M : \mathcal{M}(q) \to U$ is the oracle encoder.

*The decoder approximation is the same as Assumption 4.3 in the current paper. The encoder approximation is new and requires the oracle encoder $\mathcal{E}_\star$ to be well-approximated by networks.*

**Remark on encoder approximation rate.** The oracle encoder $\mathcal{E}_\star = \phi_\star^{-1} \circ \pi_M$ is a $C^{n-1}$ function from $\mathcal{M}(q) \subset \mathbb{R}^D$ to $U \subset \mathbb{R}^d$, with Lipschitz constant $\tau/(s_0(\tau - q))$. Naive approximation on $\mathbb{R}^D$ would give rates depending on ambient dimension $D$. Following Liu et al., manifold-adapted architectures can achieve $\delta_{\mathrm{enc}}$ depending on intrinsic dimension $d$ by exploiting the low-dimensional structure of $\mathcal{M}(q)$.

---

## Loss and Risk

### Noisy empirical $\rho$-loss (what we minimize)

$$\hat{R}_{S,\rho}(\theta) := \frac{1}{m}\sum_{i=1}^m \hat{\ell}_\rho(\tilde{x}_i, \tilde{v}_i, \hat{P}_i;\, \theta)$$

where

$$\hat{\ell}_\rho(\tilde{x}, \tilde{v}, \hat{P};\, \theta) := \|\phi_\theta(\mathcal{E}_\theta(\tilde{x})) - \tilde{v}\|^2 + \frac{\lambda}{2}\|P_{\phi_\theta}(\mathcal{E}_\theta(\tilde{x})) - \hat{P}\|_F^2$$

### Clean population $\rho$-risk (what we want to control)

$$R_\rho(\theta) := \mathbb{E}_{v \sim \gamma,\, w^x}\bigg[\|\phi_\theta(\mathcal{E}_\theta(\tilde{x})) - v\|^2 + \frac{\lambda}{2}\|P_{\phi_\theta}(\mathcal{E}_\theta(\tilde{x})) - P_\star(v)\|_F^2\bigg]$$

where $v = \pi_M(\tilde{x})$ and $\tilde{x} = v + w^x$ with $\|w^x\| < \tau$.

**Note:** The clean risk measures convergence to the true manifold point $v$ and the true tangent-space projector $P_\star(v)$, even though the ERM uses noisy surrogates $\tilde{v}$ and $\hat{P}$.

---

## Theorem Statement

**Theorem (Generalization of noisy $\rho$-ERM).**
Let Assumptions D1--D4 hold. Let $\hat{\theta}$ minimize the noisy empirical $\rho$-loss $\hat{R}_{S,\rho}$ over $\mathcal{H}^{\mathrm{enc}} \times \mathcal{H}^{\mathrm{dec}}$. Then

$$\mathbb{E}\,R_\rho(\hat{\theta}) \leq C\bigg[\underbrace{\delta_{\mathrm{dec}}^2 + \delta_{\mathrm{enc}}^2\,(1 + s_0^{-2})}_{\text{(I) approximation}} + \underbrace{\frac{\mathrm{Pdim}(\mathcal{H})\,\log m}{m}}_{\text{(II) estimation}} + \underbrace{\epsilon_{\mathrm{bias}}^2}_{\text{(III) label bias}} + \underbrace{\sqrt{d}\,\epsilon_P}_{\text{(IV) geometric noise}}\bigg]$$

where $C = C(d, D, B, s_0, \tau, q, \sigma_v, \lambda)$ and $\mathrm{Pdim}(\mathcal{H})$ is the pseudo-dimension of the composed encoder--decoder hypothesis class.

### Consequences

**(i) Rate preservation.** Under the DeNN approximation rates from Assumption D4 and manifold-adapted encoder architectures, terms (I) and (II) yield

$$m^{-\frac{2(n-1)}{2(n-1)+d}}$$

up to logarithmic factors --- the **same rate** as the clean $\rho$-ERM theorem (Theorem 5.3), depending on intrinsic dimension $d$, not ambient $D$.

**(ii) Label bias floor.** Term (III) is an irreducible $O(\epsilon_{\mathrm{bias}}^2)$ that does not improve with sample size $m$. In the ATLAS setting, $\epsilon_{\mathrm{bias}} = O(\epsilon)$ where $\epsilon$ is the timescale separation, so this vanishes as $\epsilon \to 0$.

**(iii) Geometric noise via the base-point Lipschitz lemma.** Term (IV) decomposes as

$$\epsilon_P \leq \epsilon_{\mathrm{stat}}(N, q) + \frac{\sqrt{2d}}{\tau(M)}\,h$$

where:
- $\epsilon_{\mathrm{stat}}(N, q)$ = covariance estimation error, depending on burst count $N$ and off-manifold displacement $q$ (the off-manifold location of landmarks biases $\hat{\Lambda}$; this is absorbed into $\epsilon_{\mathrm{stat}}$)
- $h = \max_i \min_j \|v_i - z_j^\star\|$ = on-manifold fill distance between data projections $v_i = \pi_M(\tilde{x}_i)$ and landmark projections $z_j^\star = \pi_M(z_j)$
- The second term is controlled by the **new base-point Lipschitz lemma** (Assumption D2.4)

The reach $\tau(M)$ enters the generalization theory as a quantitative geometric parameter: it controls both the Lipschitz constant of $\pi_M$ (how off-manifold error propagates) and the base-point variation of $P_\star$ (how landmark sparsity affects geometric label quality).

---

## Proof Sketch

### Step 0: Noise decomposition

Decompose the noisy loss into the clean loss plus perturbation:

$$\hat{\ell}_\rho = \ell_\rho^{\mathrm{clean}} + \Delta_{\mathrm{label}} + \Delta_{\mathrm{geom}}$$

**Label noise perturbation** ($\tilde{v}$ vs $v$):

$$\Delta_{\mathrm{label}} = \|\phi_\theta(\mathcal{E}_\theta(\tilde{x})) - \tilde{v}\|^2 - \|\phi_\theta(\mathcal{E}_\theta(\tilde{x})) - v\|^2 = \|w^v\|^2 + 2\langle w^v,\, v - \phi_\theta(\mathcal{E}_\theta(\tilde{x}))\rangle$$

Taking expectations: $\mathbb{E}[\Delta_{\mathrm{label}} \mid v, \tilde{x}] = \mathbb{E}[\|w^v\|^2] + 2\langle \beta(v),\, v - \phi_\theta(\mathcal{E}_\theta(\tilde{x}))\rangle$.

- The term $\mathbb{E}[\|\eta\|^2] + \|\beta(v)\|^2$ contributes a $\theta$-independent constant plus a bias piece. The constant does not affect the minimizer.
- The cross term $2\langle \beta(v),\, v - \phi_\theta(\mathcal{E}_\theta(\tilde{x}))\rangle$ is bounded via Young's inequality $2ab \leq a^2 + b^2$: for any $\alpha > 0$,
  $$2\|\beta\|_\infty \cdot \|v - \phi_\theta(\mathcal{E}_\theta(\tilde{x}))\| \leq \alpha\,\|v - \phi_\theta(\mathcal{E}_\theta(\tilde{x}))\|^2 + \frac{\epsilon_{\mathrm{bias}}^2}{\alpha}.$$
  Choosing $\alpha$ small enough to absorb the first term into the reconstruction loss gives an additive $O(\epsilon_{\mathrm{bias}}^2)$ contribution, matching term (III) in the theorem.

For **mean-zero** label noise ($\beta \equiv 0$): the conditional expectation of $\Delta_{\mathrm{label}}$ is a $\theta$-independent constant, so the population minimizer is unchanged. At finite $m$, the mean-zero cross terms $\langle \eta_i, v_i - \phi_\theta(\mathcal{E}_\theta(\tilde{x}_i))\rangle$ contribute variance (not bias) to the ERM; this is absorbed into the estimation term (II) with the constant $C$ depending on $\sigma_v$. The rate is preserved.

**Geometric noise perturbation** ($\hat{P}$ vs $P_\star$):

$$|\Delta_{\mathrm{geom}}| = \frac{\lambda}{2}\big|\|P_{\phi_\theta} - \hat{P}\|_F^2 - \|P_{\phi_\theta} - P_\star\|_F^2\big| \leq 2\lambda\sqrt{d}\,\|\hat{P} - P_\star\|_F \leq 2\lambda\sqrt{d}\,\epsilon_P$$

using $|a^2 - b^2| \leq (|a|+|b|)|a-b|$ and the fact that rank-$d$ orthogonal projectors satisfy $\|P\|_F = \sqrt{\mathrm{tr}(P)} = \sqrt{d}$.

### Step 1: Approximation

The oracle autoencoder $(\mathcal{E}_\star, \phi_\star)$ with $\mathcal{E}_\star = \pi_\star \circ \pi_M$ achieves $R_\rho(\theta_\star) = 0$: by construction $\phi_\star(\mathcal{E}_\star(\tilde{x})) = \phi_\star(\pi_\star(v)) = v$ and $P_{\phi_\star}(\mathcal{E}_\star(\tilde{x})) = P_\star(v)$ for every $\tilde{x} \in \mathcal{M}(q)$, since $v = \pi_M(\tilde{x})$ and $w^x \in (T_v M)^\perp$ by the projection decomposition.

**Decoder contribution:** By Lemma 4.5 (`rho-approx-from-W1infty`), the decoder approximation error contributes $O(\delta_{\mathrm{dec}}^2)$ to the $\rho$-risk, exactly as in the current proof.

**Encoder contribution:** The encoder error $\|\mathcal{E}_\theta(\tilde{x}) - \mathcal{E}_\star(\tilde{x})\| \leq \delta_{\mathrm{enc}}$ propagates through the decoder:

- Reconstruction: $\|\phi_\star(\mathcal{E}_\theta(\tilde{x})) - \phi_\star(\mathcal{E}_\star(\tilde{x}))\| \leq \|\phi_\star\|_{\mathrm{Lip}} \cdot \delta_{\mathrm{enc}} \leq \|D\phi_\star\|_{L^\infty(U)}\,\delta_{\mathrm{enc}}$
- Projection: $\|P_{\phi_\star}(\mathcal{E}_\theta(\tilde{x})) - P_{\phi_\star}(\mathcal{E}_\star(\tilde{x}))\|_F \leq \frac{\sqrt{2}}{s_0} \cdot \|D\phi_\star\|_{\mathrm{Lip}(U)} \cdot \delta_{\mathrm{enc}}$

Combined with decoder error via triangle inequality: approximation term is $O(\delta_{\mathrm{dec}}^2 + \delta_{\mathrm{enc}}^2(1 + s_0^{-2}))$.

### Step 2: Estimation (covering numbers)

The covering number of the composed loss class is bounded by covering both the encoder and decoder classes:

$$\log \mathcal{N}(\varepsilon, \mathcal{L}_\rho, m) \leq \log \mathcal{N}(c\varepsilon, \mathcal{H}^{\mathrm{enc}}, m) + \log \mathcal{N}(c\varepsilon, \mathcal{F}^{\mathrm{dec}}, m)$$

where $\mathcal{F}^{\mathrm{dec}}$ is the first-order feature class of the decoder (as in the current proof, Step 2). The pseudo-dimension of the composed class is bounded by the sum of the individual pseudo-dimensions.

The estimation error is $O(\mathrm{Pdim}(\mathcal{H}) \log m / m)$, with the same structure as the current proof.

### Step 3: Geometric label noise via base-point Lipschitz lemma

The geometric label $\hat{P}_i$ is estimated at a landmark $z_j \in \mathcal{M}(q)$ that may be off-manifold. Let $z_j^\star := \pi_M(z_j)$ be the on-manifold projection of the landmark. The error decomposes into two terms:

$$\|\hat{P}_i - P_\star(v_i)\|_F \leq \underbrace{\|\hat{P}_i - P_\star(z_j^\star)\|_F}_{\text{(a) statistical + off-manifold}} + \underbrace{\|P_\star(z_j^\star) - P_\star(v_i)\|_F}_{\text{(b) on-manifold displacement}}$$

**Term (a):** The covariance $\hat{\Lambda}_j$ is estimated from bursts launched at $z_j$, not at $z_j^\star \in M$. The ambient SDE coefficients at $z_j$ differ from the on-manifold coefficients at $z_j^\star$ by $O(q)$ (since $\Lambda$ is Lipschitz and $\|z_j - z_j^\star\| \leq q$). Combined with finite-burst statistical error: $\|\hat{P}_i - P_\star(z_j^\star)\|_F \leq \epsilon_{\mathrm{stat}}(N, q)$.

**Term (b):** The on-manifold displacement is controlled by the **new base-point Lipschitz lemma**:

$$\|P_\star(z_j^\star) - P_\star(v_i)\|_F \leq \frac{\sqrt{2d}}{\tau(M)}\,\|z_j^\star - v_i\|_2$$

Combining: $\epsilon_P \leq \epsilon_{\mathrm{stat}}(N, q) + \frac{\sqrt{2d}}{\tau}\,h$ where $h = \max_i \min_j \|v_i - z_j^\star\|$ is the on-manifold fill distance (between data projections $v_i = \pi_M(\tilde{x}_i)$ and landmark projections $z_j^\star = \pi_M(z_j)$).

**This is where the reach $\tau(M)$ enters the generalization bound as a quantitative parameter**, controlling:
- The Lipschitz constant of $\pi_M$ (bounding how off-manifold error propagates)
- The base-point variation of $P_\star$ (bounding displacement bias)
- The trade-off between landmark density and geometric label quality

### Step 4: Combine

Applying the oracle inequality to the noisy empirical $\rho$-risk and collecting the four terms gives the stated bound.

---

## Role of the New Lemma

The base-point Lipschitz lemma enters the theory at **Step 3**. Without it, one can only say "the true projector $P_\star$ is Lipschitz on the compact $M$, hence the displacement bias is bounded" --- with an implicit, non-quantitative constant. The new lemma makes this **explicit**:

$$C_P = \frac{\sqrt{2d}}{\tau(M)}$$

This has interpretable consequences:
- **Curvature sensitivity:** smaller $\tau$ (more curved $M$) requires denser landmarks ($h \lesssim \tau\,\epsilon_{\mathrm{target}} / \sqrt{2d}$)
- **Dimension dependence:** the $\sqrt{d}$ factor reflects the number of principal curvature directions
- **Design guideline:** to achieve geometric label accuracy $\epsilon_P$, the landmark fill distance must satisfy $h \leq (\epsilon_P - \epsilon_{\mathrm{stat}}) \cdot \tau / \sqrt{2d}$

---

## Recovery of Existing Results

Setting $q = 0$, $\sigma_v = 0$, $\epsilon_{\mathrm{bias}} = 0$, $\epsilon_P = 0$:
- Terms (III) and (IV) vanish
- The encoder is trivial ($\mathcal{E}_\star = \pi_\star$, no noise to handle), so $\delta_{\mathrm{enc}} = 0$
- The bound reduces to $O(\delta_{\mathrm{dec}}^2 + \mathrm{Pdim}/m)$, recovering Theorem 5.3

---

## Open Questions

1. **Encoder approximation without ambient dimension dependence.** Can we use Liu et al.'s manifold-adapted architectures to guarantee $\delta_{\mathrm{enc}}$ depends on $d$, not $D$? This is the key technical step borrowed from their paper.

2. **Statistical rate for $\epsilon_{\mathrm{stat}}$.** What is the convergence rate of $\hat{P}$ to $P_\star$ at a landmark $z_j$, as a function of burst count $N$ and burst length $\tau$? This connects to the ATLAS estimation theory.

3. **Curvature penalty under noise.** The current sketch only handles the tangent-bundle ($T$) penalty in the $\rho$-loss. Extending to the curvature ($K$) penalty would require controlling $\|\hat{q} - q\|$ (noisy Ito correction), which involves the Hessian. This is a harder problem.

4. **Multi-chart extension.** With multiple charts, the reach controls chart domain overlaps and transition map regularity. The new lemma would be needed to control how $P_\star$ varies across chart boundaries.
