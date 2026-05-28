# Lab notebook Jerry Liu

This notebook entails the engineering decisions made in this project, highlighting methodologies and the justifications behind them.

## Motivation
We want to design an estimator which allows us to estimate the latent state variables and input time series from the observations alone, given the dimensions of the input and latent states. The core challenge of this exercise is since we are only given the observations, the system dynamics and the inputs are both unknown, making this an ill-posed problem. In this exercise, we will choose to assume our system follows a Linear Gaussian model We will detail some of our explored methodologies as well as their theoretical backgrounds to demonstrate understanding. Mainly, we will focus our efforts on addressing the two issues : system identification (finding matrix parameters A,B,C,Q and R), and state/input estimation. 

## Methods
### Initial approach 
My initial approach begins from investigating methods for constructing a latent state representation, using our week1 intuition I attempted to extract orthogonal latent vectors at each time step from the observations. However, PCA does not capture linear dynamics, and since we are taking frozen-frames they do not provide any help for predicting the next set of latent variables at the next time step, and therefore only addresses the system identification, but does not help with state/input estimation. 

In order to encode linear-dynamics, an idea was to use a moving-window-average to create a series of linearly time-correlated PCA's, and therefore allow us to construct the linear model. 


### Subspace methods for System Identification with Stochastic Inputs

### EM algorithm with Kalman filters for estimation
The
### VAE and HMM

# 4. Design approach

The estimation problem posed here admits no unique solution: from the observations alone, the dynamics and the forcing are jointly unknown, and a continuum of (state, input) pairs reproduces any given output sequence. Rather than commit prematurely to a single estimator, we construct a sequence of models of increasing structure, each adding a *single* assumption to its predecessor, and assess each against the operational requirement that the estimator return **both** a latent-state trajectory and an input trajectory of the prescribed dimensions. This serves two purposes simultaneously. It satisfies the brief's call to explore alternatives, and it supplies the justification for the final choice, since each discarded model can be seen to fail a specific and identifiable requirement rather than being dismissed by assertion. The progression is unified by one theme, which we make explicit at its midpoint: a shift from representations optimised for *variance* to representations optimised for *predictability*.

## 4.1 Static principal component analysis

The simplest representation discards time entirely and seeks a low-dimensional linear summary of the marginal distribution of the observations. Writing $\mu$ for the sample mean and $\Sigma = \tfrac{1}{KT}\sum_{k,t}(y_t^{(k)}-\mu)(y_t^{(k)}-\mu)^\top$ for the pooled covariance, the latent is the projection onto the leading eigenvectors $W\in\mathbb{R}^{p\times n}$ of $\Sigma$,
$$
\hat{x}_t = W^\top (y_t - \mu). \tag{1}
$$
In the well-observed, high signal-to-noise regime — $p>n$ with $C$ of full column rank — the state is an instantaneous linear function of the observation, so the leading-variance subspace coincides with the column space of $C$. The recovered latent is then a linear image $M x_t$ of the true state and consequently inherits, to good approximation, the underlying linear dynamics. This is the regime in which static PCA performs deceptively well. Its limitation is nonetheless structural: the objective in (1) maximises retained variance, not temporal predictability, and it therefore cannot distinguish variance contributed by the state from variance contributed by the input or by measurement noise. Most decisively for the present task, the model contains no notion of a forcing signal, so it returns no input estimate and satisfies only one half of the interface. We retain it as a baseline and as an initialiser for the observation matrix $C$.

## 4.2 Moving-window-average PCA

The first attempt to introduce temporal structure smooths the static latent (or the observations) with a moving average of width $L$, $\tilde{x}_t = L^{-1}\sum_{i=0}^{L-1}\hat{x}_{t-i}$, on the premise that the latent varies slowly relative to the measurement noise. While this does suppress high-frequency noise, it is unsatisfactory on its own terms for two reasons. It imposes a *fixed, uniform* temporal weighting whose width is chosen by hand, an assumption about smoothness that the data are not consulted to justify; and the trailing average introduces a group delay of approximately $(L-1)/2$ samples. The latter is innocuous for a static reconstruction plot but actively harmful when the representation is later closed in a feedback loop, where the injected phase lag erodes stability margin. A centred average removes the delay only by becoming non-causal. The appropriate conclusion is not to abandon temporal averaging but to *learn* the temporal weighting rather than prescribe it.

## 4.3 Singular spectrum analysis via Hankel embedding

Learning the weighting is achieved by applying PCA not to $y_t$ but to a causal time-delay embedding, in which $L$ lagged copies of the observation are stacked into a trajectory vector,
$$
Y_t = \big[y_t^\top,\, y_{t-1}^\top,\, \ldots,\, y_{t-L+1}^\top\big]^\top \in \mathbb{R}^{pL}, \tag{2}
$$
and the eigenvectors of the resulting (block-Hankel) trajectory matrix are extracted. The eigenvectors are now *spatiotemporal* patterns, so the temporal weighting is data-driven rather than flat; this is multivariate singular spectrum analysis, equivalently dynamic PCA, and rests on the delay-embedding theorem (Takens, 1981; Broomhead and King, 1986). The construction recovers lagged and oscillatory structure to which static PCA is blind, and is genuinely advantageous when the system is *under-observed* ($p<n$): a single sample is then insufficient to determine the state, and history must be used to reconstruct it, an observability argument. When $p>n$, by contrast, the embedding offers no advantage for the latent and can dilute it by distributing $n$ informative directions across a $pL$-dimensional space; we observe this crossover empirically (Fig. 1). The method still provides no input channel. Its enduring contribution is that the Hankel SVD computed here is the very object exploited by the next rung.

## 4.4 From variance to predictability

The three preceding methods share a single objective: maximisation of retained variance. A latent that genuinely satisfies a state-space model must instead be a *Markov state* — the minimal function of the past that is sufficient for predicting the future, equivalently the statistic that renders past and future conditionally independent. Variance maximisation and predictive sufficiency coincide only in special cases, and in general the leading-variance directions of the embedding are not the directions that best predict the future. The principled replacement for the embedding's PCA is therefore a canonical correlation analysis between the past and future blocks of the Hankel matrix, which extracts precisely the subspace of the past most predictive of the future. This subspace is a Markov state by construction, and the conceptual step from §4.3 to §4.4 is exactly this substitution of a predictive criterion for a variance criterion.

## 4.5 Subspace identification

The past–future canonical analysis just described is subspace identification (Larimore, 1990; Van Overschee and De Moor, 1996). Forming an oblique projection of the future observations onto the past, followed by a reduced-rank singular value decomposition, yields an estimate of the extended observability matrix $\mathcal{O}_i = [C^\top, (CA)^\top, \ldots]^\top$ and a state sequence; the dynamics $A$ and the observation map $C$ then follow from the shift-invariance relation $\mathcal{O}_i^{\uparrow}A = \mathcal{O}_i^{\downarrow}$, solved by least squares. The estimates are consistent and, decisively, are obtained non-iteratively and so are free of the local optima that afflict likelihood-based fitting. In the present stochastic setting, where the forcing is unobserved, the algorithm treats the input as innovations and does not itself return an input trajectory. We therefore employ it not as the final estimator but as a warm start, supplying a consistent and reproducible initialisation to the iterative method below.

## 4.6 Expectation–maximisation with Kalman smoothing (selected method)

The final rung embeds the unknown input into the latent and fits a constrained augmented linear-Gaussian state-space model by expectation–maximisation, initialised from the subspace estimate of §4.5. The E-step is a Kalman filter followed by Rauch–Tung–Striebel smoothing; the M-step updates the system matrices in closed form. Alone among the rungs, it returns smoothed estimates of both the latent state and the input, as distinct quantities, while also producing a parametric plant $(A,B,C,D)$ that is directly usable for the control stage. It is detailed in §5.

We adopt this method. The selection is constructive rather than asserted: the models of §4.1–4.3 cannot return an input at all and optimise a variance criterion misaligned with state-space structure, while the subspace identifier of §4.5 returns dynamics but not the input under unobserved forcing. Only the augmented EM satisfies the complete interface while yielding a control-ready model. The discarded rungs are not wasted; they are retained as quantitative baselines (§6) and as components of the chosen pipeline, with PCA initialising $C$ and the Hankel SVD initialising the dynamics, so that the ladder is at once a comparison and a construction.


# 5. The estimator in detail

## 5.1 Model and the blind setting

We assume throughout a fixed linear-Gaussian plant,
$$
x_{t+1} = A x_t + B u_t + w_t, \quad w_t \sim \mathcal{N}(0,Q), \tag{3}
$$
$$
y_t = C x_t + D u_t + v_t, \quad v_t \sim \mathcal{N}(0,R), \tag{4}
$$
with $x_t\in\mathbb{R}^n$, $u_t\in\mathbb{R}^m$, $y_t\in\mathbb{R}^p$. The trials are independent observation sequences generated by this single plant under distinct input realisations, and the inputs are unavailable both during development and at evaluation. Identification is therefore *blind*. To recover the input we treat it as a latent process with an assumed prior, the simplest admissible choice being a driftless random walk $u_{t+1}=F u_t + \eta_t$ with $F=I$, and concatenate it with the state, $z_t=[x_t^\top,u_t^\top]^\top$. This yields an augmented model
$$
z_{t+1} = \underbrace{\begin{bmatrix} A & B \\ 0 & F \end{bmatrix}}_{\tilde{A}} z_t + \tilde{w}_t, \qquad
y_t = \underbrace{\begin{bmatrix} C & D \end{bmatrix}}_{\tilde{C}} z_t + v_t, \tag{5}
$$
with augmented process covariance $\tilde{Q}=\operatorname{diag}(Q,\Sigma_u)$ and $\tilde{R}=R$. The augmented system is an ordinary unsupervised linear-Gaussian model driven only by noise, to which standard filtering and smoothing apply.

## 5.2 Offline identification by constrained EM

The parameters are estimated by expectation–maximisation over all available trials. In the E-step, an RTS smoother applied to (5) for each trial returns the smoothed moments $\mathbb{E}[z_t\mid y_{1:T}]$, $\operatorname{Cov}[z_t\mid y_{1:T}]$ and the lag-one cross-covariance $\operatorname{Cov}[z_t,z_{t-1}\mid y_{1:T}]$, which are sufficient statistics for the update. In the M-step, the *free* blocks $(A,B,C,D,Q,R)$ are updated by the usual closed-form regressions of the smoothed state on its lag and of the observation on the smoothed state, while the input block $[0\ F]$ and the input noise $\Sigma_u$ are held fixed. Holding these blocks fixed is not an implementation convenience but the precise mechanism by which an "input" subspace is distinguished within the augmented state, as discussed in §5.3. Because blind likelihood maximisation is susceptible to local optima, the procedure is initialised from the subspace estimate of §4.5 and repeated from several random restarts, retaining the solution of highest marginal likelihood.

It is worth stating precisely what the multiplicity of trials contributes. Since the trials share the plant but differ in their inputs, pooling them improves the statistical efficiency of the shared-parameter estimates and, more importantly, provides the leverage to separate what is constant across trials, namely the plant, from what varies, namely the input realisations. It does **not** resolve the coordinate ambiguity within the augmented state, which is addressed next.

## 5.3 Identifiability and the role of the prior

From observations alone, the augmented system (5) is identifiable only up to a similarity transformation of its $(n+m)$-dimensional state. The decomposition of the augmented coordinates into $n$ state directions and $m$ input directions is therefore *not* determined by the data; it is imposed by two assumptions. The first is the exogeneity constraint embodied by the zero lower-left block of $\tilde{A}$, which encodes that the input drives the state but the state does not drive the input. This is a genuine structural restriction, since it is not preserved by an arbitrary similarity transform, and it accordingly carries real identifying information. The second is the assumed input model $(F,\Sigma_u)$, which distinguishes input coordinates from state coordinates by their temporal statistics. Even with both in place, a residual gauge freedom of block-upper-triangular form remains, so that the state is recovered only up to an invertible map $M_x$ and the input only up to an invertible map $M_u$.

The consequence, which we state plainly because it bears directly on the interpretation of the results, is that the matrix $B$ and the separation between state and input are identified *relative to the assumed input model, and are not measured from data*; there are no observed inputs against which to anchor them. Two implications follow. First, evaluation of the recovered trajectories must be invariant to the residual gauge, and we therefore report the coefficient of determination after an optimal linear alignment, which absorbs $M_x$ and $M_u$. Second, where the residual basis is itself of interest, it can be fixed by an additional prior that is not rotation-invariant — a sparsity penalty on $u_t$, or an independent-component analysis of the recovered input — at the cost of an explicit assumption about the character of the forcing. Finally, recovery of the input presupposes that the plant is left-invertible, which requires at least as many independent outputs as inputs ($p\ge m$) and the absence of blocking transmission zeros; a full-column-rank feedthrough $D$ secures instantaneous invertibility. As the plant approaches non-invertibility, the prior necessarily bears more of the identifying burden and the input estimate becomes correspondingly more dependent upon it.

## 5.4 Freezing and deployment

After convergence, the identified parameters $(A,B,C,D,Q,R)$ together with the frozen input prior $(F,\Sigma_u)$ are bound to the required interface signature through a partial application, so that no quantity is tuned at evaluation time, in conformity with the assessment protocol. The deployed estimator performs only the inexpensive operation of augmented RTS smoothing on the supplied observations; the leading $n$ smoothed coordinates are returned as the state estimate and the trailing $m$ as the input estimate, with shapes $(T,n)$ and $(T,m)$ respectively. The entire test-time computation is linear-Gaussian and deterministic, and is therefore fast and exactly reproducible, with the costly and local-optimum-prone identification confined to the offline stage.