# Lab notebook Jerry Liu

This notebook entails the engineering decisions made in this project, highlighting methodologies and the justifications behind them.

## 1. Motivation
We want to design an estimator which allows us to estimate the latent state variables and input time series from the observations alone, given the dimensions of the input and latent states. The core challenge of this exercise is since we are only given the observations, the system dynamics and the inputs are both unknown, making this an ill-posed problem. In this exercise, we will choose to assume our system follows a Linear Gaussian model We will detail some of our explored methodologies as well as their theoretical backgrounds to demonstrate understanding. Mainly, we will focus our efforts on addressing the two issues : system identification (finding matrix parameters A,B,C,Q and R), and state/input estimation. 

## 2. Preliminary work: the `l4b` toolkit and its role in the estimation pipeline

Week 1 produced `l4b`, a library that operationalises the linear-Gaussian state-space model and supplies the analytical primitives that the estimation pipeline consumes. The `Simulator` integrates the forward model for arbitrary $(A,B,C,Q,R)$; `InputSignal`/`InputBuilder` constructs canonical forcing patterns; a system-theoretic suite computes controllability and observability Gramians, eigen-spectra, and eigenpair decompositions; `KalmanFilter` implements both time-varying and steady-state (DARE) recursions; and `Illustrator` exposes a statistical battery — trial and trial-averaged traces, correlation, PCA, autocorrelation, cross-correlation, power spectra, and magnitude-squared coherence — through a single `plot_all()` call. These are not diagnostic accessories but direct inputs to the Week 2 pipeline, as detailed below.

### Justifying the linear-Gaussian model

The core modelling assumption is that the observations $y_t$ are conditionally Gaussian given the latent state, and that the state evolves linearly. The statistical tools expose the conditions under which this assumption is defensible. The **power spectrum** is the first test: for a stable LG-SSM driven by white Gaussian noise, the output spectrum is a smooth rational function of frequency with no sharp peaks, entirely determined by the poles of $A$ and the noise variances. A power spectrum dominated by low frequencies with no narrow-band peaks is therefore consistent with the model, while harmonic lines or broadband non-monotone structure would demand a non-linear or non-Gaussian alternative. Similarly, **autocorrelation** of the observations should decay as a mixture of real exponentials and damped sinusoids, one term per dynamic mode; departure from this form — a slower-than-exponential tail, or a non-decaying periodic component — would signal that the Markov-order-$n$ assumption is violated or that the true dynamics are non-linear. After fitting, the same tools provide a model-validation step: the innovations sequence produced by the Kalman filter should be white and approximately Gaussian if the model is correctly specified.

### Choosing the latent dimension and warm-starting the parameters

**PCA** on the observations identifies the effective observation rank and sets a ceiling on the meaningful latent dimension $n$: any component whose explained-variance ratio falls into the noise floor adds no predictive content. The leading $n$ eigenvectors directly initialise the observation matrix $C$ before EM begins, replacing a random start with a data-anchored one and narrowing the search region for the M-step. The **autocorrelation** of the leading PCA scores provides a complementary warm start for $A$: under a diagonal $A$, the lag-1 autocorrelation of each score equals the corresponding eigenvalue, so the diagonal of $A_\text{init}$ can be set to the vector of lag-1 autocorrelations, placing the initial spectral radius near its empirical value. Together, these two steps — $C$ from `l4b.stats.PCA` components, $A$ from `l4b.stats.Autocorrelation` of the PCA scores — are the concrete way in which the Week 1 toolkit feeds into the Week 2 pipeline and are directly implemented in the `_warm_start` function of the estimator.

### Diagnosing shared latent structure

The **cross-correlation** matrix and **coherence** matrix serve complementary roles. Zero-lag cross-correlation is a fast indicator of instantaneous co-modulation: high pairwise correlation implies that the pair shares a common latent factor, which is the baseline assumption of the low-rank $C$ model. Cross-correlations that peak at a non-zero lag indicate directed coupling through $A$, informing whether off-diagonal structure in $A$ is likely to be needed. Coherence is the frequency-resolved analogue: high magnitude-squared coherence at low frequencies across all channel pairs confirms that a single slow shared latent dominates — the same conclusion PCA draws, but now resolved by frequency and therefore more diagnostic of dynamic coupling versus static co-loading. High coherence only at low frequencies combined with near-zero coherence at high frequencies also supports a diagonal observation-noise covariance $R$, since high-frequency content is then approximately channel-independent. The `KalmanFilter` experiment in Week 1 — estimating states from simulated data with known parameters — served as an end-to-end sanity check confirming that the recursions recover the true state when the plant is fully observable, and that the resulting innovations are white, as the model requires.

## 3. Methods

The methods we explore are developed in detail in §4 (Design approach) of the
interim report, organised as a ladder of increasing structure: static PCA, a
moving-window-averaged variant, singular spectrum analysis on a delay
embedding, and the chosen method -- expectation-maximisation on a linear-
Gaussian state-space model with the unknown input folded into an augmented
state, fitted by Kalman filtering and Rauch-Tung-Striebel smoothing. Each rung
is presented there together with the requirement it fails to meet, and the
chosen method is the only one that satisfies the full interface contract
(returning both latent states and inputs as distinct, gauge-aligned quantities)
while also producing a parametric plant directly reusable in the control stage.

# 4. Design approach

The estimation problem posed here admits no unique solution: from the observations alone, the dynamics and the forcing are jointly unknown, and a continuum of (state, input) pairs reproduces any given output sequence. Rather than commit prematurely to a single estimator, we construct a sequence of models of increasing structure, each adding a *single* assumption to its predecessor, and assess each against the operational requirement that the estimator return **both** a latent-state trajectory and an input trajectory of the prescribed dimensions. This serves two purposes simultaneously: it satisfies the brief's call to explore alternatives, and it supplies the justification for the final choice, since each discarded model can be seen to fail a specific and identifiable requirement rather than being dismissed by assertion. The progression is unified by a shift from representations optimised for *variance* to representations optimised for *predictability*, which we make explicit at the transition to the chosen method.

## 4.1 Static principal component analysis

The simplest representation discards time entirely and seeks a low-dimensional linear summary of the marginal distribution of the observations. Writing $\mu$ for the sample mean and $\Sigma = \tfrac{1}{T}\sum_{t}(y_t-\mu)(y_t-\mu)^\top$ for the sample covariance, the latent is the projection onto the leading eigenvectors $W\in\mathbb{R}^{p\times n}$ of $\Sigma$,
$$
\hat{x}_t = W^\top (y_t - \mu). \tag{1}
$$
In the well-observed, high signal-to-noise regime — $p>n$ with $C$ of full column rank — the state is an instantaneous linear function of the observation, so the leading-variance subspace coincides with the column space of $C$. The recovered latent is then a linear image $M x_t$ of the true state and consequently inherits, to good approximation, the underlying linear dynamics. This is the regime in which static PCA performs deceptively well. Its limitation is nonetheless structural: the objective in (1) maximises retained variance, not temporal predictability, and it therefore cannot distinguish variance contributed by the state from variance contributed by the input or by measurement noise. Most decisively for the present task, the model contains no notion of a forcing signal, so it returns no input estimate and satisfies only one half of the interface. We retain it as a baseline (§6) and as the initialiser for the observation matrix $C$ in the chosen method.

## 4.2 Moving-window-average PCA

The first attempt to introduce temporal structure smooths the static latent (or the observations) with a moving average of width $L$, $\tilde{x}_t = L^{-1}\sum_{i=0}^{L-1}\hat{x}_{t-i}$, on the premise that the latent varies slowly relative to the measurement noise. While this does suppress high-frequency noise, it is unsatisfactory on its own terms for two reasons. It imposes a *fixed, uniform* temporal weighting whose width is chosen by hand, an assumption about smoothness that the data are not consulted to justify; and the trailing average introduces a group delay of approximately $(L-1)/2$ samples. The latter is innocuous for a static reconstruction plot but actively harmful when the representation is later closed in a feedback loop, where the injected phase lag erodes stability margin. A centred average removes the delay only by becoming non-causal. The appropriate conclusion is not to abandon temporal averaging but to *learn* the temporal weighting from data rather than prescribe it.

## 4.3 Singular spectrum analysis via Hankel embedding

Learning the weighting is achieved by applying PCA not to $y_t$ but to a causal time-delay embedding, in which $L$ lagged copies of the observation are stacked into a trajectory vector,
$$
Y_t = \big[y_t^\top,\, y_{t-1}^\top,\, \ldots,\, y_{t-L+1}^\top\big]^\top \in \mathbb{R}^{pL}, \tag{2}
$$
and the eigenvectors of the resulting (block-Hankel) trajectory matrix are extracted. The eigenvectors are now *spatiotemporal* patterns, so the temporal weighting is data-driven rather than flat; this is multivariate singular spectrum analysis, equivalently dynamic PCA, and rests on the delay-embedding theorem (Takens, 1981; Broomhead and King, 1986). The construction recovers lagged and oscillatory structure to which static PCA is blind, and is genuinely advantageous when the system is *under-observed* ($p<n$): a single sample is then insufficient to determine the state, and history must be used to reconstruct it, an observability argument. When $p>n$, by contrast, the embedding offers no advantage for the latent and can dilute it by distributing $n$ informative directions across a $pL$-dimensional space; we observe this crossover empirically (Fig. 1). The method still provides no input channel. Its enduring contribution is conceptual: it shows that lagged correlation can be learned from data, motivating the move to a model that *generates* this temporal structure rather than merely extracting it.

## 4.4 From variance to predictability: linear-Gaussian state-space models

The three preceding methods share a single objective: maximisation of retained variance. A latent that genuinely satisfies a state-space model must instead be a *Markov state* — the minimal function of the past sufficient for predicting the future, equivalently the statistic that renders past and future conditionally independent. Variance maximisation and predictive sufficiency coincide only in special cases, and in general the leading-variance directions of an embedding are not the directions that best predict the future. We therefore posit a generative model in which the latent evolves linearly and the observation is a linear function of it, the linear-Gaussian state-space model, and seek the latent that is optimal under this model. The smoothing of §4.2 reappears, but the weights are no longer chosen: they are determined by the model and the data jointly, computed exactly by the Kalman filter and Rauch–Tung–Striebel smoother of §5.

## 4.5 Expectation–maximisation with Kalman smoothing (selected method)

The chosen estimator embeds the unknown input into the latent and fits a constrained augmented linear-Gaussian state-space model by expectation–maximisation. The E-step is a Kalman filter and RTS smoother, which together compute the data-optimal latent given the current parameter estimates; the M-step updates the system matrices in closed form by regression on the smoothed moments. Identification is initialised from the principal-component projection of §4.1, supplying a sensible $C$ and a regression-based $A$ on the PCA scores, which acts as a deterministic warm start in place of the more elaborate subspace-identification procedures (Van Overschee and De Moor, 1996); we found PCA initialisation sufficient on our test data, with the residual local-optimum risk handled by a small number of random restarts.

Alone among the methods considered, this approach returns smoothed estimates of both the latent state and the input as distinct quantities, while producing a parametric plant $(A,B,C,D)$ that is directly usable for the control stage. We adopt it for these reasons. The selection is constructive rather than asserted: the models of §4.1–4.3 cannot return an input at all and optimise a variance criterion misaligned with state-space structure. The discarded rungs are retained as baselines (§6) and as components of the pipeline, with PCA initialising $C$, so the ladder is at once a comparison and a construction. The detailed model and its identifiability are developed in §5.


# 5. The estimator in detail

## 5.1 Model and the blind setting

We assume a fixed linear-Gaussian plant,
$$
x_{t+1} = A x_t + B u_t + w_t, \quad w_t \sim \mathcal{N}(0,Q), \tag{3}
$$
$$
y_t = C x_t + D u_t + v_t, \quad v_t \sim \mathcal{N}(0,R), \tag{4}
$$
with $x_t\in\mathbb{R}^n$, $u_t\in\mathbb{R}^m$, $y_t\in\mathbb{R}^p$. The inputs are unobserved. To recover them we treat each as a latent process with a driftless random-walk prior $u_{t+1}=u_t+\eta_t$, $\eta_t\sim\mathcal{N}(0,\Sigma_u)$, and concatenate it with the state, $z_t=[x_t^\top,u_t^\top]^\top$. This yields the augmented model
$$
z_{t+1} = \underbrace{\begin{bmatrix} A & B \\ 0 & I \end{bmatrix}}_{\tilde{A}} z_t + \tilde{w}_t, \qquad
y_t = \underbrace{\begin{bmatrix} C & D \end{bmatrix}}_{\tilde{C}} z_t + v_t, \tag{5}
$$
with augmented process covariance $\tilde{Q}=\operatorname{diag}(Q,\Sigma_u)$ and $\tilde{R}=R$. The augmented system is an ordinary unsupervised linear-Gaussian model driven only by noise; the Kalman filter and RTS smoother apply directly, and one mechanism — augmented-state smoothing — produces both the state and the input as the leading $n$ and trailing $m$ smoothed coordinates respectively.

## 5.2 Identification by constrained EM

The Kalman smoother requires the system matrices. We estimate them jointly with the latent trajectory by expectation–maximisation on (5). In the E-step, the RTS smoother returns the smoothed moments $\mathbb{E}[z_t\mid y_{1:T}]$, $\operatorname{Cov}[z_t\mid y_{1:T}]$ and the lag-one cross-covariance, which are sufficient statistics. In the M-step, the *free* blocks $(A,B,C,D,Q,R)$ are updated by closed-form linear regression of the smoothed state on its lag and of the observation on the smoothed state, while the input block $[0\ I]$ and the input-prior covariance $\Sigma_u$ are held fixed. Holding these blocks fixed is the mechanism by which an "input" subspace is distinguished within the augmented state, as discussed in §5.3.

Because blind likelihood maximisation is non-convex, the procedure is initialised from the principal-component projection of the observations — $C$ from the leading $n$ singular vectors of the centred observation matrix, $A$ from one-step least-squares regression on the PCA scores — and repeated from a small number of random restarts of the remaining initial values, retaining the highest-likelihood fit. This is a deterministic guard against local optima rather than a tuning step; the restart count is frozen before submission.

## 5.3 Identifiability and the role of the prior

From observations alone, the augmented system (5) is identifiable only up to a similarity transformation of its $(n+m)$-dimensional state. The decomposition of the augmented coordinates into $n$ state directions and $m$ input directions is therefore *not* determined by the data; it is imposed by two assumptions. The first is the exogeneity constraint embodied by the zero lower-left block of $\tilde{A}$, which encodes that the input drives the state but the state does not drive the input. This is a genuine structural restriction, since it is not preserved by an arbitrary similarity transform, and it accordingly carries real identifying information. The second is the input model $(F=I,\Sigma_u)$, which distinguishes input coordinates from state coordinates by their temporal statistics. Even with both in place, a residual block-upper-triangular gauge freedom remains, so the state is recovered only up to an invertible map $M_x$ and the input only up to an invertible map $M_u$.

The consequence, which we state plainly because it bears directly on the interpretation of results, is that $B$ and the separation between state and input are identified *relative to the assumed input model, and are not measured from data*; there are no observed inputs against which to anchor them. Two implications follow. Evaluation of the recovered trajectories must be invariant to the residual gauge, and we therefore report the coefficient of determination after an optimal linear alignment, which absorbs $M_x$ and $M_u$. Further, the random-walk prior penalises rapid variation in $u_t$, which fits smoothly-varying inputs well but systematically biases the estimate for inputs whose energy is concentrated in sharp transitions (square waves, step changes) or isolated samples (impulse trains); the smoother smears these and attenuates their amplitude. Recovery of the input also presupposes that the plant is left-invertible, which requires $p\ge m$ and the absence of blocking transmission zeros; a full-column-rank $D$ secures instantaneous invertibility. As the plant approaches non-invertibility, the prior bears more of the identifying burden and the estimate becomes correspondingly more prior-dependent.

Two principled extensions address the AR(1) limitation and would be natural directions for further work: replacing the Gaussian increment penalty by a heavy-tailed or sparsity-promoting prior, which preserves the conditional-Gaussian structure and yields a reweighted Kalman smoother; and parameterising $u_t$ as a linear combination of a fixed basis (Fourier, B-spline, indicator) with static coefficients, which reduces the input degrees of freedom from $T\times m$ to $K\times m$ and represents oscillatory and impulsive inputs exactly with the appropriate basis.

## 5.4 Deployment

All hyperparameters — the iteration cap, restart count, convergence tolerance, and the form of the input prior — are fixed in code before submission. The input-prior scale $\Sigma_u$ is set deterministically from the supplied observation increments; this is a fixed rule, not a quantity tuned at test time. The function receives a single observation array, performs identification on it by the constrained EM of §5.2, and then performs augmented RTS smoothing under the identified parameters. The leading $n$ smoothed coordinates are returned as the state estimate and the trailing $m$ as the input estimate, with shapes $(T,n)$ and $(T,m)$ respectively. The smoothing pass is linear-Gaussian and deterministic; the cost and the local-optimum risk are confined to the identification stage, whose iteration cap and restart count are chosen so that the call completes well within the evaluation window.







