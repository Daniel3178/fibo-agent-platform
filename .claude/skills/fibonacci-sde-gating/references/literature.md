# Literature (from the 4 research agents, 2026-10-27)

† means the citation came from the agent's memory and was not checked online. Every other entry was checked by web search for venue and URL.

## Mixture of experts: objectives, theory, collapse

- **Jacobs, Jordan, Nowlan & Hinton 1991.** Adaptive mixtures of local experts. *Neural Computation* 3(1):79-87. https://doi.org/10.1162/neco.1991.3.1.79
  - Cooperative vs competitive error functions.
  - Only the competitive (mixture-likelihood) one localises experts.
- **Jordan & Jacobs 1994.** Hierarchical mixtures of experts and the EM algorithm. *Neural Computation* 6(2):181-214. https://doi.org/10.1162/neco.1994.6.2.181
  - Responsibilities and EM.
  - Gate inputs may differ from expert inputs.
- **Bishop 1994.** Mixture Density Networks. Aston NCRG/94/004. https://publications.aston.ac.uk/id/eprint/373/
- **Weigend, Mangeas & Srivastava 1995.** Nonlinear gated experts for time series: discovering regimes and avoiding overfitting. *Int. J. Neural Systems* 6:373-399. https://research.cs.wisc.edu/areas/ai/aisem/abstracts/1995.1.spring/weigend.html
  - The gate segments regimes.
  - Each expert's variance matches the local noise. **Expect a split by volatility.**
- **Chen, Deng, Wu, Gu & Li 2022.** Towards understanding MoE in deep learning. NeurIPS. https://arxiv.org/abs/2208.02813
  - MoE helps when the inputs have cluster structure that the router can see.
- **Krishnamurthy, Watkins & Gärtner 2023.** Improving expert specialization in MoE. https://arxiv.org/abs/2302.14703
  - Soft gates find unintuitive decompositions.
  - Proposes a similarity/dissimilarity loss.
- **Chi et al. 2022.** On the representation collapse of sparse MoE. NeurIPS. https://arxiv.org/abs/2204.09179
- **Nguyen et al. 2023.** Identifiability and convergence of softmax-gated Gaussian MoE. NeurIPS. https://arxiv.org/abs/2305.03288 (details unverified)

## Load balancing and routing

- **Shazeer et al. 2017.** Outrageously large neural networks (noisy top-k, importance and load losses). ICLR. https://arxiv.org/abs/1701.06538
- **Lepikhin et al. 2021.** GShard. ICLR. https://arxiv.org/abs/2006.16668
- **Fedus, Zoph & Shazeer 2022.** Switch Transformers. Aux loss `α·K·Σ f_i P_i`, α = 1e-2. JMLR. https://arxiv.org/abs/2101.03961
- **Zoph et al. 2022.** ST-MoE (router z-loss). https://arxiv.org/abs/2202.08906
- **Zhou et al. 2022.** MoE with Expert Choice routing. NeurIPS. https://arxiv.org/abs/2202.09368
- **Lewis et al. 2021.** BASE layers. ICML. https://arxiv.org/abs/2103.16716
- **Wang et al. 2024.** Auxiliary-loss-free load balancing. https://arxiv.org/abs/2408.15664
  - See also DeepSeek-V3: https://arxiv.org/abs/2412.19437
- **Dai et al. 2022.** StableMoE (routing fluctuation; distil and freeze the router). ACL. https://aclanthology.org/2022.acl-long.489
- **Xue et al. 2024.** OpenMoE. Routing is fixed early and driven by the input token rather than its context. ICML. https://arxiv.org/abs/2402.01739
- **Jiang et al. 2024.** Mixtral of Experts. No topic specialisation, but local routing consistency. https://arxiv.org/abs/2401.04088
- **Lu et al. 2024.** Expert pruning and skipping by loss increase. ACL. https://aclanthology.org/2024.acl-long.334

## Hard and sparse gates

- **Jang, Gu & Poole 2017.** Gumbel-Softmax. ICLR. https://arxiv.org/abs/1611.01144
- **Maddison, Mnih & Teh 2017.** Concrete distribution. https://arxiv.org/abs/1611.00712
- **Martins & Astudillo 2016.** Sparsemax. ICML. https://arxiv.org/abs/1602.02068
- **Peters, Niculae & Martins 2019.** Sparse sequence-to-sequence models (entmax). ACL. https://arxiv.org/abs/1905.05702
- **Hazimeh et al. 2021.** DSelect-k. NeurIPS. https://arxiv.org/abs/2106.03760

## Specialisation, supervision, classifier routing

- **Chen et al. 2023.** Mod-Squad: maximise I(task; expert). CVPR. https://arxiv.org/abs/2212.08066
- **Gross, Ranzato & Szlam 2017.** Hard mixtures of experts. CVPR. https://arxiv.org/abs/1704.06363
- **Kudugunta et al. 2021.** Task-level MoE. Findings of EMNLP. https://arxiv.org/abs/2110.03742
- **Gururangan et al. 2022.** DEMix layers. NAACL. https://aclanthology.org/2022.naacl-main.407
- **Sukhbaatar et al. 2024.** Branch-Train-MiX. https://arxiv.org/abs/2403.07816
- **Ye & Borde 2026.** Regime-gated residual MoE for cross-sectional volatility. https://arxiv.org/abs/2608.12251 (abstract only)
  - Regime variables help when restricted to the gate.
- **Lim, Arık, Loeff & Pfister 2021.** Temporal Fusion Transformers (variable selection networks). *IJF*. https://arxiv.org/abs/1912.09363
- **Grinsztajn, Oyallon & Varoquaux 2022.** MLPs degrade with uninformative features. NeurIPS D&B. https://arxiv.org/abs/2207.08815
- **Ratner et al. 2017.** Snorkel (weak supervision). VLDB. https://arxiv.org/abs/1711.10160
- **Hu et al. 2016.** Harnessing DNNs with logic rules. ACL. https://arxiv.org/abs/1603.06318
- **Ganchev et al. 2010.** Posterior regularization. JMLR. †
- **Perez et al. 2018.** FiLM. AAAI. †

## Regime switching

- **Hamilton 1989.** *Econometrica* 57(2):357-384. †
- **Bengio & Frasconi 1995.** Input-output HMMs. NeurIPS 7. †
- **Ghahramani & Hinton 2000.** Switching state-space models. *Neural Computation*. †
- **Linderman et al. 2017.** Recurrent SLDS. AISTATS. †
- **Nystrup, Lindström & Madsen 2020.** Persistent-state HMMs via jump penalties. *ESWA*. (partly confirmed)

## Neural SDEs and synthetic financial data

- **Li, Wong, Chen & Duvenaud 2020.** Scalable gradients for SDEs (latent SDE). AISTATS. https://proceedings.mlr.press/v108/li20i.html
- **Kidger, Foster, Li, Oberhauser & Lyons 2021.** Neural SDEs as infinite-dimensional GANs. ICML. https://proceedings.mlr.press/v139/kidger21b.html
- **Issa, Horvath, Lemercier & Salvi 2023.** Non-adversarial training of Neural SDEs with signature kernel scores. NeurIPS. https://arxiv.org/abs/2305.16274
- **Gierjatowicz et al. 2022.** Robust pricing and hedging via neural SDEs. *J. Comput. Finance* 26(3). https://arxiv.org/abs/2007.04154
- **Liao et al. 2023.** Conditional Sig-Wasserstein GANs. *Math. Finance*. https://arxiv.org/abs/2006.05421
- **Venkatraman, Hebert & Bagnell 2015.** Improving multi-step prediction (DaD). AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/9590
- **Brandstetter, Worrall & Welling 2022.** Message passing neural PDE solvers (pushforward trick). ICLR. https://arxiv.org/abs/2202.03376
- **Bengio et al. 2015.** Scheduled sampling. NeurIPS. https://arxiv.org/abs/1506.03099 †
- **Cont 2001.** Empirical properties of asset returns: stylized facts. *Quant. Finance* 1(2). https://doi.org/10.1080/713665670 †
- **Wiese et al. 2020.** Quant GANs. *Quant. Finance* 20(9). https://arxiv.org/abs/1907.06673
- **Buehler et al. 2020.** A data-driven market simulator (signature MMD). https://arxiv.org/abs/2006.14498
- **Yoon, Jarrett & van der Schaar 2019.** TimeGAN. NeurIPS. †
- **Esteban, Hyland & Rätsch 2017.** TSTR/TRTS. https://arxiv.org/abs/1706.02633 †
- **Brigo et al.** OU/Vasicek MLE toolkit. https://arxiv.org/abs/0812.4210
- **Hansen & Lunde 2005.** Does anything beat a GARCH(1,1)? *J. Appl. Econometrics* 20(7). https://doi.org/10.1002/jae.800

## Support/resistance and Fibonacci evidence

- **Osler 2000.** Support for resistance. FRBNY EPR 6(2). https://www.newyorkfed.org/research/epr/00v06n2/0007osle.html
- **Osler 2003.** Currency orders and exchange-rate dynamics. *J. Finance* 58(5). † Staff report: https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr125.pdf
- **Lo, Mamaysky & Wang 2000.** Foundations of technical analysis. *J. Finance* 55(4). https://doi.org/10.1111/0022-1082.00265 †
- **Garzarelli et al. 2014.** Memory effects in stock price dynamics (bounces on S/R). https://arxiv.org/abs/1110.5197
- **Chung & Bellotti 2021.** Evidence and behaviour of support and resistance levels. https://arxiv.org/abs/2101.07410
- **Tsinaslanidis, Guijarro & Voukelatos 2022.** *Expert Syst. Appl.* 187 (abstract only). https://www.sciencedirect.com/science/article/abs/pii/S0957417421012495
  - **No difference between Fibonacci and non-Fibonacci zones.**

## Analysis methodology and statistics

- **Kornblith et al. 2019.** CKA. ICML. https://arxiv.org/abs/1905.00414 †
- **Hewitt & Liang 2019.** Control tasks for probes. EMNLP. https://arxiv.org/abs/1909.03368 †
- **Apley & Zhu 2020.** ALE plots. JRSS-B. https://arxiv.org/abs/1612.08468 †
- **Fisher, Rudin & Dominici 2019.** Model class reliance and permutation importance. JMLR. https://arxiv.org/abs/1801.01489 †
- **Shi et al. 2025.** Time-MoE. ICLR. https://arxiv.org/abs/2409.16040
- **Liu et al. 2025.** Moirai-MoE. ICML. https://arxiv.org/abs/2410.10469
  - Only qualitative gate analyses.
- **Henderson et al. 2018.** Deep RL that matters. AAAI. https://arxiv.org/abs/1709.06560
- **Bouthillier et al. 2021.** Accounting for variance in ML benchmarks. MLSys. https://arxiv.org/abs/2103.03098
- **Colas, Sigaud & Oudeyer 2018.** How many random seeds? https://arxiv.org/abs/1806.08295
- **Agarwal et al. 2021.** Deep RL at the edge of the statistical precipice (IQM, stratified bootstrap). NeurIPS. https://arxiv.org/abs/2108.13264
- **Politis & Romano 1994.** Stationary bootstrap. JASA. https://doi.org/10.1080/01621459.1994.10476870
- **López de Prado 2018.** *Advances in Financial Machine Learning*, Wiley. Ch. 7 covers purged and embargoed CV; also triple-barrier labelling. †
- **Bergmeir, Hyndman & Koo 2018.** Validity of CV for time series. *CSDA*. https://doi.org/10.1016/j.csda.2017.11.003 †
