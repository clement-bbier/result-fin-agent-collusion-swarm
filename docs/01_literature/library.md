# Literature library

Source: plan Appendix B (B.1 framing facts, B.2 organizers' table) and section 14 (pivotal list). Keys are shared with `library.bib`, `PREREG.md`, `DETECTION_PROTOCOL.md`.
Bibliographic data (title, full author list, year, venue, id/URL) were verified online on 2026-10-09 against the canonical page for each entry (arXiv abstract page, Crossref record of the DOI, NBER, publisher or GitHub page; the `Flags` line says which). Numbers in the `Setting` / `Key number` / `Relevance` lines are still copied from the plan and were NOT re-checked, except where a Flags line says otherwise. An entry keeps the **UNVERIFIED** flag only when a field could not be confirmed from a fetched page; the Flags line then says what was and was not confirmed.

Framing facts (B.1) to keep in mind:

- Fish et al. have no public code; starter = Romero-Sauer-Peist thesis (Mistral); EconEvals is the official prompt source.
- Organizers' imprecisions: Nguyen et al. is 0 % -> 95-100 %; Bonjour covers 3-player games only (threshold 0.05; 100 % detection of an always-colluding pair with 60 games).
- Conflict to settle: Lin (collusion persists at T = 1.0) vs Xu (coupling declines with T) -> temperature sensitivity on Mistral (gap 17).
- Replacement: Eschenbaum, Luo 2026, Collina 2025; no LLM study. No LLM financial simulator measures collusion.

## Algorithmic collusion

### `calvano2020`

- **Title**: Artificial Intelligence, Algorithmic Pricing, and Collusion
- **Authors**: Calvano, Emilio; Calzolari, Giacomo; Denicolò, Vincenzo; Pastorello, Sergio
- **Year / venue**: 2020 / American Economic Review 110(10), 3267-3297
- **Id / URL**: doi:10.1257/aer.20190623
- **Setting**: Bertrand competition, logit demand, Q-learning pricers
- **Key number**: Delta = 85 %; 3 firms 64 %, 4 firms 56 %
- **Code**: openICPSR
- **Relevance to finswarm**: Demand model (a=2, a0=0, mu=0.25, c=1), Nash/monopole benchmarks, Delta definition; positive-control source (R1).
- **Flags**: Verified: via Crossref record of doi:10.1257/aer.20190623 (title, authors, volume, issue, pages)

### `banerjee2023`

- **Title**: Interpreting and Countering Collusion in Deep-Learning Pricing Algorithms
- **Authors**: Banerjee, Soumen
- **Year / venue**: 2023 (v1 5 Mar 2023; v3 17 May 2026) / arXiv preprint, econ.TH
- **Id / URL**: arXiv:2303.02576, https://arxiv.org/abs/2303.02576
- **Setting**: Deep Q-network pricing algorithms; the rival's price is held up or down for a few periods (probe of the same kind as our sting S1/S2)
- **Key number**: none in the abstract. A lit-scout read of the v3 HTML reported: upward deviations "generally accommodated", downward punished (+0.099 against -0.057); not re-opened here
- **Code**: none listed
- **Relevance to finswarm**: Precedent for the forced-deviation probe on non-LLM agents; our 0 % follow of a forced rise and punished undercuts are the LLM version, and are confounded by asks above p_M.
- **Flags**: Verified: against the arXiv abs page (title, sole author, dates, category). UNVERIFIED: the 5-period hold and the accommodate / punish numbers (HTML read by a subagent, not re-checked)

### `calvano2021`

- **Title**: Algorithmic collusion with imperfect monitoring
- **Authors**: Calvano, Emilio; Calzolari, Giacomo; Denicolò, Vincenzo; Pastorello, Sergio
- **Year / venue**: 2021 / International Journal of Industrial Organization 79, 102712
- **Id / URL**: doi:10.1016/j.ijindorg.2021.102712
- **Setting**: Q-learning price games under shocks
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Prior work on shocks and collusion in non-LLM price games (Tier 2 shocks).
- **Flags**: Verified: via Crossref record of doi:10.1016/j.ijindorg.2021.102712 (title, authors, vol. 79, art. 102712). The ScienceDirect PII S0167718721000059 given in the survey could not be fetched (HTTP 403). CAUTION: the paper studies imperfect monitoring (noisy price signals), not demand shocks; the 'Setting' line below was written from the plan and should be re-read before relying on it.

### `ye2025`

- **Title**: Algorithmic Collusion under Observed Demand Shocks
- **Authors**: Ye, Zexin
- **Year / venue**: 2025 / arXiv
- **Id / URL**: arXiv:2502.15084
- **Setting**: Q-learning price games under shocks
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Same as calvano2021: shocks and collusion without LLMs.
- **Flags**: Verified: against arxiv.org/abs/2502.15084 (title, authors, date)

### `sravon2025`

- **Title**: How Market Volatility Shapes Algorithmic Collusion: A Comparative Analysis of Learning-Based Pricing Algorithms
- **Authors**: Sravon, Aheer; Ibrahim, Md.; Mazumder, Devdyuti; Aziz, Ridwan Al
- **Year / venue**: 2025 / arXiv
- **Id / URL**: arXiv:2512.02134
- **Setting**: Learning-based pricing algorithms under market volatility
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Non-LLM precedent on volatility/shocks and collusion (with ye2025).
- **Flags**: Verified: against arxiv.org/abs/2512.02134 (title, authors, date)

### `eschenbaum2022`

- **Title**: Robust Algorithmic Collusion
- **Authors**: Eschenbaum, Nicolas; Mellgren, Filip; Zahn, Philipp
- **Year / venue**: 2022 / arXiv
- **Id / URL**: arXiv:2201.00345
- **Setting**: Q-learning with rematching of agents
- **Key number**: Return to Nash after replacement
- **Code**: none
- **Relevance to finswarm**: Closest prior work on replacement (1.2), Q-learning only; no LLM replacement study exists.
- **Flags**: Verified: against arxiv.org/abs/2201.00345 (title, authors, date)

### `collina2025`

- **Title**: Breaking Algorithmic Collusion in Human-AI Ecosystems
- **Authors**: Collina, Natalie; Arunachaleswaran, Eshwar Ram; Jagadeesan, Meena
- **Year / venue**: 2025 / arXiv
- **Id / URL**: arXiv:2511.21935
- **Setting**: Theory of collusion robustness
- **Key number**: One defector suffices
- **Code**: none
- **Relevance to finswarm**: Theoretical anchor for replacement / entry-of-a-no-regret-agent levers (H1 L6, H7).
- **Flags**: Verified: against arxiv.org/abs/2511.21935 (title, authors, date)

### `tolety2025`

- **Title**: Tacit Bidder-Side Collusion: Artificial Intelligence in Dynamic Auctions
- **Authors**: Tolety, Sriram
- **Year / venue**: 2025 / arXiv
- **Id / URL**: arXiv:2511.21802
- **Setting**: AI bidders in dynamic auctions
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Non-LLM precedent for tacit collusion by AI bidders.
- **Flags**: Verified: against arxiv.org/abs/2511.21802 (title, authors, date)

### `colliard2022`

- **Title**: Algorithmic Pricing and Liquidity in Securities Markets
- **Authors**: Colliard, Jean-Edouard; Foucault, Thierry; Lovo, Stefano
- **Year / venue**: 2022 / SSRN working paper (also CEPR DP17606)
- **Id / URL**: SSRN 4252858; doi:10.2139/ssrn.4252858
- **Setting**: RL market makers
- **Key number**: Supra-competitive spreads
- **Code**: -
- **Relevance to finswarm**: Non-LLM precedent: RL market makers sustain wide spreads.
- **Flags**: Verified: via Crossref record of doi:10.2139/ssrn.4252858 (title, three authors, 2022) and a search result quoting the abstract. SSRN page itself returned HTTP 403. Author order Colliard, Foucault, Lovo follows CEPR/HAL listing (Crossref lists Lovo first). A faculty page reports Review of Financial Studies forthcoming (hhag010); not confirmed from a primary page, so cite the working paper.

### `cont_xiong2024`

- **Title**: Dynamics of market making algorithms in dealer markets: Learning and tacit collusion
- **Authors**: Cont, Rama; Xiong, Wei
- **Year / venue**: 2024 / Mathematical Finance 34(2), 467-521 (online 2023)
- **Id / URL**: doi:10.1111/mafi.12401
- **Setting**: RL dealers in dealer markets
- **Key number**: Supra-competitive spreads
- **Code**: -
- **Relevance to finswarm**: Non-LLM precedent (RL market makers, tacit collusion).
- **Flags**: Verified: via Crossref (journal version, doi:10.1111/mafi.12401; SSRN preprint doi:10.2139/ssrn.4119858, 2022)

### `dou2025`

- **Title**: AI-Powered Trading, Algorithmic Collusion, and Price Efficiency
- **Authors**: Dou, Winston Wei; Goldstein, Itay; Ji, Yan
- **Year / venue**: 2025 / NBER Working Paper 34054 (July 2025)
- **Id / URL**: NBER w34054; doi:10.3386/w34054
- **Setting**: Algorithmic trading, price triggers
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Non-LLM precedent: price-trigger strategies as collusion mechanism.
- **Flags**: Verified: against nber.org/papers/w34054 (title, authors, issue date)

### `harrington2018`

- **Title**: Developing Competition Law for Collusion by Autonomous Artificial Agents
- **Authors**: Harrington, Joseph E.
- **Year / venue**: 2018 / Journal of Competition Law & Economics 14(3), 331-363
- **Id / URL**: doi:10.1093/joclec/nhy016
- **Setting**: Antitrust doctrine for autonomous pricing agents
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Defines the scorecard logic (plus factors, no parallel-conduct-only call) behind D1-D9 (R5).
- **Flags**: Verified: via Crossref (doi:10.1093/joclec/nhy016; SSRN 3037818 version is titled '...Autonomous Price-Setting Agents')

## LLM agents and collusion

### `fish2024`

- **Title**: Algorithmic Collusion by Large Language Models
- **Authors**: Fish, Sara; Gonczarowski, Yannai A.; Shorrer, Ran I.
- **Year / venue**: 2024 / arXiv; ACM EC 2026 (accepted)
- **Id / URL**: arXiv:2404.00806
- **Setting**: Bertrand duopole, logit demand (a=2, a0=0, mu=0.25, c=1; prices x alpha in {1, 3.2, 10}), 300 periods, memory 100, PLANS/INSIGHTS scratchpads, temperature 1, no communication
- **Key number**: UNVERIFIED (from the plan; lit-scout could not confirm): GPT-5.2 price 1.79 (Nash 1.47, monopole 1.92); profit 27.19 (22.29 / 33.75); Delta ~ 0.43; band 0.35-0.95, GPT-5.2 P1 median 0.60. Fish's main runs use GPT-4-0613 with 21 runs per prompt; GPT-5.2 is an appendix check. finswarm uses these figures only for an informational soft check (PREREG Gate 1); the hard criteria decide
- **Code**: none public (EconEvals holds a derived test)
- **Relevance to finswarm**: Positive control (R1, Gate 1); prompts P0+P1 verbatim; 'collusion without messages in posted-price markets' (half of the Fish-Agrawal contradiction).
- **Flags**: Verified: against arxiv.org/abs/2404.00806 (arXiv comment: 'Accepted to EC 2026'). Numbers in Key number are from the plan, not re-checked.

### `fish2025`

- **Title**: EconEvals: Benchmarks and Litmus Tests for Economic Decision-Making by LLM Agents
- **Authors**: Fish, Sara; Shephard, Julia; Li, Minkai; Shorrer, Ran I.; Gonczarowski, Yannai A.
- **Year / venue**: 2025 / arXiv; code on GitHub (MIT)
- **Id / URL**: arXiv:2503.18825; github:sara-fish/econ-evals-paper
- **Setting**: Economic-agent evaluations incl. a duopole test derived from Fish 2024
- **Key number**: -
- **Code**: econ-evals-paper
- **Relevance to finswarm**: Official source of the Fish prompts (no official Fish code).
- **Flags**: Verified: against arxiv.org/abs/2503.18825 and github.com/sara-fish/econ-evals-paper (README)

### `romero2025`

- **Title**: Algorithmic Collusion and the Limits of Coordination: Testing Folk Theorem Predictions in AI-Mediated Markets
- **Authors**: Romero, Julian; Sauer, Lucia; Peist, Moritz
- **Year / venue**: 2025 / Master's thesis, Barcelona School of Economics (2024-2025); code on GitHub (MIT)
- **Id / URL**: github:luciasauer/algorithmic_pricing_llms
- **Setting**: 2-5 firms, Mistral-large
- **Key number**: -3.7 % per firm
- **Code**: MIT
- **Relevance to finswarm**: Starter code; comparability only (no number in the paper comes from third-party code).
- **Flags**: Verified: against github.com/luciasauer/algorithmic_pricing_llms (README: thesis title, authors, institution, supervisor C. Rauh, MIT licence). README clone command uses a different repo name.

### `riemer2026`

- **Title**: Position: Collusion Risks Among AI Reasoning Agents Justify Certification Requirements for Making Market Decisions
- **Authors**: Riemer, Matthew; Tosato, Tommaso; Memarian, Amin; Touzel, Maximilian Puelma; Berseth, Glen; Rish, Irina; Dumas, Guillaume
- **Year / venue**: 2026 / arXiv; ICML 2026 (position paper)
- **Id / URL**: arXiv:2608.18078
- **Setting**: Bertrand 2-3 firms, steering
- **Key number**: Judge scale 1-10
- **Code**: mattriemer/LLMCartel
- **Relevance to finswarm**: Starter repo (LLMCartel); judge-scale comparison.
- **Flags**: Verified: against arxiv.org/abs/2608.18078 (arXiv comment: ICML 2026) and github.com/mattriemer/LLMCartel (README cites this paper). Anomaly: arXiv page date is 2026-05-29 although id is 2608.*.

### `lin2024`

- **Title**: Strategic Collusion of LLM Agents: Market Division in Multi-Commodity Competitions
- **Authors**: Lin, Ryan Y.; Ojha, Siddhartha; Cai, Kevin; Chen, Maxwell F.
- **Year / venue**: 2024 / arXiv
- **Id / URL**: arXiv:2410.00031
- **Setting**: Cournot 2x2
- **Key number**: HHI -> 1
- **Code**: collusive-llm-agents
- **Relevance to finswarm**: Reported collusion persists at T = 1.0; conflicts with Xu on temperature (gap 17, sensitivity on Mistral).
- **Flags**: Verified: against arxiv.org/abs/2410.00031 (title, authors, date)

### `agrawal2025`

- **Title**: Evaluating LLM Agent Collusion in Double Auctions
- **Authors**: Agrawal, Kushal; Teo, Verona; Vazquez, Juan J.; Kunnavakkam, Sudarsh; Srikanth, Vishak; Liu, Andy
- **Year / venue**: 2025 / arXiv
- **Id / URL**: arXiv:2507.01413
- **Setting**: Continuous double auction, 5 sellers (cost 80) + 5 buyers (value 100), LLM agents, 30 rounds, midpoint trade, 10 sessions/condition
- **Key number**: No messages: 87.7 (CE 90) after 10 rounds; messages 90.5; monitoring 86.2; urgency 96.3; GPT-4.1 > Claude 3.7; judge 1-4 on reasoning traces
- **Code**: llm-agent-collusion
- **Relevance to finswarm**: Other half of the Fish-Agrawal contradiction; judge rubric (D9); CDA variant (Delta_p = (p - 90)/10).
- **Flags**: Verified: against arxiv.org/abs/2507.01413 (title, authors, date)

### `mazur2025`

- **Title**: Emergent Price-Fixing by LLM Auction Agents (emergent_collusion)
- **Authors**: Mazur, Lech
- **Year / venue**: 2025 / GitHub repository (year from commit dates: July 2025)
- **Id / URL**: github:lechmazur/emergent_collusion
- **Setting**: 6 agents (3 buyers, 3 sellers), 8 rounds, optional chat
- **Key number**: Illegality score >= 7 in 19-75 % of games (README)
- **Code**: lechmazur/emergent_collusion
- **Relevance to finswarm**: Uninstructed collusion among frontier LLMs in an auction with chat; comparison point.
- **Flags**: Verified: against github.com/lechmazur/emergent_collusion (title, owner, README; commits dated 15 July 2025; licence not stated). Author's full name 'Lech Mazur' is inferred from the GitHub handle and not stated on the page.

### `nakamura2026`

- **Title**: Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems
- **Authors**: Nakamura, Mason; Kumar, Abhinav; Das, Saswat; Abdelnabi, Sahar; Mahmud, Saaduddin; Fioretto, Ferdinando; Zilberstein, Shlomo; Bagdasarian, Eugene
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2602.15198
- **Setting**: DCOP, 19 models
- **Key number**: 57.9 % coalition advantage; 10.5 % detected
- **Code**: umass-ai-safety/colosseum
- **Relevance to finswarm**: Organizers' barred 3.1 / coalition-advantage metric; colluding channels.
- **Flags**: Verified: against arxiv.org/abs/2602.15198 (title, authors, date); the plan's 'Colosseum v2' label is not on the page

### `deshpande2026`

- **Title**: Strategic AI in Cournot Markets
- **Authors**: Deshpande, Sanyukta; Jacobson, Sheldon H.
- **Year / venue**: 2026 (v1 24 Jan 2026) / arXiv preprint, cs.GT
- **Id / URL**: arXiv:2601.17263, https://arxiv.org/abs/2601.17263
- **Setting**: LLM agents in repeated Cournot markets; the abstract names decision type, opponent strategies and market composition as analysis dimensions
- **Key number**: LLMs drive prices "up to 200% above Nash equilibrium levels" through sustained tacit collusion (abstract)
- **Code**: none listed
- **Relevance to finswarm**: Comparator for entry and composition effects in LLM markets. Two readings disagree on whether the abstract says forcing a few agents to best-respond restores competitive prices; cite only the 200 % figure until the full text is read.
- **Flags**: Verified: against the arXiv abs page (title, authors, date, 200 % figure). UNVERIFIED: any claim that best-responding agents or entry restore competitive prices

### `keppo2026`

- **Title**: On the Fragility of AI Agent Collusion
- **Authors**: Keppo, Jussi; Li, Yuze; Tsoukalas, Gerry; Yuan, Nuo
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2603.20281
- **Setting**: Bertrand logit, R1-Qwen
- **Key number**: Patience +22 %; LLM vs Q-learner +2/+8 %; 3/4/5 firms +36/+11/+3 %; 32B vs 14B +21 % (leader-follower); GPT-5-mini +57 %; anti-collusion prompt 28-30 % below Nash
- **Code**: none
- **Relevance to finswarm**: Model-capability effects; motivates the Sonnet 5.5 capability arm. Abstract claims (read 2026-10-10): more competing LLMs and LLM-vs-Q-learner heterogeneity break up collusion (qualitative); patience heterogeneity cuts the price lift from 22 % to 10 %, asymmetric data access to 7 %; model-size differences do not break collusion. Closest LLM evidence that an information lever and composition matter.
- **Flags**: Verified: against arxiv.org/abs/2603.20281 (title, authors, date, abstract claims above). UNVERIFIED: any quantitative firm-count or Q-learner-insertion result beyond the abstract

### `aharon2026`

- **Title**: Tacit Coordination of Large Language Models
- **Authors**: Aharon, Ido; La Malfa, Emanuele; Wooldridge, Michael; Kraus, Sarit
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2601.22184
- **Setting**: Focal points
- **Key number**: Same-family models coordinate better
- **Code**: EmanueleLM/focal-points
- **Relevance to finswarm**: Schelling-point vector; same-family coordination (cross-family robustness).
- **Flags**: Verified: against arxiv.org/abs/2601.22184 (title, authors, date)

### `rippin2026`

- **Title**: Tool Use Enables Undetectable Steganography in Multi-Agent LLM Systems
- **Authors**: Rippin, Jimmy Laurence; Marshall, Simon C.; Africa, David Demitri; de Witt, Christian Schroeder
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2606.28425
- **Setting**: Provable steganography
- **Key number**: CCI = 0
- **Code**: none
- **Relevance to finswarm**: Schelling/steganography vector; what text-free coordination is NOT shown here.
- **Flags**: Verified: against arxiv.org/abs/2606.28425 (title, authors, date)

### `bracale2026`

- **Title**: Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets via Public Governance Graphs
- **Authors**: Syrnikov, Marcantonio Bracale; Pierucci, Federico; Galisai, Marcello; Prandi, Matteo; Bisconti, Piercosma; Giarrusso, Francesco; Sorokoletova, Olga; Suriani, Vincenzo; Nardi, Daniele
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2601.11369
- **Setting**: Cournot, 3 regimes
- **Key number**: Institutional 3.10 -> 1.82; prompt alone: nothing
- **Code**: none
- **Relevance to finswarm**: Institutional controls beat prompts in LLM Cournot; supports 'institution over model' theory of change.
- **Flags**: Verified: against arxiv.org/abs/2601.11369 (title, authors, date); first author's surname is 'Syrnikov' on arXiv (given name Marcantonio Bracale), key kept

### `dhanda2026`

- **Title**: A Price Cap Can Act as a Focal Point for LLM Pricing Agents: Evidence from a Repeated Bertrand Duopoly
- **Authors**: Dhanda, Kunal
- **Year / venue**: 2026 / Research Square preprint (In Review), posted 2026-06-30
- **Id / URL**: doi:10.21203/rs.3.rs-10183486/v1
- **Setting**: Duopole, Llama-3.2-3B
- **Key number**: Announced cap -> Delta 0.665 (focal point); aggressive -> -0.49; neutral 0.11 n.s.
- **Code**: none
- **Relevance to finswarm**: Announced numbers as focal point; 'compete aggressively' control.
- **Flags**: Verified: via researchsquare.com/article/rs-10183486/v1 (title, shown with '(Revised)') and Crossref record of the DOI (sole author Kunal Dhanda, independent researcher, posted 30 June 2026)

### `liu2026`

- **Title**: Breaking the Secret: Economic Interventions for Combating Collusion in Embodied Multi-Agent Systems
- **Authors**: Liu, Qi; Chen, Xiaohui; Zhao, Zhihui; Zheng, Yaowen; Yu, Dan; Zhang, Zehua; Sun, Limin; Chen, Yongle
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2604.23511
- **Setting**: Embodied multi-agent
- **Key number**: Deposit >= 2000 -> ~0 %
- **Code**: none
- **Relevance to finswarm**: Cost of defection / deterrence evidence.
- **Flags**: Verified: against arxiv.org/abs/2604.23511 (title, authors, date)

### `tomasev2026`

- **Title**: Distributional AGI Safety
- **Authors**: Tomašev, Nenad; Franklin, Matija; Jacobs, Julian; Krier, Sébastien; Osindero, Simon
- **Year / venue**: 2025 / arXiv
- **Id / URL**: arXiv:2512.16856
- **Setting**: Conceptual
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Conceptual framing of multi-agent risk.
- **Flags**: Verified: against arxiv.org/abs/2512.16856 (title, authors, date); year corrected from 2026 to 2025 (submitted 2025-12-18), key kept

### `shi2026`

- **Title**: Emergent Collusion in Long-Horizon LLM Agent Interaction
- **Authors**: Shi, Xinrui; Zhang, Yanzhe; Yang, Diyi
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2609.24967
- **Setting**: Mutual verification
- **Key number**: 93.6 %
- **Code**: SALT-NLP
- **Relevance to finswarm**: Pivotal list; related-work contrast. Key per task statement is shi2026; see shi2026dilemmas for 2607.05132.
- **Flags**: Verified: against arxiv.org/abs/2609.24967 (title, authors, date); the plan's 'Mutual verification, 93.6 %' description was NOT confirmed against the abstract (title is about emergent collusion), re-read before citing; distinct from shi2026dilemmas

### `shi2026dilemmas`

- **Title**: When Agents Lie: Premeditation, Persistence, and Exploitation in Repeated Games
- **Authors**: Shi, Jerick; Zhang, Terry Jingcheng; Schölkopf, Bernhard; Conitzer, Vincent; Jin, Zhijing
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2607.05132
- **Setting**: 6 social dilemmas
- **Key number**: > 90 % premeditated
- **Code**: LLM-Trust-Breaking
- **Relevance to finswarm**: Evidence of premeditated defection in dilemmas.
- **Flags**: Verified: against arxiv.org/abs/2607.05132 (title, authors, date); arXiv comment: Best Paper Award at ICML NExT-Game Workshop

### `tian2026`

- **Title**: Prompt Optimization Enables Stable Algorithmic Collusion in LLM Agents
- **Authors**: Tian, Yingtao
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2604.17774
- **Setting**: Nested-logit pricing
- **Key number**: Prices move toward monopole
- **Code**: none
- **Relevance to finswarm**: Price as vector; demand-system generality.
- **Flags**: Verified: against arxiv.org/abs/2604.17774 (title, authors, date); single author (plan said 'Tian et al.'); 'nested-logit' setting not confirmed from the page

### `lee2026`

- **Title**: Who Leads and Who Collects: Algorithmic Collusion in Markets of Heterogeneous Language Models
- **Authors**: Lee, Jun Yeong
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2610.11256
- **Setting**: Bertrand, 4 firms, model composition
- **Key number**: Claude 0.79, Gemini 0.72, DeepSeek 0.24, GPT -0.15; mixed 0.02; mixing not significant (p = 0.38)
- **Code**: none
- **Relevance to finswarm**: Closest prior work on monoculture vs heterogeneity (taken, 8 Oct 2026): robustness arm only.
- **Flags**: Verified: against arxiv.org/abs/2610.11256 (title, authors, date); single author (plan said 'Lee et al.'); submitted 2026-10-08

### `garra2026`

- **Title**: Mitigating Emergent Collusion in LLM Pricing Agents
- **Authors**: Garra, Abdullah
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2609.13037
- **Setting**: Fish-style duopoly baseline (two LLM pricers, DeepSeek-V3.1 via DeepInfra, 300 periods, 7 runs per prompt, outcomes on periods 251-300) with three intervention arms evaluated in independent runs: a prompt warning, a Harrington-style expected-damages regulator, and a random-price entrant (three firms once the entrant is in)
- **Key number**: entrant arm: P1 Delta -0.367 (average price 1.325), P2 Delta -0.571 (1.290), measured against a random-entrant Nash benchmark of about 1.3888, 0 of 7 runs above Nash in each prompt; baseline P1 Delta 0.569, warning 0.242, regulator 0.080 (plan figures). The -0.367 is a level against a different benchmark, not a pre/post change
- **Code**: none
- **Relevance to finswarm**: Closest prior work on interventions (H1); differs: interventions evaluated in independent runs with no checkpoint pairing, single provider, no efficiency-cost measure. Write 'present in every period' for the entrant (not mid-run) and 'three firms' for the entrant arm; the duopoly is the baseline only.
- **Flags**: Verified: against arxiv.org/abs/2609.13037 (title, authors, date); single author (plan said 'Garra et al.'); setting re-read from arxiv.org/html/2609.13037 on 2026-10-10 (lit-scout); entrant read 2026-10-10 (about 100k of 110k characters): non-LLM i.i.d. price Unif[1.4277, 1.5181] around the duopoly Nash price of about 1.4729, present in all 300 periods (not added mid-run), three firms with the entrant

### `luo2026`

- **Title**: Algorithmic Collusion at Test Time: A Meta-game Design and Evaluation
- **Authors**: Luo, Yuhong; Schoepflin, Daniel; Wang, Xintong
- **Year / venue**: 2026 / arXiv; AAMAS 2026
- **Id / URL**: arXiv:2602.17203
- **Setting**: Meta-game
- **Key number**: Symmetric costs
- **Code**: CollusionMetagame
- **Relevance to finswarm**: Replacement-adjacent (1.2); meta-game.
- **Flags**: Verified: against arxiv.org/abs/2602.17203 (title, authors, date); arXiv comment: AAMAS 2026

### `struski2026`

- **Title**: Competitive Market Behavior of LLMs
- **Authors**: Struski, Pawel; Swistak, Jakub; Okulska, Inez; Biecek, Przemyslaw
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2609.02580
- **Setting**: LLM double auctions
- **Key number**: Slow convergence; attribution to collusion hard
- **Code**: none
- **Relevance to finswarm**: Motivates scripted competitive benchmark + ghost markets.
- **Flags**: Verified: against arxiv.org/abs/2609.02580 (title, authors, date)

### `zeng2026`

- **Title**: Voluntary Collusion with Secret Tools in Competing LLM Agents
- **Authors**: Zeng, Xijie; Rudzicz, Frank
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2605.27593
- **Setting**: Liar's Bar, Cleanup
- **Key number**: 10/12
- **Code**: LLM-Agent-collusion
- **Relevance to finswarm**: Accepted channels.
- **Flags**: Verified: against arxiv.org/abs/2605.27593 (title, authors, date)

### `ross_so_lo2026`

- **Title**: Why Better Models Can Create Riskier Systems: Evidence from LLM Agents in Financial Markets
- **Authors**: Ross, Jillian; So, Eric; De Simone, Zoe; Pozniak, Charles; Lo, Andrew W.
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2609.04373
- **Setting**: Capability and strategy
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Capability arm motivation (Sonnet 5.5).
- **Flags**: Verified: against arxiv.org/abs/2609.04373 (title, authors, date)

## Detection and evidence

### `bonjour2022`

- **Title**: Information theoretic approach to detect collusion in multi-agent games
- **Authors**: Bonjour, Trevor; Aggarwal, Vaneet; Bhargava, Bharat
- **Year / venue**: 2022 / UAI 2022, PMLR 180, pp. 223-232
- **Id / URL**: proceedings.mlr.press/v180/bonjour22a.html
- **Setting**: RPS and 3-player Leduc; net influence
- **Key number**: Seuil 0.05; 100 % detection of an always-colluding pair with 60 games (3-player games only)
- **Code**: none
- **Relevance to finswarm**: D6 net influence: gamma(i;j) = I(A_i; A_j | S); RPS validation control; never applied to LLMs before.
- **Flags**: Verified: against proceedings.mlr.press/v180 (title, three authors, pages; the page text was read only up to 100000 of 118200 characters)

### `xu2026`

- **Title**: Collusion with Competitive Marginals: Price-Level Audits Are Blind by Construction
- **Authors**: Xu, Xin; Wu, Chengrui; Lu, Jiayu; Tan, Kaizhen; Tao, Siru; Hong, Hanzhe
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2607.26385
- **Setting**: Auctions, 60 agents; MEV-Boost
- **Key number**: +0.053 vs +0.0001; honest floor +0.50 to +0.81; coupling declines with T
- **Code**: at publication
- **Relevance to finswarm**: Single-agent test is blind -> joint, ghost-netted tests (D5/D6); temperature conflict with Lin (gap 17).
- **Flags**: Verified: against arxiv.org/abs/2607.26385 (title, authors, date)

### `antovazquez2026`

- **Title**: Oversight is Not Compliance: Tacit Collusion in LLM Pricing Agents Under Antitrust Regulation
- **Authors**: Anto, Meiri; Vazquez, Juan (affiliations not listed on the ICML page: UNVERIFIED)
- **Year / venue**: 2026 / ICML 2026 workshop paper (poster, "The Second Workshop on Agents in the Wild: Safety, Security, and Beyond"), not the main conference
- **Id / URL**: icml.cc/virtual/2026/67865 ; OpenReview aPnAkoXjF2 (page not readable on 2026-10-10, bot check); no arXiv version found
- **Setting**: Repeated pricing duopoly; Claude Sonnet 4, DeepSeek-Chat V3, GPT-5.4 mini; three oversight modes (passive logging, one-shot revision, hard veto); agent output split into private notes, public justification and a structured self-disclosure flag. "Cosmetic compliance" = the public channel hides a strategy present in the private notes. Veto moves Sonnet 4 to the competitive benchmark, revision moves DeepSeek V3 past it, GPT-5.4 mini stays supracompetitive under every mode. Market institution fixed; no mid-run intervention stated (paper body not read: UNVERIFIED)
- **Key number**: beta_veto = -0.89, beta_revision = -0.46 on the supracompetitive index (abstract)
- **Code**: none found
- **Relevance to finswarm**: Reasoning-trace and justification monitoring of colluding LLM pricing agents is already reported to fail here and in `leepark2026`, so finswarm's scratchpad judge is a baseline, not a headline; finswarm's claim is the ghost-netted action detectors on process-labelled negatives. Cite as a workshop paper and only for what the abstract states.
- **Flags**: Verified: against the ICML virtual page on 2026-10-10 (title, authors, venue type, abstract summary, beta values; no code or arXiv link listed). UNVERIFIED: affiliations, full abstract wording, OpenReview content, paper body (methods, sample sizes, prompts), overlap with action-level detectors D1-D9, presence of a communication channel.

### `leepark2026`

- **Title**: Faithful yet Collusive: Why Chain-of-Thought Monitoring Cannot Detect Collusion in LLM Pricing Agents under Oligopolistic Competition
- **Authors**: Lee, Dohun; Park, Hyunwoo
- **Year / venue**: 2026 / arXiv; Findings of EMNLP 2026
- **Id / URL**: arXiv:2609.18346
- **Setting**: Duopole/triopole
- **Key number**: CoT monitoring does not detect collusion
- **Code**: none
- **Relevance to finswarm**: Reasoning-trace monitoring fails -> actions beat judge (H9).
- **Flags**: Verified: against arxiv.org/abs/2609.18346 (title, authors, date); arXiv comment: Findings of EMNLP 2026. Not to be confused with arXiv 2609.18357 (see leepark2026signal)

### `leepark2026signal`

- **Title**: Market Signal Injection: Adversarial Context Manipulation of LLM Pricing Agents
- **Authors**: Lee, Dohun; Park, Hyunwoo
- **Year / venue**: 2026 / arXiv; FinNLP 2026 Workshop @ EMNLP 2026
- **Id / URL**: arXiv:2609.18357
- **Setting**: Adversarial context manipulation of LLM pricing agents
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Related work on manipulation of LLM pricing agents (same authors as leepark2026).
- **Flags**: Verified: against arxiv.org/abs/2609.18357 (title, authors, date); arXiv comment: accepted to FinNLP 2026. NEW entry (companion of leepark2026)

### `papadopoulos2026`

- **Title**: Non-Stationarity Breaks Permutation Surrogates in Multi-Agent Reinforcement Learning: Diagnosis and Remedies
- **Authors**: Papadopoulos, Nikolaos Al.; Psannis, Konstantinos E.
- **Year / venue**: 2026 (v1 26 Apr 2026; v4 8 Sep 2026) / arXiv preprint, cs.AI, 28 pp.
- **Id / URL**: arXiv:2604.23716, https://arxiv.org/abs/2604.23716
- **Setting**: MARL influence and coordination detection with permutation tests in a social dilemma and a coordination game
- **Key number**: false-positive rate 100.00 % (social dilemma) and 99.95 % (coordination race) when the non-stationary training transient is included; 3.0 % and 11.8 % when excluded; block permutation within training time 5.25 % and 5.50 %
- **Code**: https://github.com/dentros/te-nonstationarity ; Zenodo doi:10.5281/zenodo.22658530
- **Relevance to finswarm**: Threat to a plain cyclic-shift null on non-stationary dealer series; our ghost null (fresh draws, no feedback) must be compared with a cyclic-shift null on the ghost windows (A-019 check).
- **Flags**: Verified: against the arXiv abs page (title, authors, dates, numbers, code links). UNVERIFIED: that the statistic is transfer entropy (inferred from the repository name) and the 'independent runs' null construction

### `riedl2025`

- **Title**: Emergent Coordination in Multi-Agent Language Models
- **Authors**: Riedl, Christoph
- **Year / venue**: 2025 (v1 5 Oct 2025; v4 28 Apr 2026) / ICLR 2026
- **Id / URL**: arXiv:2510.05174, https://arxiv.org/abs/2510.05174
- **Setting**: LLM agents in a guessing game without direct communication, three randomised interventions
- **Key number**: none in the abstract
- **Code**: none listed
- **Relevance to finswarm**: Applies null-netting (observed minus a null median) to LLM-agent logs; measures emergent coordination, not collusion. Do not cite as a collusion detector.
- **Flags**: Verified: against the arXiv abs page (title, sole author, dates, venue). UNVERIFIED: the information measures used (time-delayed mutual information, PID) and the surrogate construction, both from a subagent's reading of the full text

### `rose2026`

- **Title**: Detecting Multi-Agent Collusion Through Multi-Agent Interpretability
- **Authors**: Rose, Aaron; Cullen, Carissa; Abdelnabi, Sahar; Torr, Philip; Kaplowitz, Brandon Gary; de Witt, Christian Schroeder
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2604.01151
- **Setting**: 4 agents, 2 colluders
- **Key number**: AUROC 1.00 / 0.73; judges 1-3/20
- **Code**: narcbench
- **Relevance to finswarm**: Detection baselines.
- **Flags**: Verified: against arxiv.org/abs/2604.01151 (title, authors, date); the plan's 'NARCBench' label is not in the title (benchmark name presumably inside the paper, not checked)

### `kaur2026`

- **Title**: Beyond the Transcript: Detecting Covert Coordination in Latent Multi-Agent Communication
- **Authors**: Kaur, Ramneet; Chari, Pradyumna; Raskar, Ramesh; Singh, Jugad; Jha, Sumit Kumar; Roy, Anirban
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2608.19161
- **Setting**: Hidden states
- **Key number**: 0.99 -> 0.85 mixed
- **Code**: -
- **Relevance to finswarm**: Probe-based detection baseline.
- **Flags**: Verified: against arxiv.org/abs/2608.19161 (title, authors, date); arXiv title renders 'Co ordination' with a stray space, normalised here

### `tailor2025`

- **Title**: Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs
- **Authors**: Tailor, Om
- **Year / venue**: 2025 / arXiv
- **Id / URL**: arXiv:2510.04303
- **Setting**: ColludeBench
- **Key number**: Union at FPR 1e-3
- **Code**: Ostailor/Steganographic_Collusion (MIT)
- **Relevance to finswarm**: Mutual-information audit; union-of-detectors at tiny FPR.
- **Flags**: Verified: against arxiv.org/abs/2510.04303 (title, authors, date); single author; code github.com/Ostailor/Steganographic_Collusion (MIT, README: NeurIPS submission). The plan's 'ColludeBench' label was not seen on the pages fetched

### `hartline2025`

- **Title**: Regulation of Algorithmic Collusion, Refined: Testing Pessimistic Calibrated Regret
- **Authors**: Hartline, Jason D.; Wang, Chang; Zhang, Chenhao
- **Year / venue**: 2025 / arXiv; ACM CS&Law 2025 (to appear per arXiv comment)
- **Id / URL**: arXiv:2501.09740
- **Setting**: Calibrated regret
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Regret-based detection.
- **Flags**: Verified: against arxiv.org/abs/2501.09740 (title, authors, date); arXiv comment: 'To appear in ACM CS&Law 25'

### `motwani2024`

- **Title**: Secret Collusion among AI Agents: Multi-Agent Deception via Steganography
- **Authors**: Motwani, Sumeet Ramesh; Baranchuk, Mikhail; Strohmeier, Martin; Bolina, Vijay; Torr, Philip H. S.; Hammond, Lewis; de Witt, Christian Schroeder
- **Year / venue**: 2024 / arXiv
- **Id / URL**: arXiv:2402.07510
- **Setting**: CASE benchmark
- **Key number**: GPT-4 jump
- **Code**: not found
- **Relevance to finswarm**: Steganography/Schelling vector; overlap risk flagged by judges of past entries.
- **Flags**: Verified: against arxiv.org/abs/2402.07510 (title, authors, date)

### `dietrich2026`

- **Title**: Can Multi-Agent LLMs Identify Their Peers? Stylometric Fingerprinting in Role-Constrained Political Analysis
- **Authors**: Dietrich, Juergen
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2606.09854
- **Setting**: Stylometry
- **Key number**: F1 0.991
- **Code**: on request
- **Relevance to finswarm**: Agent identity detection (3.3, out of scope).
- **Flags**: Verified: against arxiv.org/abs/2606.09854 (title, authors, date); single author (plan said 'Dietrich et al.')

### `radev2026`

- **Title**: The Best-Laid SCHEMEs: Coordinated Sabotage and Monitoring in Multi-Agent Systems
- **Authors**: Radev, Nikolay; Haas, Lennart; Arnav, Benjamin; Bernabeu-Pérez, Pablo
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2605.29178
- **Setting**: Software sabotage; monitoring notice
- **Key number**: Notice 84 -> 41 %
- **Code**: none
- **Relevance to finswarm**: Monitoring-notice effect (H10, 3.4).
- **Flags**: Verified: against arxiv.org/abs/2605.29178 (title, authors, date)

### `nguyen2026`

- **Title**: TraceGuard: Structured Multi-Dimensional Monitoring as a Collusion-Resistant Control Protocol
- **Authors**: Nguyen, Khanh Linh; Nghiem, Hoa; Tran, Tu
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2604.03968
- **Setting**: BashArena
- **Key number**: 0 % -> 95-100 % (organizers' figure imprecise)
- **Code**: TraceGuard
- **Relevance to finswarm**: Monitoring baseline; organizers' imprecision noted.
- **Flags**: Verified: against arxiv.org/abs/2604.03968 (title, authors, date); the plan's '0 % -> 95-100 %' figure was not checked

### `arike2026`

- **Title**: How does information access affect LLM monitors' ability to detect sabotage?
- **Authors**: Arike, Rauno; Moreno, Raja Mehta; Subramani, Rohan; Biswas, Shubhorup; Ward, Francis Rhys
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2601.21112
- **Setting**: Sabotage monitoring
- **Key number**: Less information = better (7/8)
- **Code**: 2 repos
- **Relevance to finswarm**: Monitor-information fraction (gap 7).
- **Flags**: Verified: against arxiv.org/abs/2601.21112 (title, authors, date)

### `kale2025`

- **Title**: Reliable Weak-to-Strong Monitoring of LLM Agents
- **Authors**: Kale, Neil; Zhang, Chen Bo Calvin; Zhu, Kevin; Aich, Ankit; Rodriguez, Paula; Team, Scale Red; Knight, Christina Q.; Wang, Zifan
- **Year / venue**: 2025 / arXiv
- **Id / URL**: arXiv:2508.19461
- **Setting**: SHADE-Arena
- **Key number**: AUC > 0.85 hybrid
- **Code**: ScaleAI/mrt
- **Relevance to finswarm**: Monitor baselines.
- **Flags**: Verified: against arxiv.org/abs/2508.19461 (title, authors, date); the plan's 'SHADE' label is not in the title (presumably the SHADE-Arena setting, not checked)

### `najt2026`

- **Title**: SLEIGHT-Bench: A Benchmark of Evasion Attacks Against Agent Monitors
- **Authors**: Najt, Elle; Toft, Colin; Tracy, Tyler; Roger, Fabien; Benton, Joe
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2605.16626
- **Setting**: 40 attacks
- **Key number**: 20/40
- **Code**: sleight-bench
- **Relevance to finswarm**: Monitoring evasion.
- **Flags**: Verified: against arxiv.org/abs/2605.16626 (title, authors, date)

### `jiralerspong2026`

- **Title**: Noticing the Watcher: LLM Agents Can Infer CoT Monitoring from Blocking Feedback
- **Authors**: Jiralerspong, Thomas; Kondrup, Flemming; Bengio, Yoshua
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2603.16928
- **Setting**: 2,400 episodes
- **Key number**: 3 intents
- **Code**: none
- **Relevance to finswarm**: Intent evidence.
- **Flags**: Verified: against arxiv.org/abs/2603.16928 (title, authors, date)

### `kovacic2011`

- **Title**: Plus Factors and Agreement in Antitrust Law
- **Authors**: Kovacic, William E.; Marshall, Robert C.; Marx, Leslie M.; White, Halbert L.
- **Year / venue**: 2011 / Michigan Law Review 110(3), from p. 393
- **Id / URL**: doi:10.36644/mlr.110.3.plus
- **Setting**: Antitrust doctrine
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Plus-factor taxonomy: scorecard structure (R5, 2.3).
- **Flags**: Verified: via Crossref (doi:10.36644/mlr.110.3.plus: title, four authors, journal, start page 393). Volume/issue 110(3) is from the survey (Crossref hit did not show them); end page not retrieved.

### `fisher2025`

- **Title**: Tacit Agreements to Collude: Enforcing Section 1 of the Sherman Act in the Age of Algorithms
- **Authors**: Fisher, David O.
- **Year / venue**: 2025 / ABA Antitrust Magazine, Fall 2025 (AAI/ABA)
- **Id / URL**: (no DOI; see antitrustinstitute.org work-product page)
- **Setting**: Doctrine
- **Key number**: Plus factors
- **Code**: -
- **Relevance to finswarm**: Doctrine for LLM collusion (with Berger).
- **Flags**: UNVERIFIED (partial): title, author and venue come from a web-search summary of the American Antitrust Institute page; the page and the ABA article were not fetched; volume/issue '40(1)' from the plan not confirmed

### `berger2025`

- **Title**: Coordination by Code: Rethinking Antitrust Liability in the Age of AI Pricing
- **Authors**: Berger, Lee; Weiner, Michael; Jiang, Weisiyu
- **Year / venue**: 2025 / ABA Antitrust Magazine, Fall 2025
- **Id / URL**: (no DOI; see steptoe.com publication page)
- **Setting**: Doctrine
- **Key number**: Plus factors
- **Code**: -
- **Relevance to finswarm**: Doctrine.
- **Flags**: UNVERIFIED (partial): title and authors come from a web-search summary of a Steptoe publication page (the page itself returned HTTP 403); author spellings may differ; volume/issue '40(1)' from the plan not confirmed

### `cma2026`

- **Title**: AI and collusion: frontiers, opportunities and challenges
- **Authors**: Croxson, Karen; Enser, Juliette
- **Year / venue**: 2026 / UK Competition and Markets Authority blog, 4 March 2026
- **Id / URL**: https://competitionandmarkets.blog.gov.uk/2026/03/04/ai-and-collusion-frontiers-opportunities-and-challenges/
- **Setting**: Doctrine
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Regulator perspective.
- **Flags**: Verified: against competitionandmarkets.blog.gov.uk (title, authors Karen Croxson and Juliette Enser, date). It is a blog post, not a CMA report; the entry's author field was the organisation in the plan.

### `idowu2026`

- **Title**: Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems
- **Authors**: Idowu, Jamiu; Almasoud, Ahmed S.; Alfahid, Ayman
- **Year / venue**: 2026 / Knowledge-Based Systems 344, 116067 (arXiv:2601.00360)
- **Id / URL**: doi:10.1016/j.knosys.2026.116067; arXiv:2601.00360
- **Setting**: Doctrine
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Doctrine.
- **Flags**: Verified: against arxiv.org/abs/2601.00360 (journal reference) and the Crossref record of doi:10.1016/j.knosys.2026.116067

### `abada2026`

- **Title**: Algorithmic Collusion: Where Are We and Where Should We Be Going?
- **Authors**: Abada, Ibrahim; Harrington Jr., Joseph E.; Lambin, Xavier; Meylahn, Janusz M.
- **Year / venue**: 2024 / SSRN working paper
- **Id / URL**: SSRN 4891033; doi:10.2139/ssrn.4891033
- **Setting**: Criteria for judging simulations
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: To read: criteria for evaluating simulations.
- **Flags**: Verified: via Crossref record of doi:10.2139/ssrn.4891033 (title, four authors; record created Aug 2024). Year corrected from 2026 to 2024 (the key still says 2026). SSRN page returned HTTP 403.

### `delgiudice2026`

- **Title**: Prompted Identity Degrades Cooperation in Multi-Agent LLM Systems
- **Authors**: Del Giudice, Xavier; Palma, Alessio; Migliarini, Matteo; Galasso, Fabio; Spinelli, Indro
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2609.35928
- **Setting**: Factions
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Identity/faction detection (out of scope).
- **Flags**: Verified: against arxiv.org/abs/2609.35928 (title, authors, date); WARNING: the paper is about prompted identity and cooperation, which does not match the 'detecting a replaced peer' use in gap_table row 3

### `ghanem2026`

- **Title**: Steganalysis of Adaptive Covert Collusion in Tool-Using Agent Populations: A Black-Box, Cross-Principal Approach
- **Authors**: Ghanem, Mohamed Chahine
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2608.02698
- **Setting**: Tool/timing channels
- **Key number**: placeholders
- **Code**: -
- **Relevance to finswarm**: Gap 14, out of scope.
- **Flags**: Verified: against arxiv.org/abs/2608.02698 (title, authors, date); single author

### `cloud2025`

- **Title**: Subliminal Learning: Language models transmit behavioral traits via hidden signals in data
- **Authors**: Cloud, Alex; Le, Minh; Chua, James; Betley, Jan; Sztyber-Betley, Anna; Hilton, Jacob; Marks, Samuel; Evans, Owain
- **Year / venue**: 2025 / arXiv
- **Id / URL**: arXiv:2507.14805
- **Setting**: Common base model
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Common-base vector.
- **Flags**: Verified: against arxiv.org/abs/2507.14805 (title, authors, date)

## Market microstructure and mechanism design

### `christie_schultz1994`

- **Title**: Why Do NASDAQ Market Makers Avoid Odd-Eighth Quotes?
- **Authors**: Christie, William G.; Schultz, Paul H.
- **Year / venue**: 1994 / Journal of Finance 49(5), 1813-1840
- **Id / URL**: doi:10.1111/j.1540-6261.1994.tb04782.x
- **Setting**: NASDAQ dealer quotes
- **Key number**: Dealers avoided odd-eighth quotes
- **Code**: -
- **Relevance to finswarm**: Historical dealer collusion; motivates tick lever L3 and order-handling lever.
- **Flags**: Verified: via Crossref record of the DOI (title, authors, volume, issue, pages)

### `barclay1999`

- **Title**: Effects of Market Reform on the Trading Costs and Depths of Nasdaq Stocks
- **Authors**: Barclay, Michael J.; Christie, William G.; Harris, Jeffrey H.; Kandel, Eugene; Schultz, Paul H.
- **Year / venue**: 1999 / Journal of Finance 54(1), 1-34
- **Id / URL**: doi:10.1111/0022-1082.00097
- **Setting**: 1997 order-handling rules
- **Key number**: Spreads fell after limit-order display
- **Code**: -
- **Relevance to finswarm**: Historical analogue of the public limit-order fringe (L2).
- **Flags**: Verified: via Crossref (title, five authors, volume, issue, pages)

### `bessembinder2003`

- **Title**: Trade Execution Costs and Market Quality after Decimalization
- **Authors**: Bessembinder, Hendrik
- **Year / venue**: 2003 / Journal of Financial and Quantitative Analysis 38(4), 747-777
- **Id / URL**: doi:10.2307/4126742
- **Setting**: 2001 decimalization
- **Key number**: Spreads fell
- **Code**: -
- **Relevance to finswarm**: Historical analogue of tick lever L3 and its quality cost.
- **Flags**: Verified: via Crossref (title, author, volume, issue, start page 747; end page 777 from the survey)

### `budish2015`

- **Title**: The High-Frequency Trading Arms Race: Frequent Batch Auctions as a Market Design Response
- **Authors**: Budish, Eric; Cramton, Peter; Shim, John
- **Year / venue**: 2015 / Quarterly Journal of Economics 130(4), 1547-1621
- **Id / URL**: doi:10.1093/qje/qjv027
- **Setting**: Frequent batch auctions
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Batch auctions (L8, conditional on mechanism D).
- **Flags**: Verified: via Crossref (title, authors, volume, issue, pages)

### `gode_sunder1993`

- **Title**: Allocative Efficiency of Markets with Zero-Intelligence Traders: Market as a Partial Substitute for Individual Rationality
- **Authors**: Gode, Dhananjay K.; Sunder, Shyam
- **Year / venue**: 1993 / Journal of Political Economy 101(1), 119-137
- **Id / URL**: doi:10.1086/261868
- **Setting**: CDA with ZI traders
- **Key number**: Institution alone yields efficiency
- **Code**: -
- **Relevance to finswarm**: Swarm ZI-C traders; 'the institution decides' premise (R2).
- **Flags**: Verified: via Crossref (title, authors, volume, issue, pages)

### `leal2019`

- **Title**: Market Stability vs. Market Resilience: Regulatory Policies Experiments in an Agent-Based Model with Low- and High-Frequency Trading
- **Authors**: Leal, Sandrine Jacob; Napoletano, Mauro
- **Year / venue**: 2019 / Journal of Economic Behavior & Organization 157, 15-41
- **Id / URL**: doi:10.1016/j.jebo.2017.04.013
- **Setting**: ABM, transaction taxes / resting time
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Non-LLM lever comparison (H1).
- **Flags**: Verified: via Crossref (title, authors, volume 157, pages 15-41)

### `duffin2018`

- **Title**: Agent-Based Model Exploration of Latency Arbitrage in Fragmented Financial Markets
- **Authors**: Duffin, Matthew; Cartlidge, John
- **Year / venue**: 2018 / 2018 IEEE Symposium Series on Computational Intelligence (SSCI), pp. 2312-2320
- **Id / URL**: doi:10.1109/SSCI.2018.8628638
- **Setting**: FBA vs CDA with adaptive traders
- **Key number**: FBA result reverses with adaptive traders
- **Code**: -
- **Relevance to finswarm**: Lesson: robustness to swarm composition (charter point 9).
- **Flags**: Verified: via Crossref. PLAN ERROR: the plan/survey cite a 'batch auctions with adaptive traders' paper at IEEE CIFEr 2018; no such Duffin & Cartlidge paper was found, the 2018 Duffin & Cartlidge paper is this SSCI latency-arbitrage paper. Check that the use in the text ('reverses wah_wellman2013 on FBA') is supported by this paper before citing.

### `wah_wellman2013`

- **Title**: Latency arbitrage, market fragmentation, and efficiency
- **Authors**: Wah, Elaine; Wellman, Michael P.
- **Year / venue**: 2013 / Proceedings of the 14th ACM Conference on Electronic Commerce (EC '13), pp. 855-872
- **Id / URL**: doi:10.1145/2482540.2482577
- **Setting**: Latency arbitrage, FBA
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Pre-LLM FBA precedent (reversed by duffin2018).
- **Flags**: Verified: via Crossref (title, authors, venue, pages). Crossref also lists a duplicate DOI 10.1145/2492002.2482577.

### `changcarteapenalva2022`

- **Title**: Algorithmic Collusion in Electronic Markets: The Impact of Tick Size
- **Authors**: Cartea, Álvaro; Chang, Patrick; Penalva, José
- **Year / venue**: 2022 / SSRN working paper (Oxford-Man Institute, Universidad Carlos III); no published version found in two searches
- **Id / URL**: SSRN 4105954, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4105954
- **Setting**: Learning algorithms as ODEs in repeated games, applied to market making; larger tick size weakens competition, smaller tick slows convergence (qualitative)
- **Key number**: none confirmed
- **Code**: none found
- **Relevance to finswarm**: RL market-maker comparator for our tick lever L3 (our LLM dealers show no tick effect); the collusive markup spans many ticks, so a null is plausible by theory.
- **Flags**: Verified: only from search snippets and a third-party PDF copy (title, author order, qualitative findings). UNVERIFIED: SSRN page (403), posting date, numbers, code. Same group as `cartea2022tick` (check for duplication of the same paper before citing both)

### `cartea2022tick`

- **Title**: Algorithms and Supracompetitive Prices in Electronic Markets: The Impact of Tick Size
- **Authors**: Cartea, Álvaro; Chang, Patrick; Penalva, José
- **Year / venue**: 2022 / SSRN working paper
- **Id / URL**: SSRN 4105954; doi:10.2139/ssrn.4105954
- **Setting**: RL dealers, tick size
- **Key number**: Wider tick -> easier RL collusion
- **Code**: -
- **Relevance to finswarm**: Tick lever L3 precedent.
- **Flags**: Verified: via Crossref record of doi:10.2139/ssrn.4105954 (title, three authors, 2022). SSRN page returned HTTP 403.

### `cartea2022dealers`

- **Title**: AI-driven liquidity provision in OTC financial markets
- **Authors**: Cartea, Álvaro; Chang, Patrick; Mroczka, Mateusz; Oomen, Roel
- **Year / venue**: 2022 / Quantitative Finance 22(12), 2171-2204
- **Id / URL**: doi:10.1080/14697688.2022.2130087
- **Setting**: Bandit dealers
- **Key number**: Collusion disappears with finer grid, more dealers, or reset of one dealer
- **Code**: -
- **Relevance to finswarm**: Precedents for L3, entry (L6) and replacement (1.2).
- **Flags**: Verified: via Crossref (title, four authors, volume, issue, pages)

### `cartea_chang_graumans`

- **Title**: Anonymity, Signaling, and Collusion in Limit Order Books
- **Authors**: Cartea, Álvaro; Chang, Patrick; Graumans, Rob
- **Year / venue**: 2025 / SSRN working paper
- **Id / URL**: SSRN 5080700; doi:10.2139/ssrn.5080700
- **Setting**: Market makers signal through large limit orders in an anonymous book; competitive vs collusive equilibria
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Non-LLM precedent for signalling-based tacit collusion among market makers (order-book venue).
- **Flags**: Verified: via Crossref record of doi:10.2139/ssrn.5080700 (title, three authors, 2025). SSRN page returned HTTP 403. NEW entry. Note: the paper's topic is market makers signalling through large limit orders (empirical, ETFs on a European exchange per a Risk.net summary).

### `conjeaud2026`

- **Title**: Algorithmic collusion under asynchronous price updating
- **Authors**: Conjeaud, Ivan; Abel, Gaspard; Kalogeratos, Argyris
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2608.01406
- **Setting**: Asynchronous updating
- **Key number**: (see source)
- **Code**: none
- **Relevance to finswarm**: H5 asynchrony lever (L5); never tested with LLMs.
- **Flags**: Verified: against arxiv.org/abs/2608.01406 (title, authors, date)

## Agent-based finance and simulators

### `sivachandran2026`

- **Title**: Mitigating Retaliatory Algorithmic Collusion in Repeated Games
- **Authors**: Sivachandran, Karthik; Paleja, Rohan
- **Year / venue**: 2026 / arXiv
- **Id / URL**: arXiv:2609.20548
- **Setting**: Repeated games, retaliation-based collusion
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Mitigation of retaliatory collusion; related to deterrence levers.
- **Flags**: Verified: against arxiv.org/abs/2609.20548 (title, authors, date)

### `abides`

- **Title**: ABIDES: Towards High-Fidelity Market Simulation for AI Research
- **Authors**: Byrd, David; Hybinette, Maria; Balch, Tucker Hybinette
- **Year / venue**: 2019 / arXiv
- **Id / URL**: arXiv:1904.12066
- **Setting**: Event-driven ABM of limit order books
- **Key number**: ~10k events/s (archived)
- **Code**: jpmorganchase/abides-jpmc-public
- **Relevance to finswarm**: Throughput benchmark for design choice.
- **Flags**: Verified: against arxiv.org/abs/1904.12066 (title, authors, date); code github.com/abides-sim/abides (organisation 'abides-sim'; README points to this arXiv paper). 'Agent-Based Interactive Discrete Event Simulation' is the repo tagline, not the paper title

### `pymarketsim`

- **Title**: A Financial Market Simulation Environment for Trading Agents Using Deep Reinforcement Learning
- **Authors**: Mascioli, Chris; Gu, Anri; Wang, Yongzhao; Chakraborty, Mithun; Wellman, Michael
- **Year / venue**: 2024 / Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF '24); code github.com/umichsrg/pymarketsim
- **Id / URL**: doi:10.1145/3677052.3698639; github:umichsrg/pymarketsim
- **Setting**: Market simulator
- **Key number**: ~15k actions/s
- **Code**: -
- **Relevance to finswarm**: Throughput benchmark.
- **Flags**: Verified: via Crossref record of doi:10.1145/3677052.3698639 (title, five authors, venue, 2024); the name 'PyMarketSim' and the link to this paper come from secondary sources (search results) and the repo page (a fork of dipplestix/pymarketsim, no README text retrieved), so the paper-to-repo link is not confirmed from a primary page

### `pams`

- **Title**: PAMS: Platform for Artificial Market Simulations
- **Authors**: Hirano, Masanori; Takata, Ryosuke; Izumi, Kiyoshi
- **Year / venue**: 2023 / arXiv
- **Id / URL**: arXiv:2309.10729
- **Setting**: ~1,000 rule agents + 0-5 LLM
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Direct precedent for a hybrid pyramid (swarm + small LLM layer).
- **Flags**: Verified: against arxiv.org/abs/2309.10729 (title, authors, date); code github.com/masanorihirano/pams (EPL-1.0; README asks to cite this arXiv entry). The plan's 'arXiv:2510.12189 (companion)' is a different paper (hashimoto2025)

### `hashimoto2025`

- **Title**: Agent-Based Simulation of a Financial Market with Large Language Models
- **Authors**: Hashimoto, Ryuji; Takayanagi, Takehiro; Suzuki, Masahiro; Izumi, Kiyoshi
- **Year / venue**: 2025 / arXiv
- **Id / URL**: arXiv:2510.12189
- **Setting**: LLM agents inside a PAMS-style artificial market
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: LLM-financial-simulator precedent (hybrid with rule-based FCL agents).
- **Flags**: Verified: against arxiv.org/abs/2510.12189 (title, authors, date). NEW entry; the survey calls it the 'FCLAgent hybrid' paper (name not seen on the abstract page). Wrongly attached to pams in the plan.

### `bse`

- **Title**: BSE: A Minimal Simulation of a Limit-Order-Book Stock Exchange
- **Authors**: Cliff, Dave
- **Year / venue**: 2018 / EMSS 2018 (30th European Modeling and Simulation Symposium); arXiv; code on GitHub (MIT)
- **Id / URL**: arXiv:1809.06027; github:davecliff/BristolStockExchange
- **Setting**: Teaching LOB simulator
- **Key number**: -
- **Code**: BSE
- **Relevance to finswarm**: Prior simulator.
- **Flags**: Verified: against arxiv.org/abs/1809.06027 (title, author, date, comment: EMSS-2018) and the GitHub README (citation, MIT licence)

### `maxe`

- **Title**: Fast Agent-Based Simulation Framework with Applications to Reinforcement Learning and the Study of Trading Latency Effects
- **Authors**: Belcak, Peter; Calliess, Jan-Peter; Zohren, Stefan
- **Year / venue**: 2020 / arXiv; MABS@AAMAS 2021 workshop
- **Id / URL**: arXiv:2008.07871
- **Setting**: Simulator
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Prior simulator.
- **Flags**: Verified: against arxiv.org/abs/2008.07871 (title, authors, date); arXiv comment: MABS@AAMAS 2021. 'MAXE' is the framework name (not in the title)

### `jaxlob`

- **Title**: JAX-LOB: A GPU-Accelerated limit order book simulator to unlock large scale reinforcement learning for trading
- **Authors**: Frey, Sascha; Li, Kang; Nagy, Peer; Sapora, Silvia; Lu, Chris; Zohren, Stefan; Foerster, Jakob; Calinescu, Anisoara
- **Year / venue**: 2023 / arXiv
- **Id / URL**: arXiv:2308.13289
- **Setting**: GPU limit order book
- **Key number**: needs a GPU
- **Code**: -
- **Relevance to finswarm**: Rejected: no CUDA.
- **Flags**: Verified: against arxiv.org/abs/2308.13289 (title, authors, date)

### `mars`

- **Title**: MarS: a Financial Market Simulation Engine Powered by Generative Foundation Model
- **Authors**: Li, Junjie; Liu, Yang; Liu, Weiqing; Fang, Shikai; Wang, Lewen; Xu, Chang; Bian, Jiang
- **Year / venue**: 2024 / arXiv; ICLR 2025
- **Id / URL**: arXiv:2409.07486
- **Setting**: Market simulation engine
- **Key number**: needs a GPU
- **Code**: -
- **Relevance to finswarm**: Rejected: GPU; no collusion metric.
- **Flags**: Verified: against arxiv.org/abs/2409.07486 (title, authors, date); arXiv comment: ICLR 2025; code github.com/microsoft/MarS (MIT)

### `twinmarket`

- **Title**: TwinMarket: A Scalable Behavioral and Social Simulation for Financial Markets
- **Authors**: Yang, Yuzhe; Zhang, Yifei; Wu, Minghao; Zhang, Kaidi; Zhang, Yunmiao; Yu, Honghai; Hu, Yan; Wang, Benyou
- **Year / venue**: 2025 / arXiv
- **Id / URL**: arXiv:2502.01506
- **Setting**: LLM market simulator
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: LLM finance simulator without collusion metrics (gap 12).
- **Flags**: Verified: against arxiv.org/abs/2502.01506 (title, authors, date)

### `stockagent`

- **Title**: When AI Meets Finance (StockAgent): Large Language Model-based Stock Trading in Simulated Real-world Environments
- **Authors**: Zhang, Chong; Liu, Xinyi; Zhang, Zhongmou; Jin, Mingyu; Li, Lingyao; Wang, Zhenting; Hua, Wenyue; Shu, Dong; Zhu, Suiyuan; Jin, Xiaobo; Li, Sujian; Du, Mengnan; Zhang, Yongfeng
- **Year / venue**: 2024 / arXiv; ACM Transactions on Intelligent Systems and Technology (per arXiv comment)
- **Id / URL**: arXiv:2407.18957
- **Setting**: LLM stock market
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Same.
- **Flags**: Verified: against arxiv.org/abs/2407.18957 (title, authors, date); arXiv comment: published in ACM TIST

### `tradingagents`

- **Title**: TradingAgents: Multi-Agents LLM Financial Trading Framework
- **Authors**: Xiao, Yijia; Sun, Edward; Luo, Di; Wang, Wei
- **Year / venue**: 2024 / arXiv
- **Id / URL**: arXiv:2412.20138
- **Setting**: LLM trading firm
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Same.
- **Flags**: Verified: against arxiv.org/abs/2412.20138 (title, authors, date)

### `fincon`

- **Title**: FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making
- **Authors**: Yu, Yangyang; Yao, Zhiyuan; Li, Haohang; Deng, Zhiyang; Cao, Yupeng; Chen, Zhi; Suchow, Jordan W.; Liu, Rong; Cui, Zhenyu; Xu, Zhaozhuo; Zhang, Denghui; Subbalakshmi, Koduvayur; Xiong, Guojun; He, Yueru; Huang, Jimin; Li, Dong; Xie, Qianqian
- **Year / venue**: 2024 / arXiv
- **Id / URL**: arXiv:2407.06567
- **Setting**: LLM finance agents
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Same.
- **Flags**: Verified: against arxiv.org/abs/2407.06567 (title, authors, date)

### `econagent`

- **Title**: EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities
- **Authors**: Li, Nian; Gao, Chen; Li, Mingyu; Li, Yong; Liao, Qingmin
- **Year / venue**: 2023 / arXiv; ACL 2024 (main conference)
- **Id / URL**: arXiv:2310.10436
- **Setting**: LLM macro agents
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Same.
- **Flags**: Verified: against arxiv.org/abs/2310.10436 (title, authors, date); arXiv comment: ACL 2024 main conference

### `asfm`

- **Title**: Simulating Financial Market via Large Language Model based Agents
- **Authors**: Gao, Shen; Wen, Yuntao; Zhu, Minghang; Wei, Jianing; Cheng, Yuhan; Zhang, Qunzi; Shang, Shuo
- **Year / venue**: 2024 / arXiv
- **Id / URL**: arXiv:2406.19966
- **Setting**: LLM stock market
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Same.
- **Flags**: Verified: against arxiv.org/abs/2406.19966 (title, authors, date)

### `lopezlira2025`

- **Title**: Can Large Language Models Trade? Testing Financial Theories with LLM Agents in Market Simulations
- **Authors**: Lopez-Lira, Alejandro
- **Year / venue**: 2025 / arXiv (also SSRN 5217340); code github.com/alejandroll10/llm_trading_sim
- **Id / URL**: arXiv:2504.10789
- **Setting**: LLM trading behaviour
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Same.
- **Flags**: Verified: against arxiv.org/abs/2504.10789 (title, author, date) and github.com/alejandroll10/llm_trading_sim (README cites the paper, SSRN 5217340)

### `llm_trading_sim`

- **Title**: llm_trading_sim: trading simulation environment with LLM-powered agents
- **Authors**: Lopez-Lira, Alejandro
- **Year / venue**: 2025 / GitHub repository (companion code of lopezlira2025)
- **Id / URL**: github:alejandroll10/llm_trading_sim
- **Setting**: LLM trading simulation
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Same.
- **Flags**: Verified: against github.com/alejandroll10/llm_trading_sim (owner alejandroll10, description, paper reference). Licence not stated on the page; the repository year is taken from the companion paper.


## Market-quality and dependence metrics (classics)

### `lerner1934`

- **Title**: The Concept of Monopoly and the Measurement of Monopoly Power
- **Authors**: Lerner, Abba P.
- **Year / venue**: 1934 / Review of Economic Studies 1(3), 157-175
- **Id / URL**: doi:10.2307/2967480
- **Setting**: Classic reference (not LLM-specific)
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Lerner index L=(P-MC)/P, the price-cost markup as a measure of market power. Used in finswarm for the the markup metric (collusion/quality metrics).
- **Flags**: Verified: against Crossref record for the DOI (title, author, volume/issue/pages)

### `abrantes_metz2006`

- **Title**: A variance screen for collusion
- **Authors**: Abrantes-Metz, Rosa M.; Froeb, Luke M.; Geweke, John; Taylor, Christopher T.
- **Year / venue**: 2006 / International Journal of Industrial Organization 24(3), 467-486
- **Id / URL**: doi:10.1016/j.ijindorg.2005.10.003
- **Setting**: Classic reference (not LLM-specific)
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Collusion screen based on abnormally low price variance / dispersion. Used in finswarm for the the price-dispersion screen (collusion metrics).
- **Flags**: Verified: against Crossref record for the DOI (title, authors, volume/issue/pages)

### `hasbrouck2007`

- **Title**: Empirical Market Microstructure: The Institutions, Economics, and Econometrics of Securities Trading
- **Authors**: Hasbrouck, Joel
- **Year / venue**: 2007 / Oxford University Press (book)
- **Id / URL**: isbn:9780195301649
- **Setting**: Classic reference (not LLM-specific)
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Textbook definitions of quoted, effective and realised spread and market depth. Used in finswarm for the spread and depth market-quality metrics.
- **Flags**: Verified: against Open Library record OL7392620M (title, author, OUP, ISBN); Open Library lists Dec 2006 publication, year 2007 follows the common citation; subtitle not re-checked

### `hasbrouck1993`

- **Title**: Assessing the Quality of a Security Market: A New Approach to Transaction-Cost Measurement
- **Authors**: Hasbrouck, Joel
- **Year / venue**: 1993 / Review of Financial Studies 6(1), 191-212
- **Id / URL**: doi:10.1093/rfs/6.1.191
- **Setting**: Classic reference (not LLM-specific)
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Pricing error: std. deviation of the transitory component of price relative to the efficient price. Used in finswarm for the pricing-error market-quality metric.
- **Flags**: Verified: against Crossref record for the DOI

### `kyle1985`

- **Title**: Continuous Auctions and Insider Trading
- **Authors**: Kyle, Albert S.
- **Year / venue**: 1985 / Econometrica 53(6), 1315-1335
- **Id / URL**: doi:10.2307/1913210
- **Setting**: Classic reference (not LLM-specific)
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Kyle's lambda: price impact per unit of signed order flow (inverse market depth). Used in finswarm for the price-impact (lambda) metric.
- **Flags**: Verified: against Crossref record for the DOI (title, author, volume/issue, start page; end page 1335 from memory)

### `andersen2001`

- **Title**: The Distribution of Realized Exchange Rate Volatility
- **Authors**: Andersen, Torben G.; Bollerslev, Tim; Diebold, Francis X.; Labys, Paul
- **Year / venue**: 2001 / Journal of the American Statistical Association 96(453), 42-55
- **Id / URL**: doi:10.1198/016214501750332965
- **Setting**: Classic reference (not LLM-specific)
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Realised volatility: sum of squared high-frequency returns. Companion paper: Andersen et al. 2003, Econometrica 71(2), 579-625, doi:10.1111/1468-0262.00418, 'Modeling and Forecasting Realized Volatility' (also Crossref-verified); key kept for the 2001 paper. Used in finswarm for the realised-volatility metric.
- **Flags**: Verified: against Crossref records for both DOIs

### `smith1962`

- **Title**: An Experimental Study of Competitive Market Behavior
- **Authors**: Smith, Vernon L.
- **Year / venue**: 1962 / Journal of Political Economy 70(2), 111-137
- **Id / URL**: doi:10.1086/258609
- **Setting**: Classic reference (not LLM-specific)
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Experimental double-auction markets; efficiency as realised share of total surplus. Used in finswarm for the total-surplus / allocative-efficiency metric.
- **Flags**: Verified: against Crossref search record (title, author, volume/issue/pages)

### `box_jenkins1970`

- **Title**: Time Series Analysis: Forecasting and Control
- **Authors**: Box, George E. P.; Jenkins, Gwilym M.
- **Year / venue**: 1970 / Holden-Day, San Francisco (book)
- **Id / URL**: none (no DOI)
- **Setting**: Classic reference (not LLM-specific)
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Classic reference for the sample cross-correlation function at lags. Used in finswarm for the lagged cross-correlation dependence metric.
- **Flags**: Verified: against Open Library confirms title, authors, 1970; publisher Holden-Day not confirmed on a fetched page (UNVERIFIED publisher)

### `granger1969`

- **Title**: Investigating Causal Relations by Econometric Models and Cross-spectral Methods
- **Authors**: Granger, C. W. J.
- **Year / venue**: 1969 / Econometrica 37(3), 424-438
- **Id / URL**: doi:10.2307/1912791
- **Setting**: Classic reference (not LLM-specific)
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Granger causality: past of X improves prediction of Y. Used in finswarm for the Granger-causality dependence metric.
- **Flags**: Verified: against Crossref record for the DOI (end page 438 from memory)

### `schreiber2000`

- **Title**: Measuring Information Transfer
- **Authors**: Schreiber, Thomas
- **Year / venue**: 2000 / Physical Review Letters 85(2), 461-464
- **Id / URL**: doi:10.1103/PhysRevLett.85.461
- **Setting**: Classic reference (not LLM-specific)
- **Key number**: -
- **Code**: -
- **Relevance to finswarm**: Transfer entropy: directed, nonparametric information flow between series. Used in finswarm for the transfer-entropy dependence metric.
- **Flags**: Verified: against Crossref record for the DOI
