# Draft conventions

Paper: `DM.tex` (revtex4-2, two-column, APS prd). Compile from this folder:
```
pdflatex -interaction=nonstopmode DM.tex   # run twice
```

## Writing style

- **Reduce `;`, `:`, em-dashes (`---`), and en-dashes (`--`) in prose.** Prefer short period-separated sentences. If a colon feels natural, try splitting into two sentences first. Replace `---` parentheticals with commas or `( )`. Legitimate en-dashes stay: numeric ranges (`epochs 49--249`, `10--35`) and named compounds (`Metropolis--Hastings`, `Kullback--Leibler`).
- **Plain physics language over ML jargon.** Audience includes physicists who may not know "transition kernel", "Ornstein--Uhlenbeck process", "Gaussian kernel". Paraphrase (e.g. "the distribution of $\phi_t$ conditioned on $\phi_0$" instead of "the transition kernel").
- **Let formulas speak when words are vaguer.** For the VP variance decomposition, `\mathrm{Var}[\phi_t]=\bar\alpha(t)\,\mathrm{Var}[\phi_0]+(1-\bar\alpha(t))=1` is clearer than a prose gloss of "signal contribution + noise contribution".
- **Don't pad to explain.** If a revision grows longer without being clearer, back off. The user will flag this; respect it.
- **Keep generic sections generic.** The diffusion-model intro (`sec:dm_framework`) must not presume lattice discretisation — use `\delta(x-x')`, "spatial points/components", etc. Lattice-specific language belongs in the $\phi^4$ sections.
- **Be precise about what a variable depends on.** "Linearly" is ambiguous ("linear in $\phi$" vs "linear in $t$"); spell out explicitly.
- **Keep each subsection's primary object consistent.** If paragraph 1 prescribes $\sigma^2(t)$ but paragraph 2 prescribes $g(t)$, pick one as primary and derive the other.
- **Parallel structure between sibling paragraphs.** VP and VE paragraphs share the same skeleton: state $(f,g)$, give physical intuition for the drift, show the SDE, integrate, state the conditional Gaussian, annotate mean/variance, explain the name, cite.
- **`sec:dm_framework` subsection order is fixed:** forward SDE → reverse-time generation → score matching. Reverse-time generation introduces $s(\phi,t)$ via Anderson's formula; score matching is how to learn it. Inverting this creates a forward reference from "score matching" back to "the reverse dynamics", which is circular.
- **One phrase, not a paragraph, for properties implicit in the formula.** For $s(\phi,t)=\nabla_\phi\log p_t(\phi)$, "a vector of the same dimension as $\phi$" is enough — the gradient structure is already visible. Don't expand into two sentences on vector-valued functions over configuration space.
- **Unpack ML-sounding claims operationally when they'd leave a physicist guessing.** "The three dynamics share the same one-time marginals" is opaque; spell out the recipe (ensemble initialised at $p_{\text{data}}$ for forward integration or at $p_{t_{\max}}$ for reverse integration; distribution at every intermediate $t$ agrees across dynamics; individual trajectories differ).

## Notation

- **Time as subscript:** `\phi_0`, `\phi_t`, `\phi_{t_{\max}}` — never `\phi(0)`, `\phi(t)`, `\phi(T)`.
- **Terminal time is `t_{\max}`**, not `T` (reserved for temperature in later sections).
- **VE hyperparameter `\sigma`** in $g(t)=\sigma^t$ is \emph{not} the Song--Ermon $\sigma_{\max}$. They are related by $\sigma_{\max}=\sigma(1)$ via `\eq{eq:ve_sigma_schedule}`. Flag any draft passage that conflates them.
- **White noise on a continuum field:** `\eta(t,x)` with `\langle\eta(t,x)\eta(t',x')\rangle = \delta(x-x')\,\delta(t-t')`. Do not replace Dirac $\delta(x-x')$ with Kronecker $\delta_{x,x'}$ in the generic intro — that presumes a lattice.
- **Terminal distribution is `p_{t_{\max}}`.** Don't introduce a separate symbol (`\pi` etc.) — `p_{t_{\max}}` is already set up by the forward-SDE discussion and is self-explanatory. Prefer "terminal distribution" over the ML word "prior".
- **Reverse-time Wiener increment: avoid "Wiener process running backwards in time".** That phrasing makes `\bar w` sound like a time-reversed process. Spell out operationally: the reverse SDE is integrated from `t=t_{\max}` down to `t=0`, so `\mathrm{d}t<0`, and `\mathrm{d}\bar w` is an independent standard Wiener increment at each reverse step. Follows the Song et al.\ 2021 convention.

## APS cross-reference macros

Defined in the preamble; always use these instead of raw `\ref` / `\eqref`.

| Macro | Expansion | Use |
|---|---|---|
| `\Eq{label}` | `Eq.~(n)` | default for equation refs |
| `\fig{label}` / `\Fig{label}` | `Fig.~\ref{label}` | identical |
| `\figs{a}{b}` / `\Figs{a}{b}` | `Figs.~\ref{a} and \ref{b}` | |
| `\Tab{label}`, `\Tabs{a}{b}` | `Tab.~\ref{...}`, `Tabs.~\ref{a} and \ref{b}` | |
| `\Sec{label}`, `\App{label}` | analogous | |

Never write `equation~\eq{...}` (renders "equation~(n)"); use `\Eq{...}`.
For two equations use `\Eq{a} and~\Eq{b}`, not `\Eq{a}--\Eq{b}` (the latter renders "Eq.~(a)--Eq.~(b)", which reads awkwardly).

## Editing workflow

- Section-by-section, paragraph-by-paragraph under the user's direction. Don't pre-emptively rewrite sections the user hasn't asked about.
- When changing notation, use `Edit` with `replace_all: true` after confirming the pattern is unambiguous (e.g. `\phi(0)` → `\phi_0`).
- Before recommending a swap like `T` → `t_{\max}`, grep usages and flag collisions.
- After an edit, summarise the change in 2--4 bullets: what changed, why. Don't restate the prose.
