# Chaos RRS – Connecting Humans

<!--
<a href="https://git.io/chaos-rrs"><img alt="Logo" align="right" width="200" src="https://raw.githubusercontent.com/kdevo/chaos-rrs/main/.github/chaos.png"></a>

> There are only patterns, patterns on top of patterns,
patterns that affect other patterns. Patterns hidden by
patterns. Patterns within patterns. If you watch close,
history does nothing but repeat itself.
>
> What we call chaos is just patterns we haven’t recognized.
What we call random is just patterns we can’t decipher.
What we can’t understand we call nonsense.
>
> – Chuck Palahniuk • Survivor
-->

### A first of its kind framework for researching Reciprocal Recommender Systems (RRSs)
[![GitHub Release](https://img.shields.io/github/v/release/kdevo/chaos-rrs?style=flat-square&color=%230097a7&logo=github)](https://github.com/kdevo/chaos-rrs/releases/latest)
[![PDF Download](https://img.shields.io/badge/thesis-PDF-%230097a7?logo=PDF&style=flat-square)](https://github.com/gohugoio/hugo/releases)

Chaos is the accompanying proof of concept of my master's thesis 
"Engineering a Hybrid Reciprocal Recommender System Specialized in Human-to-Human Implicit Feedback" (FH Aachen, February 2021).

I've recently [published my thesis][thesis], and everyone can now fully read, comprehend and reproduce the results to enter the world of human-to-human recommendations. 
You're invited to use the link for documentation, to understand what Chaos does, what RRSs and their challenges are, or just for your own research. 

### Motivation

Just as the Recommender System (RS) domain, research on RRSs suffers from very limited reproducibility. I've engineered Chaos to tackle this problem ([thesis section 5.2][thesis-pdf]):
> Chaos aims to build a solid bridge between the research and development departments of RRSs, with the ultimate goal that in the future,
> improvements are not developed in a decentralized fashion anymore.

I've contributed my work to the public to help to accelerate research together. This project can only flourish when other contributors join!

It is currently not meant to be ready for production or commercial applications. 
[Please consult me][contact] to discuss potential use-cases where Chaos could help you or your business with. 
If applicable, you can also [start a public discussion][discussion].

## Getting started: Experiments

This section is a great demonstration of Chaos capabilities and usages.
Work your way through the steps and take your time to experiment. The [second experiment](#2-chaos-for-github) is an exciting possibility to get to know your personal social GitHub universe!

### Preparation

1. Clone this repo with all submodules: `git clone --recurse-submodules https://github.com/kdevo/chaos-rrs.git`
2. Conda is needed as a cross-platform package and environment manager. Refer to the [user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for installation.
3. Create the `chaos` environment via `	conda env create -f environment.yml` (see also [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file))
4. Wait until the installation is finished and then activate `chaos` by calling `conda activate chaos`
5. Trust the notebooks: `jupyter trust notebooks/`

### 1. Intro: Study community example

This optional example functions as a basic introduction to the framework's core components (see [thesis section 4.1 onwards][thesis-pdf]).

1. Follow [Preparation](#preparation)
2. Start JupyterLab as follows: [`jupyter lab notebooks/learning-group.ipynb`](notebooks/learning-group.ipynb)

### 2. Chaos for GitHub

In this advanced scenario (see [thesis section 4.4][thesis-pdf]), you learn how to create a tailored RRS based on your own personal GitHub user
profile!

1. Follow [Preparation](#preparation) if not done yet
2. Have your GitHub account, and a stable internet connection ready
3. Start JupyterLab as follows: [`jupyter lab notebooks/chaos-github.ipynb`](notebooks/chaos-github.ipynb)

## Features

This section provides a brief and non-complete overview of Chaos features. 
> :warning: Please read [thesis chapter 4][thesis-pdf] for the complete description.

### Supported recommendation algorithms

- [`LFMPredictor`](chaos/recommend/predict/predictor.py#76): Latent Factor Model User-to-User RS via [LightFM](https://github.com/lyst/lightfm) (able to handle cold-start through metadata), wrapped with [`ReciprocalWrapper`](./chaos/recommend/predict/reciprocal.py#L85) to fulfill reciprocity criterion
- [`RCFPredictor`](chaos/recommend/predict/reciprocal.py#L189): [Reciprocal Collaborative Filtering](https://arxiv.org/abs/1501.06247v2), implemented as baseline algorithm
- Anything that you provide by implementing the [`Predictor`](chaos/recommend/predict/predictor.py) class

### Relevant Technologies

<img alt="Tech Stack" width="400" src="https://raw.githubusercontent.com/kdevo/chaos-rrs/main/.github/tech.png">

- Pandas' DataFrame for `DataModel` and more
- Grapresso with NetworkX backend for representing the interaction graph from the `DataModel`
- LightFM for a LFM based on Factorization Machines to mitigate the cold-start problem
- TensorBoard/Projector to visualize the learned embeddings
- altair-viz for visualizations, e.g. for the LFMEvaluator results
- spaCy for NLP feature extraction used in the `process` module
- Optuna for Hyperparameter Optimization
- Jupyter Lab for reproducible notebooks

### Data Model


Chaos' `DataModel` is kept relatively simple:

<img alt="Interaction Data Model" align="right" width="400" src="https://raw.githubusercontent.com/kdevo/chaos-rrs/main/.github/graph.png">

- Interactions between users are stored in a graph/network:
    - User `a` is interested in `b` if at least one interaction exists
    - If `a` is interested in `b` an edge (`a`, `b`) will be created

      If `b` is interested in `a` an edge (`b`, `a`) will be created
    - Each interaction linearly increases the strength of an edge

- Metadata of users are stored in a long-format user data frame:
    - Each row represents the metadata of one user
    - Columns can contain variable data in form of collections (i.e., tags)
    - Stored in a pandas data frame

### Components

The following provides a broad overview of the components:

- `fetch` - Retrieve data from a source
    - Very simple `Source` interface (implement a function that returns Chaos' DataModel)
- `process` - Process features, i.e. common Feature Engineering tooling for (R)RS
    - Extract common tags
    - NLP to retrieve textual entities
    - Graph-based feature extraction
- `recommend` - (R)RS implementations and typical workflows
    - `Translator` is used to make Chaos' data model understandable to the Predictor
    - `CandidateGenerator`s can be chained to retrieve compatible candidates
    - `Predictor` is used for actual recommendations
    - `Evaluator` for performance evaluation/comparison/optimization (e.g. precision, recall and f1)
    - `ReciprocalWrapper` helper to transform an RS to an RRS and perform reciprocal recommendations

The `recommend` module is the core. You can skip `fetch` and `process` entirely, if you provide a proper DataModel on
your own. The above image shows the UML of the model.

[thesis-pdf]: https://kdevo.github.io/docs/2021-Kai_Dinghofer-Master_Thesis-en.pdf
[thesis]: https://kdevo.github.io/#master-thesis
[contact]: https://kdevo.github.io/#contact
[discussion]: https://github.com/kdevo/chaos-rrs/discussions/categories/ideas