# FinOps LLM --- Domain-Adapted Qwen3-4B for Cloud Cost Optimization

A domain-adapted language model exploring how parameter-efficient
fine-tuning can support cloud financial operations (FinOps), workload
analysis, and cloud cost optimization.

**Hugging Face adapter:**
[Chandu06/finops-qwen3-4b](https://huggingface.co/Chandu06/finops-qwen3-4b)

```{=html}
<p align="center">
```
`<img src="Finops_tuning_architecture.png" alt="FinOps LLM architecture: Qwen3-4B with a LoRA adapter for cloud cost optimization" width="100%">`{=html}
```{=html}
</p>
```
> **Project status:** The LoRA adapter is published on Hugging Face, and
> inference was tested in a Kaggle GPU environment. This repository is a
> portfolio/research project. A persistent public API, production
> monitoring, and end-to-end reproducibility have not been established.

------------------------------------------------------------------------

## Table of Contents

-   [Overview](#overview)
-   [Architecture](#architecture)
-   [Technology Stack](#technology-stack)
-   [Dataset Strategy](#dataset-strategy)
-   [Fine-Tuning Workflow](#fine-tuning-workflow)
-   [Results and Evaluation](#results-and-evaluation)
-   [Quick Start: Inference](#quick-start-inference)
-   [Docker](#docker)
-   [Repository Structure](#repository-structure)
-   [Limitations](#limitations)
-   [Deployment Constraints](#deployment-constraints)
-   [Responsible Use](#responsible-use)
-   [Future Work](#future-work)
-   [Links](#links)

## Overview

Cloud cost optimization involves more than identifying expensive
resources. Sound recommendations depend on workload behavior,
utilization, configuration, pricing context, and the evidence available
for each decision.

FinOps LLM explores whether a compact, domain-adapted language model can
produce structured, evidence-aware FinOps analysis. Its intended use
cases include:

-   Interpreting cloud workload and utilization data
-   Identifying potential cost anomalies
-   Suggesting resource right-sizing opportunities
-   Reviewing compute and storage optimization options
-   Producing structured recommendations and identifying missing
    evidence

This model is an analytical assistant, not a replacement for cloud
billing systems, current pricing sources, or human review.

## Architecture

1.  **User input:** A FinOps question and any available cost, workload,
    or utilization evidence.
2.  **Pretrained base model:** `Qwen/Qwen3-4B-Instruct-2507` supplies
    the underlying language-model capability.
3.  **LoRA adapter:** `Chandu06/finops-qwen3-4b` contains the published
    domain-adaptation weights.
4.  **Inference pipeline:** Transformers loads the base model; PEFT
    attaches the adapter; the model generates a response.
5.  **Structured output:** The intended report format separates
    observations, calculations, evidence, assessment, recommendations,
    savings, missing evidence, and confidence.

**Important:** The Hugging Face repository contains the LoRA adapter,
not a standalone copy of the base model. Both the base model and adapter
are required for inference.

## Technology Stack

  Component                      Technology
  ------------------------------ ---------------------------------------
  Base LLM                       Qwen3-4B-Instruct-2507
  Fine-tuning method             LoRA / PEFT
  Model loading and generation   Hugging Face Transformers
  Quantized inference            bitsandbytes, 4-bit NF4 configuration
  Training workflow              Python, Hugging Face Trainer
  Experiment environment         Kaggle GPU runtime
  Packaging                      Dockerfile
  Tests                          pytest

Check `requirements.txt` for the dependency versions in the current
repository. Hardware and package compatibility must be validated in the
environment where the project is run.

## Dataset Strategy

The training notebook expects domain-specific FinOps examples stored as
JSONL shards. Its configured input layout is:

``` text
training_shards/
├── layer1_finops_train_part_1.jsonl
├── layer1_finops_train_part_2.jsonl
├── layer1_finops_train_part_3.jsonl
└── layer1_finops_train_part_4.jsonl
```

The notebook expects **8,400 examples per shard**---33,600 examples if
all four shards are present. This is the expected input layout, not an
independently verified count of unique, valid, or high-quality examples
in the final training corpus.

### Intended example coverage

The dataset strategy is designed to teach the model to:

-   Interpret cloud cost and resource-utilization scenarios
-   Ground recommendations in the supplied evidence
-   Separate observed facts from assumptions
-   Avoid inventing prices, configurations, workload behavior, or
    historical data
-   State which evidence is missing when a conclusion cannot be
    supported
-   Follow a consistent report structure

### Recommended data-quality checks

For a stronger and more reproducible dataset, audit:

-   Duplicate and near-duplicate examples across shards
-   Train/validation/test leakage
-   Arithmetic errors and unsupported savings claims
-   Missing provider, region, currency, and time-period context
-   Implausible billing and utilization combinations
-   Coverage across cloud services and workload types
-   Personally identifiable, confidential, or proprietary information

Document the data sources, licenses, generation or collection method,
filtering rules, split strategy, and quality checks for each dataset
version. Do not publish restricted training data.

**Dataset transparency:** The notebook expects the shard files, but the
dataset files are not included in the public repository by default.
Dataset provenance, deduplication statistics, exact split methodology,
and independent quality-audit results should be documented before making
stronger claims about dataset quality.

## Fine-Tuning Workflow

The training notebook follows this general process:

1.  **Prepare the environment** and locate the JSONL training shards.
2.  **Load the tokenizer** associated with the Qwen base model.
3.  **Load the base model** using 4-bit quantization to reduce GPU
    memory requirements.
4.  **Configure LoRA** with rank `r=16`, alpha `32`, dropout `0.05`, and
    target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
    `up_proj`, and `down_proj`.
5.  **Tokenize examples** using the model's chat template and mask
    non-assistant tokens so the loss focuses on assistant responses.
6.  **Train with Hugging Face Trainer** using the configured shards.
7.  **Save the adapter** and perform a basic generation check.
8.  **Publish the adapter** separately from the base model on Hugging
    Face.

The notebook is configured to process four shards sequentially, with one
epoch per shard. This should not be described as one epoch over a
globally shuffled dataset. Shard order, optimizer-state continuity,
checkpoint recovery, random seeds, and the evaluation split should be
reviewed and documented before claiming fully reproducible training.

### Why LoRA?

LoRA trains a relatively small set of adapter parameters instead of
updating all base-model weights. This reduces the number of trainable
parameters and allows the adapter to be stored and shared separately
from the larger base model.

### Why 4-bit quantization?

The 4-bit NF4 configuration reduces memory required to load the base
model. It does not eliminate the need for GPU memory for activations and
the generation key/value cache, or the storage and bandwidth required to
download the model.

## Results and Evaluation

### Verified project milestones

-   The adapter is published at
    [Chandu06/finops-qwen3-4b](https://huggingface.co/Chandu06/finops-qwen3-4b).
-   The adapter artifact is approximately **132 MB**.
-   The base model and adapter were loaded together in a Kaggle
    environment using two NVIDIA Tesla T4 GPUs and 4-bit NF4
    quantization.
-   A basic inference run completed in that environment.

These are implementation and smoke-test milestones. They do not
establish production readiness or broad FinOps accuracy.

### Quantitative evaluation

A comparable, independently verified base-versus-fine-tuned benchmark is
not documented in the current project record. Therefore, this README
does not claim a measured accuracy improvement.

A rigorous next evaluation should compare the base and fine-tuned models
on the same held-out test examples and prompts. Useful measures include:

-   Correctness of deterministic cost calculations
-   Evidence support and factual consistency
-   Recommendation quality scored using a documented rubric
-   Unsupported or fabricated pricing and savings claims
-   Required-section completion rate
-   Performance across providers and workload types
-   Latency, peak GPU memory, and tokens per second

Publish the evaluation split, prompts, decoding settings, hardware,
metrics, and representative outputs alongside future benchmark results.
A few successful generations or training-time metrics alone are not
evidence of general domain accuracy.

## Quick Start: Inference

### Requirements

-   Python with compatible versions of PyTorch, Transformers, PEFT,
    Accelerate, and bitsandbytes
-   A compatible NVIDIA GPU and CUDA environment for the current 4-bit
    configuration
-   Internet access for downloading the base model and adapter on first
    use
-   Sufficient disk space, bandwidth, and runtime memory

### 1. Clone the GitHub repository

Replace the placeholders with the actual published repository URL and
directory name:

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### 2. Create and activate a virtual environment

``` bash
python -m venv .venv
```

**Windows PowerShell:**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run inference

``` bash
python inference.py
```

Enter a FinOps question when prompted. The intended loading flow uses
the base model ID `Qwen/Qwen3-4B-Instruct-2507` and downloads the
adapter from the Hugging Face Hub using `Chandu06/finops-qwen3-4b`.

The current script is a terminal-based inference entry point. It does
**not** start a web server or create a public API. Verify the checked-in
`inference.py` and dependency versions in a fresh environment before
relying on these commands.

## Docker

The repository includes a Dockerfile intended to package the inference
environment.

Build the image from the repository root:

``` bash
docker build -t finops-llm .
```

Run it on a host configured with the NVIDIA Container Toolkit:

``` bash
docker run --rm -it --gpus all finops-llm
```

The container downloads the base model and adapter at runtime; their
weights are not embedded in the image. The current Docker command starts
interactive terminal inference, not an HTTP service.

**Verification status:** A complete Docker build-and-run test has not
been confirmed. Treat Docker support as intended packaging until it
passes on a compatible GPU host. CUDA, drivers, PyTorch, Python,
Transformers, and bitsandbytes versions must be compatible.

## Repository Structure

The intended public repository layout is:

``` text
FinOps-LLM/
├── docs/
│   └── deployment_limitations.md
├── notebooks/
│   └── FinOps_LLM_Training.ipynb
├── tests/
│   └── test_output_format.py
├── .gitignore
├── Dockerfile
├── inference.py
├── requirements.txt
└── README.md
```

Keep large datasets, downloaded base-model weights, local adapter
copies, virtual environments, caches, and secrets out of Git. Link to
the Hugging Face adapter instead of committing its weights.

The architecture image referenced at the top of this README should be
committed at the repository root as `finops_tuning_architecture.png`.

## Limitations

-   **Adapter, not standalone:** The base Qwen model must also be
    downloaded and loaded.
-   **No established benchmark improvement:** Domain accuracy,
    calculation accuracy, and improvements over the base model have not
    been established through a documented held-out evaluation.
-   **No live pricing source:** Cloud prices vary by provider, region,
    service, discount, commitment, and time. The model is not a
    real-time pricing database.
-   **Potential hallucinations:** Recommendations and savings estimates
    can be incorrect or unsupported; independently validate
    calculations.
-   **Dataset documentation gaps:** Provenance, coverage, deduplication,
    and independent quality audits need fuller documentation.
-   **Training reproducibility:** The notebook has not been verified end
    to end in a clean environment. Shard order, optimizer-state
    continuity, random seeds, checkpoint recovery, and evaluation splits
    need explicit documentation.
-   **Output-format reliability:** Prompt instructions do not guarantee
    that every generated response follows the requested format. Validate
    the output programmatically.
-   **No live cloud integration:** The current project does not connect
    to AWS, Azure, or GCP billing APIs or inspect a user's cloud
    account.
-   **No persistent API or UI:** The current implementation provides
    terminal inference only.
-   **Operational behavior untested:** Concurrent requests, sustained
    service stability, throughput, latency under load, uptime, and
    monitoring have not been validated.

## Deployment Constraints

-   **GPU requirement:** The current quantized inference setup targets a
    compatible NVIDIA GPU. CPU-only execution is not the supported
    configuration documented here.
-   **Memory:** Four-bit weights reduce memory use but do not eliminate
    memory requirements for activations and the key/value cache. Usage
    varies with input length, output length, and runtime configuration.
-   **Initial download:** The base model is substantially larger than
    the adapter. First startup can require significant bandwidth, disk
    space, and time.
-   **Package compatibility:** CUDA runtime, NVIDIA drivers, PyTorch,
    Transformers, PEFT, Accelerate, and bitsandbytes must work together.
-   **Hosting costs:** Kaggle is an experimentation environment, not
    persistent production hosting. A continuously available service
    needs a separate host or managed inference provider and may incur
    GPU charges.
-   **Data protection:** Never commit API keys, cloud credentials,
    `.env` files, private billing exports, or confidential workload
    logs.
-   **Serving layer:** Public HTTP inference needs a server/API layer,
    request validation, authentication, rate limiting, logging controls,
    and monitoring. The current terminal script does not provide these.
-   **Security and governance:** A future deployment using real billing
    data should define access controls, retention, redaction, audit
    logging, and approval requirements for cost-changing actions.

## Responsible Use

Use FinOps LLM as a decision-support tool. Validate recommendations
against current provider billing data, pricing documentation, service
constraints, and workload requirements. Do not apply infrastructure
changes or make financial decisions solely on a generated response.
Treat savings figures as estimates unless independently calculated from
verified inputs.

## Future Work

-   Establish a documented, leakage-resistant train/validation/test
    split.
-   Benchmark the fine-tuned model against the original base model and
    report reproducible metrics.
-   Add automated tests for output structure, arithmetic, evidence
    support, and unsupported savings claims.
-   Improve checkpointing and recovery for long training runs.
-   Publish a dataset card and data-quality report.
-   Verify the Docker build and inference run on a clean GPU host.
-   Add an optional API layer only if a serving endpoint is needed.
-   Explore retrieval of current cloud pricing documentation for
    evidence-grounded responses.

## Links

-   **Hugging Face adapter:**
    https://huggingface.co/Chandu06/finops-qwen3-4b
-   **Base model:** https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

------------------------------------------------------------------------

Built as an AI/ML portfolio and research project exploring
parameter-efficient fine-tuning for FinOps analysis.

