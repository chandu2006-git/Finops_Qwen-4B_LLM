
# Deployment Limitations

## 1. Current Deployment Status

FinOps LLM has been tested in a GPU-enabled Kaggle environment using the Qwen3-4B-Instruct-2507 base model and the fine-tuned LoRA adapter.

The model and adapter load successfully, and inference has been demonstrated on cloud cost analysis and rightsizing scenarios.

A persistent public inference API or live application has not yet been deployed.

## 2. Compute Requirements

The base model contains approximately 4 billion parameters. Although 4-bit quantization reduces memory requirements, inference still requires memory for model weights, intermediate activations, and the KV cache.

The current inference implementation expects a CUDA-compatible GPU. CPU-only hosting is not supported by the current configuration without modifications and additional performance testing.

## 3. Model Hosting

The trained LoRA adapter is published separately on Hugging Face:

- **Base model:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **LoRA adapter:** [Chandu06/finops-qwen3-4b](https://huggingface.co/Chandu06/finops-qwen3-4b)

The adapter is not a standalone language model. Inference requires loading the compatible base model and attaching the adapter.

Publishing model files does not automatically provide a running inference API.

## 4. Hosting and Cost Constraints

Kaggle has been used for GPU-based development and inference testing. Its notebook runtime is not intended to serve as a permanent public inference service.

A persistent deployment requires a compatible inference host or managed inference provider. GPU hosting may incur costs even when the service is idle.

As this is a student project without a dedicated GPU budget, development has prioritized reproducible inference, source-code publication, and technical documentation over always-on hosting.

Free hosting options may impose compute limits, usage quotas, sleep behavior, or hardware restrictions. No permanent free GPU inference endpoint has been verified for this project.

## 5. Model Reliability

Initial inference tests identified limitations in generated responses:

- Required report sections may be omitted.
- Low utilization can be interpreted incorrectly.
- Similar information may be repeated across sections.
- Correct output formatting does not guarantee correct reasoning.

The model's reported validation token accuracy should not be interpreted as real-world FinOps reasoning accuracy.

The model's performance has not yet been comprehensively compared against the unfine-tuned base model.

## 6. Financial Analysis Limitations

The model should not invent cloud pricing, configuration details, or savings estimates when evidence is missing.

Cost calculations should be validated deterministically whenever sufficient input values are available.

Recommendations remain advisory. Workload history, peak utilization, application requirements, and performance constraints should be reviewed before infrastructure changes are implemented.

## 7. Security and Operational Considerations

- API tokens and other secrets must not be committed to source control.
- Private model repositories require appropriate authentication.
- Inputs and generated outputs should be validated before being used in automated workflows.
- Latency, throughput, concurrency, and uptime have not been comprehensively tested.

## 8. Future Deployment Work

Planned improvements include:

1. Implementing input and output validation.
2. Adding deterministic cost calculations.
3. Improving domain restriction and report-format compliance.
4. Comparing the fine-tuned model against the base model.
5. Selecting a suitable inference host based on available resources and budget.
6. Deploying and testing a persistent API when suitable compute becomes available.

## Summary

FinOps LLM is a working model-inference prototype with a published LoRA adapter. Persistent public inference remains future work because it requires suitable hosting resources.

The project documents its current capabilities, limitations, and reproducible inference workflow without claiming production deployment.
