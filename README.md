# DiffusionGemma 26B NVFP4 on NVIDIA DGX Spark: 100+ tok/s with vLLM

This repository is not a code project. It is a practical note from my NVIDIA DGX Spark setup, written in the hope that it helps other people bring up `nvidia/diffusiongemma-26B-A4B-it-NVFP4` with vLLM. In direct vLLM API testing, the model exceeded 100 tok/s for single-request generation with thinking disabled.

Hardware note: this was tested on NVIDIA DGX Spark, reported by `nvidia-smi` as NVIDIA GB10.

I tested the model through vLLM's OpenAI-compatible API and have also been using it through a Hermes AI Agent Telegram gateway. So far, the model has been working without major issues in that agent setup.

Subjectively, compared with running `Qwen3.6-35B-A3B-NVFP4` on vLLM in my environment, DiffusionGemma 26B A4B NVFP4 feels more than 50% faster in day-to-day agent/chat usage. This is a subjective impression, not a controlled apples-to-apples benchmark, but it is noticeable enough that I wanted to mention it.

## Environment

```text
Model: nvidia/diffusiongemma-26B-A4B-it-NVFP4
GPU: NVIDIA DGX Spark (NVIDIA GB10)
vLLM version: 0.22.1rc1.dev357+g74b5964f0
Container image: vllm/vllm-openai:gemma
API endpoint: http://localhost:8000/v1
Serving mode: vLLM OpenAI-compatible API
Tensor parallel: 1
Max model length: 100,000
```

## Launch Command

This was the final vLLM command that worked well for me:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype auto \
  --max-model-len 100000 \
  --gpu-memory-utilization 0.70 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --diffusion-config '{"canvas_length": 256, "max_denoising_steps": 48}' \
  --attention-backend TRITON_ATTN \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --override-generation-config '{"max_new_tokens": null}' \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  -tp 1
```

For benchmark requests, I explicitly disabled thinking per request:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

The server default above keeps thinking enabled because I also wanted to test agent-style reasoning behavior. For clean throughput numbers, disabling thinking per request is important.

## Confirmed vLLM Runtime Details

From the vLLM logs:

```text
Resolved architecture: DiffusionGemmaForBlockDiffusion
dtype: bfloat16
quantization: modelopt_fp4
KV cache dtype: fp8_e4m3
Attention backend: TRITON_ATTN
MoE backend: FLASHINFER_CUTLASS
Chunked prefill: enabled
Prefix caching: enabled
Async scheduling: enabled
enforce_eager: False
Compilation mode: VLLM_COMPILE
CUDA graph mode: FULL_AND_PIECEWISE
```

Loading result:

```text
Model loading memory: 18.16 GiB
Model loading time: 99.5 sec
GPU KV cache size: 3,530,532 tokens
Maximum concurrency for 100,000-token requests: 35.31x
Graph capturing: completed
```

## Benchmark Method

The benchmark below was measured directly against the vLLM OpenAI-compatible API.

```text
Path: direct vLLM API
Excluded: Telegram gateway, Hermes agent loop, tool calls
Request mode: non-streaming
Temperature: 0
max_tokens: 512 for long-answer tests
Thinking: disabled per request
Measurement: client-side wall clock
Token count source: OpenAI API usage.completion_tokens
Warmup: excluded
```

## Benchmark Results

```text
Single request, sequential x4
Output tokens: 1,349 total
Mean latency: 3.33s
Median latency: 3.30s
Mean speed: 101.26 tok/s
Median speed: 101.07 tok/s
```

```text
Concurrency=4, parallel x4
Output tokens: 1,382 total
Wall time: 9.34s
Aggregate speed: 148.00 tok/s
Per-request mean speed: 41.36 tok/s
```

```text
Short-answer sequential x4
Output tokens: 375 total
Mean latency: 1.42s
Median latency: 1.43s
Mean speed: 65.99 tok/s
Median speed: 65.21 tok/s
```

In short:

```text
On NVIDIA DGX Spark (NVIDIA GB10), DiffusionGemma 26B A4B NVFP4 with vLLM reached about
101 tok/s for single-request generation and about 148 tok/s aggregate
at concurrency=4, with thinking disabled for benchmark requests.
```

## Hermes AI Agent Usage

I have also been trying this model through Hermes AI Agent as a Telegram gateway model. So far, it has worked without major issues for normal agent/chat usage.

One important observation: measuring speed through a gateway or agent loop can be misleading. A Telegram/Hermes request may include long conversation history, tool calls, retries, hidden reasoning tokens, and message delivery overhead. In one gateway test, the visible answer looked short, but the API reported far more output tokens, which strongly suggested that thinking/reasoning tokens were included.

For fair model throughput testing, I recommend measuring direct vLLM API calls with `enable_thinking=false`.

## What Failed Initially

The first attempts were not smooth. The main problems were:

1. Treating the model like a normal autoregressive vLLM model.

   DiffusionGemma needs the newer vLLM DiffusionGemma support path. Using the NVIDIA/Hugging Face recommended `vllm/vllm-openai:gemma` image fixed this.

2. Missing `/workspace` inside the container.

   My local launcher tried to copy an execution script into `/workspace`, but the official image did not have that directory. Creating `/workspace` inside the container before copying the script fixed the launch failure.

3. Running with `--enforce-eager`.

   I initially used eager mode for conservative debugging, but it prevented the faster compile/CUDA graph path. Removing `--enforce-eager` allowed vLLM compile and CUDA graph capture to work. After that, the logs showed `enforce_eager=False`, `VLLM_COMPILE`, and `CUDAGraphMode.FULL_AND_PIECEWISE`.

4. Too-conservative batching.

   Raising `max_num_batched_tokens` from 4096 to 8192 helped batching/chunked prefill behavior. The final setting was:

   ```text
   --max-num-batched-tokens 8192
   --max-num-seqs 4
   ```

5. Over-aggressive GPU memory utilization.

   I tried higher values, but settled on `--gpu-memory-utilization 0.70` for stability. Even at 0.70, the KV cache was large enough for my use case.

## Changes That Helped Performance

The biggest improvements came from:

- Updating to a vLLM build that supports this DiffusionGemma path.
- Using the recommended `vllm/vllm-openai:gemma` image.
- Setting `VLLM_USE_V2_MODEL_RUNNER=1`.
- Removing `--enforce-eager`.
- Confirming CUDA graph capture was active.
- Using `--attention-backend TRITON_ATTN`.
- Increasing `--max-num-batched-tokens` to 8192.
- Disabling thinking for benchmark requests.

## Notes and Cautions

- You likely need a recent or upgraded vLLM build. Older vLLM versions may not load this model correctly.
- Use the `vllm/vllm-openai:gemma` image or another vLLM build with DiffusionGemma support.
- Make sure `VLLM_USE_V2_MODEL_RUNNER=1` is set.
- Do not use `--enforce-eager` if you want the faster compile/CUDA graph path.
- Tool calling requires the Gemma parser flags:

  ```bash
  --enable-auto-tool-choice
  --tool-call-parser gemma4
  --reasoning-parser gemma4
  ```

- If you benchmark through an agent gateway, the result may include thinking tokens, tool loop overhead, long history prefill, and messaging latency.
- For clean throughput tests, call vLLM directly and pass:

  ```json
  {"chat_template_kwargs": {"enable_thinking": false}}
  ```

## Summary

This setup made DiffusionGemma 26B A4B NVFP4 practical for me on NVIDIA DGX Spark (NVIDIA GB10). The model loaded successfully with vLLM, served through the OpenAI-compatible API, worked with Hermes AI Agent in everyday use, and delivered strong direct-API throughput when thinking was disabled.

