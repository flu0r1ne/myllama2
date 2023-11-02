MyLlama2
========

This is a ground-up reimplementation of the Llama 2 family of large language models.
It adheres to the exact same architecture, based on a decoder-only transformer model
equipped with Group-Query Attention (GQA), key-value caching, SwiGLU feedforward layers,
and SentencePiece embeddings. It is compatible with the original Llama 2 weights.
Unlike Meta's model, this implementation does not incorporate parallel layers or any
distributed processing APIs. Consequently, it can only run on a single GPU and is also
capable of running on a CPU without the need for special tools (e.g., `torchrun`, `mpi`, etc.).

This model was created for demonstration purposes, with the intent of sharing it with the
community. During its development, I identified a few minor issues in FAIR's
implementation, which I plan to contribute back through pull requests. I believe this
implementation is more accessible for those new to AI, and I've included references to the papers
where these concepts were first introduced. However, this code has not been extensively reviewed.
For production projects, I recommend starting with [Meta's implementation](https://github.com/facebookresearch/llama).
For high-performance CPU-only inference, consider compiling
[llama.cpp](https://github.com/ggerganov/llama.cpp) while targeting the native architecture.

Example usage:
--------------

```bash
python inference_example.py \
	llama/llama-2-7b \
	./tokenizer.model \
	--top_p 0.8 \
	--max_generation_length 100 \
	--context "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal."
```
