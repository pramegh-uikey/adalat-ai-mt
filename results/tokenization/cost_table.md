| tokenizer | EN tok/judgment | HI tok/judgment | HI/EN premium | % test HI > 256 | cost/judgment (USD) |
|---|---|---|---|---|---|
| indicbert-v2 | 1853.79 | 2002.79 | 1.05 | 0.0 | 0.0247 |
| muril | 1990.71 | 2072.38 | 1.01 | 0.0 | 0.0257 |
| indictrans2 | 1986.54 | 2079.50 | 1.02 | 0.0 | 0.0258 |
| mbart-50 | 2138.00 | 2318.54 | 1.05 | 0.0 | 0.0285 |
| nllb-200 | 2147.92 | 2354.38 | 1.06 | 0.0 | 0.0289 |
| gemma-3 | 2030.83 | 2369.42 | 1.13 | 0.0 | 0.0288 |
| sarvam-1 | 2402.42 | 2476.08 | 1.00 | 0.0 | 0.0308 |
| o200k_base | 1957.42 | 2730.83 | 1.35 | 0.0 | 0.0322 |
| llama-3.2 | 1964.50 | 4436.46 | 2.20 | 0.0 | 0.0493 |
| qwen2.5 | 2078.92 | 7864.67 | 3.69 | 14.2 | 0.0838 |
| cl100k_base | 1965.21 | 8242.17 | 4.09 | 15.8 | 0.0873 |
| tinyllama(llama-2) | 2357.67 | 9052.29 | 3.74 | 20.0 | 0.0964 |

Prices are illustrative 2024 list prices (see `configs/tokenizers.yaml`), an ASSUMPTION not a verified live quote; costs ignore prompt/system-message overhead. Decoder latency scales with output length: HI tok/judgment = sequential decoding steps for an autoregressive LLM generating the Hindi translation.
