"""Build train.jsonl: Soup-specific Q&A (facts from README/docs) + alpaca-cleaned filler."""
import json
import random

from datasets import load_dataset

# (list of question phrasings, answer). Grounded in README.md and docs/commands.md.
FACTS = [
    (["What is Soup?", "What is Soup in the context of LLMs?", "Tell me about the Soup project.",
      "What does the Soup CLI do?", "Explain what soup-cli is."],
     "Soup is a CLI-first tool for fine-tuning and post-training LLMs. You write one YAML config "
     "and run one command, with no SSH and no config hell."),
    (["Who is Soup for?", "Why would I use Soup?", "What problem does Soup solve?"],
     "Soup removes the infrastructure pain of LLM fine-tuning: no SSH into GPU boxes, one simple "
     "YAML config, and automatic batch size, GPU detection and quantization. It works locally "
     "on your own GPU with QLoRA, so no cloud is required."),
    (["How do I install Soup?", "What is the pip command to install soup-cli?",
      "How can I install Soup with training support?"],
     "Install it with pip: `pip install \"soup-cli[train]\"`. The bare `soup-cli` package is the "
     "light CLI without PyTorch; add the `[train]` extra to fine-tune. You can also use pipx or "
     "`uv tool install soup-cli`."),
    (["Which Python versions does Soup support?", "What Python version do I need for Soup?"],
     "Soup supports Python 3.10 to 3.12 only. On 3.13 and newer, pip could resolve untested "
     "PyTorch wheels that crash before Soup runs."),
    (["What license is Soup under?", "Is Soup open source?"],
     "Soup is open source under the Apache-2.0 license."),
    (["How do I create a Soup config?", "How do I start a new Soup project?",
      "What does soup init do?"],
     "Run `soup init` for an interactive wizard, or `soup init --template chat` to start from a "
     "template. It writes a `soup.yaml` config file."),
    (["Which templates does soup init offer?", "What templates are available in Soup?"],
     "The templates are chat, code, tool-calling, medical, reasoning, vision, kto, orpo, simpo, "
     "ipo, bco, rlhf, pretrain, moe, longcontext, embedding and audio."),
    (["How do I train a model with Soup?", "What command starts fine-tuning in Soup?",
      "How do I run training?"],
     "Run `soup train --config soup.yaml`. LoRA, quantization and batching are handled for you."),
    (["How do I chat with my fine-tuned model in Soup?", "How can I test the trained model?",
      "What command lets me talk to my model?"],
     "Use `soup chat --model ./output` to talk to your fine-tuned model in the terminal."),
    (["How do I merge a LoRA adapter into the base model with Soup?",
      "What does soup merge do?"],
     "`soup merge --adapter ./output` merges the LoRA adapter into the base model."),
    (["How do I export a Soup model to GGUF?", "How do I use my Soup model with Ollama?",
      "What command exports to GGUF?"],
     "Run `soup export --model ./output --format gguf --quant q4_k_m` to get a GGUF for Ollama or "
     "llama.cpp. Add `--deploy ollama` to deploy it to Ollama automatically."),
    (["How do I upload my model to Hugging Face with Soup?", "What does soup push do?"],
     "`soup push --model ./output --repo user/name` uploads the model to the Hugging Face Hub."),
    (["How do I serve a model with Soup?", "Does Soup have an API server?"],
     "Yes. `soup serve --model ./output --port 8000` starts an OpenAI-compatible API server. You "
     "can pick a backend such as vllm or sglang with `--backend`."),
    (["What does a Soup config file look like?", "Show me a minimal soup.yaml.",
      "What are the main fields in soup.yaml?"],
     "A soup.yaml has `base` (the model), `task` (for example sft), a `data` section "
     "(train path, format, val_split), a `training` section (epochs, lr, batch_size, lora, "
     "quantization) and an `output` directory."),
    (["Where is the Soup config schema defined?", "What is the source of truth for Soup config fields?"],
     "The Pydantic v2 schema in `src/soup_cli/config/schema.py` is the single source of truth "
     "for every config field."),
    (["What happens if I mistype a key in soup.yaml?", "Does Soup ignore unknown config keys?"],
     "Since v0.75.0 an unknown config key refuses the load: the CLI exits with code 1 and names "
     "the field you probably meant. Earlier versions only warned and dropped it."),
    (["Can I fine-tune a big model on a small GPU with Soup?",
      "What is layer streaming in Soup?", "How does Soup fit an 8B model on a 4 GB GPU?"],
     "Layer streaming (beta, `stream_layers: true`) keeps the frozen base model out of VRAM and "
     "feeds it to the GPU one decoder layer at a time. Llama-3.1-8B with NF4 was measured at "
     "119.6 tokens/s with a 3.32 GB peak on an RTX 3050 Laptop 4 GB."),
    (["Does Soup have a web interface?", "What does soup ui do?"],
     "Yes. `soup ui` serves a local dashboard at http://127.0.0.1:7860 for experiments, training "
     "setup, live metrics, dataset exploration and model chat. Install it with "
     "`pip install \"soup-cli[ui]\"`."),
    (["How do I check or clean my dataset in Soup?", "What do the soup data commands do?"],
     "`soup data inspect` shows dataset stats, `soup data validate` checks the format, "
     "`soup data clean` removes control characters and empty turns, `soup data dedup` removes "
     "duplicates and `soup data convert` changes formats. They work on the light install."),
    (["Which training tasks does Soup support?", "What can Soup train besides SFT?"],
     "Besides SFT, Soup supports preference and RL methods such as DPO, KTO, ORPO, SimPO, IPO, "
     "BCO, RLHF, GRPO and PPO, plus pretraining, vision, audio and embedding tasks."),
    (["How do I resume training in Soup?", "Can Soup resume from a checkpoint?"],
     "Use `soup train --config soup.yaml --resume ./output/checkpoint-100` to resume from a local "
     "checkpoint."),
    (["How do I evaluate a model with Soup?", "What does soup eval do?"],
     "`soup eval` has several subcommands: `benchmark` for standard benchmarks, `custom` for your "
     "own tasks, `judge` for LLM-as-a-judge, `compare` to compare runs and `gate` for eval gates."),
    (["Which libraries are lazy-imported in Soup?", "What coding conventions does Soup follow?"],
     "Heavy dependencies (torch, transformers, peft, trl, mlx) are lazy-imported inside "
     "functions. Output goes through rich.console.Console, never bare print, and the line "
     "length is 100 with ruff rules E, F, I, N, W."),
    (["How do I run the Soup tests?", "How do I contribute to Soup and run its test suite?"],
     "Install with `pip install -e \".[dev]\"`, run `pytest tests/ -v --tb=short`, and make sure "
     "`ruff check src/soup_cli/ tests/` is clean before any commit."),
    (["Where can I find the Soup website?", "What is the Soup website?"],
     "The Soup website is trysoup.dev."),
    (["What is QLoRA in Soup?", "How does Soup use quantization?"],
     "Soup trains with LoRA on top of a quantized base model. Setting `quantization: 4bit` in the "
     "training section loads the base in 4-bit (QLoRA), which saves a lot of GPU memory."),
]

SYSTEM_FREE = "What is the capital of Italy?"  # sanity: not in training data

rng = random.Random(7)
soup_rows = []
for questions, answer in FACTS:
    for q in questions:
        soup_rows.extend([{"instruction": q, "input": "", "output": answer}] * 3)

ds = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=42)
n_filler = 150
filler = [
    {"instruction": r["instruction"], "input": r["input"], "output": r["output"]}
    for r in ds
    if len(r["instruction"] + r["input"] + r["output"]) < 1200
][:n_filler]

rows = soup_rows + filler
rng.shuffle(rows)
with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"soup={len(soup_rows)} filler={len(filler)} total={len(rows)}")
