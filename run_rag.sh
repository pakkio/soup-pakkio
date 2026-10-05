#!/bin/bash
cd ~/w/soup-demo
. ../Soup/.venv/bin/activate
set -a; . ../.env; set +a
export TORCH_DISABLE_NATIVE_JIT=1
printf '%s\n' \
  'Can you tell me what Soup is and what it is for?' \
  'What is Soup?' \
  'Why would I use Soup instead of doing it by hand?' \
  'What is the command to start training with Soup?' \
  'How can I get a GGUF out of my Soup-trained model?' \
  'Which Python versions work with Soup?' \
  'How do I serve my model with an OpenAI-compatible API?' \
  'What is layer streaming?' \
  'What is the capital of Italy?' \
| python -W ignore rag_demo.py 2>&1 | grep -v -E 'Loading weights|^\s*$|torch_dtype'
