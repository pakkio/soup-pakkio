#!/bin/bash
cd ~/w/soup-demo
. ../Soup/.venv/bin/activate
set -a; . ../.env; set +a
export TORCH_DISABLE_NATIVE_JIT=1
printf '%s\n' \
  'What is Soup?' \
  'How do I train a model with Soup?' \
  'How do I export a Soup model to GGUF?' \
  'Which Python versions does Soup support?' \
  '--- reworded ---' \
  'Can you tell me what Soup is and what it is for?' \
  'What is the command to start training with Soup?' \
  'How can I get a GGUF out of my Soup-trained model?' \
  'Which Python versions work with Soup?' \
  'What is the capital of Italy?' \
  'Write a haiku about GPUs.' \
  '/quit' \
| soup chat --model output_v2 --max-tokens 110 --temperature 0.2 2>&1 \
| tr '\r' '\n' | sed -n '/Model loaded/,$p' | grep -v -E '^\s*$|Model loaded|  /quit|  /clear|  /system|Goodbye'
