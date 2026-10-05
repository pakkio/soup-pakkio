#!/bin/bash
cd ~/w/soup-demo
while pgrep -f 'bin/soup train' >/dev/null; do sleep 5; done
tail -c 3000 train2.log | tr '\r' '\n' | grep -E 'Loss|Duration|Error'
. ../Soup/.venv/bin/activate
set -a; . ../.env; set +a
export TORCH_DISABLE_NATIVE_JIT=1
for m in output_alpaca output; do
  echo
  echo "=== $m ==="
  printf '%s\n' \
    'Can you tell me what Soup is and what it is for?' \
    'What is the command to start training with Soup?' \
    'How can I get a GGUF out of my Soup-trained model?' \
    'Which Python versions work with Soup?' \
    'What is the capital of Italy?' \
    '/quit' \
  | soup chat --model "$m" --max-tokens 100 --temperature 0.2 2>&1 \
  | tr '\r' '\n' | sed -n '/Model loaded/,$p' | grep -v -E '^\s*$|Model loaded|  /quit|  /clear|  /system|Goodbye'
done
