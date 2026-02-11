python run.py --agent-strategy tool-calling --env retail --model ollama/qwen3:8b --model-provider ollama --user-model ollama/llama3.1:8b --user-model-provider ollama --user-strategy llm --max-concurrency 1 --num-trials 5 --task-ids 0
sleep 30
python run.py --agent-strategy act --env retail --model ollama/qwen3:8b --model-provider ollama --user-model ollama/llama3.1:8b --user-model-provider ollama --user-strategy llm --max-concurrency 1 --num-trials 5 --task-ids 0
sleep 30
python run.py --agent-strategy react --env retail --model ollama/qwen3:8b --model-provider ollama --user-model ollama/llama3.1:8b --user-model-provider ollama --user-strategy llm --max-concurrency 1 --num-trials 5 --task-ids 0
sleep 30

python run.py --agent-strategy tool-calling --env airline --model ollama/qwen3:8b --model-provider ollama --user-model ollama/llama3.1:8b --user-model-provider ollama --user-strategy llm --max-concurrency 1 --num-trials 5 --task-ids 0
sleep 30
python run.py --agent-strategy act --env airline --model ollama/qwen3:8b --model-provider ollama --user-model ollama/llama3.1:8b --user-model-provider ollama --user-strategy llm --max-concurrency 1 --num-trials 5 --task-ids 0
sleep 30
python run.py --agent-strategy react --env airline --model ollama/qwen3:8b --model-provider ollama --user-model ollama/llama3.1:8b --user-model-provider ollama --user-strategy llm --max-concurrency 1 --num-trials 5 --task-ids 0
sleep 30