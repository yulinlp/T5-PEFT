source /home/lin/sdk/T5_PEFT/bin/activate

python src/eval.py --config 'config/t5-base/'

python src/eval.py --config 'config/gpt2-small/'

python src/eval.py --config 'config/flan-t5-base/'