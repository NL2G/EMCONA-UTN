python 2bgen.py -m "deepseek/deepseek-r1-distill-llama-70b" -f "outputs_all/filtered/cross.tsv" --out_dir "outputs_all/generated/" --method "vivid language"
python 2bgen.py -m "qwen/qwen3-32b" -f "outputs_all/filtered/cross.tsv" --out_dir "outputs_all/generated/" --method "vivid language"
python 2bgen.py -m "openai/o3-mini" -f "outputs_all/filtered/cross.tsv" --out_dir "outputs_all/generated/" --method "vivid language"

python 2bgen.py -m "deepseek/deepseek-r1-distill-llama-70b" -f "outputs_all/filtered/cross.tsv" --out_dir "outputs_all/generated/"
python 2bgen.py -m "openai/gpt-4o-mini" -f "outputs_all/filtered/cross.tsv" --out_dir "outputs_all/generated/"
python 2bgen.py -m "qwen/qwen3-32b" -f "outputs_all/filtered/cross.tsv" --out_dir "outputs_all/generated/"
python 2bgen.py -m "meta-llama/llama-3.3-70b-instruct" -f "outputs_all/filtered/cross.tsv" --out_dir "outputs_all/generated/"