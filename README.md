# MLX_W4_DD_Transfer_Learning
A home for code for the 'Dropout Disco' team, and their code throughout the Week 4 project on Transfer Learning


## Init a new GPU:
Assuming your ssh config is saved as `computa`, run this from the project root:
`scp ./run_on_gpu.sh computa:~`
Then ssh into `computa` and run the script. 

## Preprocess Flickr Dataset: 
Run 
`uv run .\model\dataset.py --preprocess`

After the data set is created, upload to huggingface using
`uv run .\model\dataset.py --upload <repo_name>`