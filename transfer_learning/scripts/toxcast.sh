#### GIN fine-tuning
split=scaffold
python finetune.py \
        --input_model_file ./checkpoints/pretrained.pth \
       	--split $split \
	      --batch_size 32 \
	      --runseed 0 1 2 3 4 5 6 7 8 9 \
        --dataset "toxcast"  \
	      --coarse_layer 2 \
	      --coarse_rate 0.8\
	      --epochs 50\
	      --device 0 \

# --lr 1e-3 --epochs 100
