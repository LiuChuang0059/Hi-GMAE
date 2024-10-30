#### GIN fine-tuning
split=scaffold
for rate in 0.1
do
python finetune.py \
       	--input_model_file "checkpoints/_gin_60.pth" \
	--split $split \
	--batch_size 256 \
	--runseed 0 1 2\
	--lr 0.001 \
       	--dataset "qm9" \
	--coarse_layer 2 \
	--coarse_rate $rate\
	--epochs 100\
	--device 0
done
# --lr 1e-3 --epochs 100
