CUDA_VISIBLE_DEVICES=7 python search.py --model deit_small_patch16_224 --batch-size 32 --data-set IMNET --data-path /home/aamer/data/ImageNet --pretrained_path /home/aamer/weights/deit_small_patch16_224-cd65a155.pth --pruning_strat
egy layer --freeze_weights

CUDA_VISIBLE_DEVICES=7 python search.py --model deit_small_patch16_224 --batch-size 32 --data-set IMNET --data-path /home/aamer/data/ImageNet --pretrained_path /home/aamer/weights/deit_small_patch16_224-cd65a155.pth --pruning_strategy none &

CUDA_VISIBLE_DEVICES=7 python search.py --model deit_small_patch16_224 --batch-size 32 --data-set IMNET --data-path /home/aamer/data/ImageNet --pretrained_path /home/aamer/weights/deit_small_patch16_224-cd65a155.pth --pruning_strategy layer & 

# CUDA_VISIBLE_DEVICES=7 python search.py --model deit_small_patch16_224 --batch-size 32 --data-set IMNET --data-path /home/aamer/data/ImageNet --pretrained_path /home/aamer/weights/deit_small_patch16_224-cd65a155.pth --pruning_strategy layer --freeze_weights &

# CUDA_VISIBLE_DEVICES=7 python search.py --model deit_small_patch16_224 --batch-size 32 --data-set IMNET --data-path /home/aamer/data/ImageNet --pretrained_path /home/aamer/weights/deit_small_patch16_224-cd65a155.pth --pruning_strategy node &

# CUDA_VISIBLE_DEVICES=7 python search.py --model deit_small_patch16_224 --batch-size 32 --data-set IMNET --data-path /home/aamer/data/ImageNet --pretrained_path /home/aamer/weights/deit_small_patch16_224-cd65a155.pth --pruning_strategy node --freeze_weights &


# # Retrain
# python retrain.py --model deit_small_patch16_224 --batch-size 256 --data-set IMNET --data-path /home/aamer/data/ImageNet --output_dir /home/aamer/repos/ViT-Slim/ViT-Slim/retrain_logs --searched_path /home/aamer/repos/ViT-Slim/ViT-Slim/logs/checkpoint.pth --budget_attn 0.7 --budget_mlp 0.7 --budget_patch 0.7
# python eval.py --model deit_small_patch16_224 --batch-size 64 --data-set IMNET --data-path /home/aamer/data/ImageNet --output_dir /home/aamer/repos/ViT-Slim/ViT-Slim/logs --searched_path /home/aamer/weights/deit-small-patch16-224/deit_small_weights.pth --budget_attn 1 --budget_mlp 1 --budget_patch 1


# # Eval
# CUDA_AVAILABLE_DEVICES=1  python retrain.py --model deit_small_patch16_224 --eval --batch-size 256 --data-set IMNET --data-path /home/aamer/data/ImageNet --output_dir /home/aamer/repos/ViT-Slim/ViT-Slim/retrain_logs --searched_path /home/aamer/repos/ViT-Slim/ViT-Slim/retrain_logs/checkpoint.pth --budget_attn 0.7 --budget_mlp 0.7 --budget_patch 0.7
