CUDA_AVAILABLE_DEVICES=1 python search.py --model deit_small_patch16_224 --batch-size 64 --data-set IMNET --data-path /home/aamer/data/ImageNet --output_dir /home/aamer/repos/ViT-Slim/ViT-Slim/frozen_logs --pretrained_path /home/aamer/weights/deit-small-patch16-224/deit_small_patch16_224-cd65a155.pth --w1 2e-4 --w2 5e-5 --w3 1e-4


# Retrain
python retrain.py --model deit_small_patch16_224 --batch-size 64 --data-set IMNET --data-path /home/aamer/data/ImageNet --output_dir /home/aamer/repos/ViT-Slim/ViT-Slim/logs --searched_path /home/aamer/repos/ViT-Slim/ViT-Slim/logs/checkpoint.pth --budget_attn 0.7 --budget_mlp 0.7 --budget_patch 0.7
python eval.py --model deit_small_patch16_224 --batch-size 64 --data-set IMNET --data-path /home/aamer/data/ImageNet --output_dir /home/aamer/repos/ViT-Slim/ViT-Slim/logs --searched_path /home/aamer/weights/deit-small-patch16-224/deit_small_weights.pth --budget_attn 1 --budget_mlp 1 --budget_patch 1