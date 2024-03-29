### Current progress:

- General
	- [x] Random color distortion (from SimCLR)
	- [x] Random Gaussian blur (from SimCLR)
	- [x] Random multi-crop (from SwAV)
	- [x] InfoNCE loss
	- [ ] Momentum encoder
	- [ ] Selective Kernel (SK) convolution layer (used in SimCLR v2)
	- [x] LARS optimizer
	- [x] Consine annealing with linear warmup scheduler
	- [x] Linear scheduler (for torch<=1.9)
	- [x] CSV logger
	- [x] TensorBoard logger
	- [x] Checkpoint saving and restoring 
	- [x] Command line parameter parser
	- [x] YAML parameter parser
	- [x] PyTorch workflow encapsulation
	- [x] Dedicated evaluation script
	- [ ] Detailed readme file
	- [ ] Data Parallel
	- [ ] Distributed Data Parallel
	- [ ] Global batch normalization
	- [ ] Shuffling batch normalization

- Supervised baseline
	- [x] ResNet
	- [ ] ViT
	- [x] CIFAR-10
	- [x] CIFAR-100
	- [x] ImageNet-1k

- Self-supervised baseline
	- [x] SimCLR (contrastive, ResNet)
	- [ ] MoCo v2 (contrastive, ResNet)
	- [ ] MoCo v3 (contrastive, ViT)
	- [ ] BYOL (non-contrastive, ResNet)
	- [ ] DINO (self-distillation, ViT)

To be continued ...
