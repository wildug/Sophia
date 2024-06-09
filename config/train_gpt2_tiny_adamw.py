wandb_log = True
wandb_project = 'NanoGPT'
wandb_run_name='gpt2-small-adamW-10k'

# these make the total batch size be ~0.5M
# 8 batch size * 1024 block size * 6 gradaccum * 10 GPUs = 491,520
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 6
total_bs = 480 # way to large in practice

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False

# this makes total number of tokens be 300B
max_iters = 10000
lr_decay_iters = 10000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10
always_save_checkpoint = False  # only save best checkpoint

# optimizer
optimizer_name = 'adamw'
learning_rate = 1e-3 # max learning rate
weight_decay = 2e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 200 # how many steps to warm up for
min_lr = 1.5e-5 

compile = False

out_dir = 'out_small_adam_10k'
