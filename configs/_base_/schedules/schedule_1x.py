# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)  # bs=4
# optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999))  # bs=4,weight_decay考虑调高(官方0.05)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # max_norm考虑调低，gpt建议1或5


# learning policy

# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2096,
    warmup_ratio=0.001,
    min_lr=1e-6)
total_epochs = 12

