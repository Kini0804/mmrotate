_base_ = ['../rotated_reppoints/rotated_reppoints_hrsc_FPN.py']

model = dict(
    bbox_head=dict(
        type='SAMRepPointsHead',
        loss_bbox_init=dict(type='BCConvexGIoULoss', loss_weight=0.375)),

    # training and testing settings
    train_cfg=dict(
        refine=dict(
            _delete_=True,
            assigner=dict(type='SASAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)))
evaluation = dict(
    save_best='auto', interval=5, dynamic_intervals=[(85, 1)], metric='mAP')