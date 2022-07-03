"""
Train a YOLOv5 model on a custom dataset.

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (RECOMMENDED)
    $ python path/to/train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
"""

import os
import yaml
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

import val  # for end-of-epoch mAP
from utils.metrics import fitness
from utils.loss import ComputeLoss
from utils.torch_utils import EarlyStopping
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.plots import plot_labels, plot_images, plot_results
from utils.dataloaders import create_dataloader
from models.experimental import attempt_load
from utils.general import increment_path, LOGGER, colorstr, init_seeds, print_args, strip_optimizer, write_csv, intersect_dicts, labels_to_class_weights, labels_to_image_weights, get_latest_run


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yologrid.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/pavement.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.pavement.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--channel', default=3, help='image channel')
    parser.add_argument('--fusion', action='store_true', help='box sand grid fusion model')
    parser.add_argument('--multi-dataset', action='store_true', help='multiple datasets')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    opt = parser.parse_args()
    opt.noautoanchor = True
    opt.fusion = True
    opt.epochs = 100
    opt.cfg='models/yologrid.yaml'
    opt.multi_dataset = True
    opt.rect = True
    opt.data = 'data/pavement_rural.yaml'
    opt.imgsz = 640
    opt.weights = 's_version2_1c.pt'
    opt.image_weights = True
    opt.resume=True
    print_args(vars(opt))
    return opt


def main(
        weights='yolov5s.pt',  # initial weights path
        cfg='models/yolov5s.yaml',  # model.yaml path
        data='data/pavement.yaml',  # dataset.yaml path
        hyp='data/hyps/hyp.pavement.yaml',  # hyperparameters path
        epochs=100,
        batch_size=16,  # total batch size for all GPUs
        imgsz=640,  # train, val image size (pixels)
        rect=False,  # rectangular training
        nosave=False,  # only save final checkpoint
        noautoanchor=False,  # disable AutoAnchor
        noplots=False,  # save no plot files
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        project='runs/train',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        patience=100,  # EarlyStopping patience (epochs without improvement)
        channel=3,  # image channel
        fusion=False,  # box sand grid fusion model
        multi_dataset=False,  # multiple datasets
        image_weights=False,  # use weighted image selection for training
        resume=False,  # resume
    ):
    # Directories
    save_dir = increment_path(Path(project) / Path(cfg).stem / name, exist_ok=exist_ok)
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    
    # Hyperparameters
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    
    # Config
    init_seeds()
    device = torch.device('cuda:0')
    plots = not noplots

    # Datasets
    with open(data, errors='ignore') as f:
        data_dict = yaml.safe_load(f)  # dictionary
    path = Path(data_dict.get('path'))
    if multi_dataset:
        for k in 'train', 'val', 'test':
            data_dict[k] = [str(path / mini_task) for mini_task in data_dict[k]]
    else:
        for k in 'train', 'val', 'test':
            data_dict[k] = str(path / data_dict[k])
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = int(data_dict['nc'])  # number of classes
    names = data_dict['names']  # class names
    names_grid = data_dict['names_grid']

    # Model
    if weights.endswith('.pt'):
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        if resume:
            model = Model(ckpt['model'].yaml, ch=channel, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        else:
            model = Model(cfg or ckpt['model'].yaml, ch=channel, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=channel, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = True  # PyTorch Automatic Mixed Precision (AMP)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = hyp['lr0'],
                                 betas = (hyp['momentum'], 0.999)
                                 )
    
    # Scheduler
    lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if weights.endswith('.pt'):
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size,
                                              gs,
                                              hyp=hyp,
                                              augment=True,
                                              rect=rect,
                                              workers=workers,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              image_weights=image_weights
                                              )

    # Validationloader
    val_loader = create_dataloader(val_path,
                                   imgsz,
                                   batch_size * 2,
                                   gs,
                                   hyp=hyp,
                                   rect=True,
                                   workers=workers * 2,
                                   pad=0,
                                   prefix=colorstr('val: '),
                                   )[0]

    # Labels
    labels = np.vstack([np.concatenate(mini_dataset.labels, 0) for mini_dataset in dataset])
    if fusion:
        isbox = labels[:, 0] >= 0
        labels = labels[isbox]
    if plots and not resume:
        plot_labels(labels, names, save_dir)

    # Anchors
    if not noautoanchor:
        check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz, fusion = fusion)

    # Model attributes
    model.half().float()  # pre-reduce anchor precision
    nl = model.model[-3].nl if model.fusion else model.model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.names = names
    model.names_grid = names_grid
    model.class_weights = labels_to_class_weights(labels, nc).to(device) * nc  # attach class weights

    # Start training
    t0 = time.time()
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper = EarlyStopping(patience=patience)
    compute_loss = ComputeLoss(model, fusion=fusion)  # init loss class
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader[0].num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    
    # Epoch
    for epoch in range(start_epoch, epochs):
        model.train()

        mloss = torch.zeros(4, device=device) if fusion else torch.zeros(3, device=device)  # mean losses

        # pbar
        if fusion:
            LOGGER.info(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'grid', 'boxlbs', 'gridlbs', 'img_size'))
        else:
            LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        
        for loader_i, mini_train_loader in enumerate(train_loader):
            pbar = enumerate(mini_train_loader)
            nb = len(mini_train_loader)  # number of batches
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

            # Update image weights (optional, single-GPU only)
            if image_weights:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset[loader_i].labels, nc=nc, class_weights=cw, fusion=fusion)  # image weights
                dataset[loader_i].indices = random.choices(range(dataset[loader_i].n), weights=iw, k=dataset[loader_i].n)  # rand weighted idx
            
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
                if fusion:
                    isbox = targets[:, 1] >= 0
                    isgrid = targets[:, 1] < 0
                    targetsneg = targets[isgrid]
                    targetspos = targets[isbox]

                # Forward
                with torch.cuda.amp.autocast(amp):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                
                # Optimize
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Log
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # (GB)
                if fusion:
                    pbar.set_description(('%10s' * 2 + '%10.5g' * 7) %
                                        (f'{epoch}/{epochs - 1}', mem, *mloss, targetspos.shape[0], targetsneg.shape[0], imgs.shape[-1]))
                else:
                    pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                        (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                
                # runs on train batch end
                if plots and ni < 3:
                    if fusion:
                        plot_images(imgs, targetspos, paths, save_dir / f'train_batch_box_{loader_i}_{ni}.jpg')
                        plot_images(imgs, targetsneg, paths, save_dir / f'train_batch_grid_{loader_i}_{ni}.jpg')
                    else:
                        plot_images(imgs, targets, paths, save_dir / f'train_batch_{loader_i}_{ni}.jpg')
                # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        # mAP
        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
        results, maps = val.run(data_dict,
                        batch_size=batch_size * 2,
                        imgsz=imgsz,
                        model=model,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        plots=False,
                        compute_loss=compute_loss,
                        fusion=fusion
                        )

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi
        log_vals = list(mloss) + list(results) + lr

        # Save pocess
        write_csv(log_vals, save_dir, epoch, fusion)

        # Save model
        if (not nosave) or (final_epoch):
            ckpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'model': deepcopy(model).half(),
                'optimizer': optimizer.state_dict(),
            }
            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            del ckpt

        # Stop Single-GPU
        if stopper(epoch=epoch, fitness=fi):
            break
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    for f in last, best:
        if f.exists():
            strip_optimizer(f)  # strip optimizers
            if f is best:
                LOGGER.info(f'\nValidating {f}...')
                results, _ = val.run(
                        data_dict,
                        batch_size=batch_size * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.60,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        plots=plots,
                        compute_loss=compute_loss,
                        fusion=fusion
                    )  # val best model with plots

    # runs on training end
    if plots:
        plot_results(file=save_dir / 'results.csv', fusion = fusion)  # save results.png

    torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    opt = parse_opt()
    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.weights, opt.resume = ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')

    main(**vars(opt))
