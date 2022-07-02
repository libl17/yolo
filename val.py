"""
Validate a trained model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640
"""

import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy)
from utils.metrics import ConfusionMatrix, ap_per_class, ap_per_class_grid, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import time_sync


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--grid-thres', type=float, default=0.001, help='grid threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--channel', default=3, help='image channel')
    parser.add_argument('--fusion', default=3, help='box sand grid fusion model')
    parser.add_argument('--grid-confusion', action='store_true', help='grid confusion matrix')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@torch.no_grad()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        grid_thres=0.001,  # grid threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='test',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        augment=False,  # augmented inference
        project='runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        compute_loss=None,
        channel = 3,  # image channel
        fusion=False,  # box sand grid fusion model
        grid_confusion=False,  # grid confusion matrix
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
        model.half() if half else model.float()
    else:  # called directly
        device = torch.device('cuda:0')

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, fp16=half)
        stride = model.stride
        half = model.fp16  # FP16 supported on limited backends with CUDA
        device = model.device

        # Datasets
        with open(data, errors='ignore') as f:
            data= yaml.safe_load(f)  # dictionary
        task_path = Path(data.get('path'))
        for k in 'train', 'val', 'test':
            data[k] = str(task_path / data[k])

    # Configure
    model.eval()
    nc = int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    stats, ap, ap_class = [], [], []
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    loss = torch.zeros(4, device=device) if fusion else torch.zeros(3, device=device)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

    # Grid configure
    nc_grid = int(data['nc_grid'])
    confusion = np.zeros([nc_grid + 1, nc_grid + 1])
    gridiouv = torch.tensor([0]).to(device)
    gridstats = []
    names_grid = {k: v for k, v in enumerate(model.names_grid if hasattr(model, 'names_grid') else model.module.names_grid)}
    names_grid_bg = {k: v for k, v in enumerate(model.names_grid if hasattr(model, 'names_grid') else model.module.names_grid)}
    names_grid_bg[-1] = 'Background'

    # Dataloader
    if not training:
        model.warmup(imgsz=(1, channel, imgsz, imgsz))  # warmup
        task = task if task in ('train', 'val', 'test') else 'test'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       rect=True,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    # pbar
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    for mini_dataloader in dataloader:
        pbar = tqdm(mini_dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        
        # Batch
        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            t1 = time_sync()
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
            if fusion:
                isbox = targets[:, 1] >= 0
                isgrid = targets[:, 1] < 0
                targetsneg = targets[isgrid]
                targetspos = targets[isbox]
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            _, _, height, width = im.shape  # batch size, channels, height, width
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
            dt[1] += time_sync() - t2
            if fusion:
                gridout = out[0]
                out = out[1]

            # Loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

            # NMS
            targetspos[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
            t3 = time_sync()
            out = non_max_suppression(out, conf_thres, iou_thres, multi_label=True)
            dt[2] += time_sync() - t3

            # Grid ConfusionMatrix
            if not training and fusion and grid_confusion:
                for si, pred in enumerate(gridout):
                    # Predition grid class
                    labelsneg = targetsneg[targetsneg[:, 0] == si, 1:].to('cpu')
                    isneg = pred[:, :].max(2)[0] < grid_thres
                    predcls = pred[:, :].max(2)[1].view(pred.shape[0],pred.shape[1],1).to('cpu') + 1
                    predcls[isneg] = 0

                    # Target grid class
                    targetscls = torch.full_like(predcls, 0, device='cpu')  # targets
                    for ele in labelsneg:
                        y = (ele[2]*predcls.shape[0]).int()
                        x = (ele[1]*predcls.shape[1]).int()
                        elecls = ele[0].abs().int()
                        targetscls[y, x] = elecls

                    # Build grid ConfusionMatrix
                    gridy, gridx, _ = pred.shape
                    for ii in range(gridy):
                        for jj in range(gridx):
                            confusion[targetscls[ii, jj, 0], predcls[ii, jj, 0]] += 1

            # Metrics of box
            for si, pred in enumerate(out):
                labels = targetspos[targetspos[:, 0] == si, 1:] if fusion else targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                shape = shapes[si][0]
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
                seen += 1

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((3, 0), device=device)))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Metrics of grid
            if fusion:
                for si, pred in enumerate(gridout):
                    gridy, gridx, clsnum = pred.shape
                    labelsneg = targetsneg[targetsneg[:, 0] == si, 1:]
                    predcls = pred[:, :].max(2)[1].view(pred.shape[0],pred.shape[1],1) + 1
                    predconf = pred[:, :].max(2)[0].view(pred.shape[0],pred.shape[1],1)
                    nl = len(labelsneg)
                    tcls = labelsneg[:, 0].abs().tolist() if nl else []  # target class
                    path, shape = Path(paths[si]), shapes[si][0]

                    targetscls = torch.full_like(predcls, 0, device=gridiouv.device)  # targets
                    for ele in labelsneg:
                        x = (ele[1]*predcls.shape[1]).int()
                        y = (ele[2]*predcls.shape[0]).int()
                        elecls = ele[0].abs().int()
                        targetscls[y, x] = elecls

                    if len(pred) == 0:
                        if nl:
                            gridstats.append((torch.zeros(0, 1, dtype=torch.bool), torch.Tensor(), torch.Tensor(), targetscls.flatten().cpu()))
                        continue

                    # Evaluate
                    if nl:
                        correct = (predcls.flatten() == targetscls.flatten())
                    else:
                        correct = torch.zeros(gridx*gridy, 1, dtype=torch.bool)
                    gridstats.append((correct.view(correct.shape[0], 1).cpu(), predconf.flatten().cpu(), predcls.flatten().cpu(), targetscls.flatten().cpu()))  # (correct, conf, pcls, tcls)

            # Plot images
            if plots and batch_i < 3:
                if fusion:
                    plot_images(im, targetspos, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
                else:
                    plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
                plot_images(im, output_to_target(out), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

    # Compute metrics of grid
    if fusion:
        if not os.path.exists(save_dir / 'grid'):
            os.makedirs(save_dir / 'grid') 
        gridstats = [np.concatenate(x, 0) for x in zip(*gridstats)]  # to numpy
        if len(gridstats) and gridstats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class_grid(*gridstats, plot=plots, save_dir=save_dir / 'grid', names=names_grid)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp_grid, mr_grid, map50_grid, map_grid = p.mean(), r.mean(), ap50.mean(), ap.mean()
            map50_crm = ap50[1:].mean()
            crack_ap = ap50[1:-2].mean()
            nt_grid = np.bincount(gridstats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt_grid = torch.zeros(1)
    
    # Compute metrics of box
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    if fusion:
        LOGGER.info(pf % ('Grid', seen, nt_grid.sum(), mp_grid, mr_grid, map50_crm, crack_ap))
    LOGGER.info(pf % ('Box', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if not training and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print grid confusion matrix
    if not training and fusion and grid_confusion:
        # first line
        mat_line = ('Matrix',)
        for mat_j in range(nc_grid + 1):
            mat_line = mat_line + (names_grid_bg[-1 + mat_j],)
        mat_line = mat_line + ('Sum', 'Recall')
        LOGGER.info(('%20s' * (nc_grid + 3))% mat_line)
        # middle line
        for mat_i in range(nc_grid + 1):
            mat_line = (names_grid_bg[-1 + mat_i],)
            for mat_j in range(nc_grid + 1):
                mat_line = mat_line + (confusion[mat_i,mat_j],)
            mat_line = mat_line + (sum(confusion[0,:]), confusion[0,0]/sum(confusion[0,:]))
            LOGGER.info(('%20s' + '%20.6g' * (nc_grid + 3)) % mat_line)
        # sublast line
        mat_line = ('Sum',)
        for mat_j in range(nc_grid + 1):
            mat_line = mat_line + (sum(confusion[:,mat_j]),)
        mat_line = mat_line + ('Sum', 'Recall')
        LOGGER.info(('%20s' * (nc_grid + 1))% mat_line)
        # last line
        mat_line = ('Precision', 1 if sum(confusion[:,0]) == 0 else confusion[0,0]/sum(confusion[:,0]))
        for mat_j in range(nc_grid):
            mat_line = mat_line + (confusion[1 + mat_j,1 + mat_j]/sum(confusion[:,1 + mat_j]),)
        LOGGER.info(('%20s' * (nc_grid + 1))% mat_line)
        
    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, channel, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    
    # Save grid results
    if not training and fusion:
        gridouttxt = save_dir / 'grid' / 'grid_info.txt'
        with open(gridouttxt, 'w') as f:
            f.write(f'grid_thres={grid_thres}\n')
            f.write(f'weights={weights}\n')

    # Return results
    model.float()  # for training
    if not training:
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp_grid, mr_grid, map50_crm, crack_ap, mp, mr, map50, map, *(loss.cpu() / sum([len(mini) for mini in dataloader])).tolist())


def main(opt):
    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
