import argparse
import torch
import time
from model import SCEModel, ResNet34
from dataset import cifarDataset
from tqdm import tqdm
from utils.utils import AverageMeter, accuracy, count_parameters_in_MB
from torch.optim.lr_scheduler import MultiStepLR
from train_util import TrainUtil
from loss import SCELoss

# ArgParse
parser = argparse.ArgumentParser(description='SCE Loss')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--l2_reg', type=float, default=5e-3)
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--train_log_every', type=int, default=50)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_path', default='data', type=str)
parser.add_argument('--checkpoint_path', default='checkpoints', type=str)
parser.add_argument('--data_nums_workers', type=int, default=8)
parser.add_argument('--epoch', type=int, default=120)
parser.add_argument('--nr', type=float, default=0.4, help='noise_rate')
parser.add_argument('--loss', type=str, default='SCE', help='SCE, CE')
parser.add_argument('--version', type=str, default='SCE0.0', help='Version')
parser.add_argument('--train_cifar100', action='store_true', default=False)

args = parser.parse_args()
GLOBAL_STEP, EVAL_STEP, EVAL_BEST_ACC, EVAL_BEST_ACC_TOP5 = 0, 0, 0, 0
cell_arc = None

for arg in vars(args):
    print(arg, getattr(args, arg))

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    print("Using CUDA!")
else:
    device = torch.device('cpu')


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        display += '\t' + str(key) + '=%.5f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display


def model_eval(epoch, fixed_cnn, data_loader):
    global EVAL_STEP
    fixed_cnn.eval()
    valid_loss_meters = AverageMeter()
    valid_acc_meters = AverageMeter()
    valid_acc5_meters = AverageMeter()
    ce_loss = torch.nn.CrossEntropyLoss()

    for images, labels in tqdm(data_loader["test_dataset"]):
        start = time.time()
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            pred = fixed_cnn(images)
            loss = ce_loss(pred, labels)
            acc, acc5 = accuracy(pred, labels, topk=(1, 5))

        valid_loss_meters.update(loss.item())
        valid_acc_meters.update(acc.item())
        valid_acc5_meters.update(acc5.item())
        end = time.time()

        EVAL_STEP += 1
        if EVAL_STEP % args.train_log_every == 0:
            display = log_display(epoch=epoch,
                                  global_step=GLOBAL_STEP,
                                  time_elapse=end-start,
                                  loss=loss.item(),
                                  test_loss_avg=valid_loss_meters.avg,
                                  acc=acc.item(),
                                  test_acc_avg=valid_acc_meters.avg,
                                  test_acc_top5_avg=valid_acc5_meters.avg)
            tqdm.write(display)
    display = log_display(epoch=epoch,
                          global_step=GLOBAL_STEP,
                          time_elapse=end-start,
                          loss=loss.item(),
                          test_loss_avg=valid_loss_meters.avg,
                          acc=acc.item(),
                          test_acc_avg=valid_acc_meters.avg,
                          test_acc_top5_avg=valid_acc5_meters.avg)
    tqdm.write(display)
    return valid_acc_meters.avg, valid_acc5_meters.avg


def train_fixed(starting_epoch, data_loader, fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler, utilHelper):
    global GLOBAL_STEP, reduction_arc, cell_arc, EVAL_BEST_ACC, EVAL_STEP, EVAL_BEST_ACC_TOP5

    for epoch in tqdm(range(starting_epoch, args.epoch)):
        tqdm.write("=" * 20 + "Training" + "=" * 20)
        fixed_cnn.train()
        train_loss_meters = AverageMeter()
        train_acc_meters = AverageMeter()
        train_acc5_meters = AverageMeter()

        for images, labels in tqdm(data_loader["train_dataset"]):
            images, labels = images.to(device), labels.to(device)
            start = time.time()
            fixed_cnn.zero_grad()
            fixed_cnn_optmizer.zero_grad()
            pred = fixed_cnn(images)
            loss = criterion(pred, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(fixed_cnn.parameters(), args.grad_bound)
            fixed_cnn_optmizer.step()

            acc, acc5 = accuracy(pred, labels, topk=(1, 5))
            acc_sum = torch.sum((torch.max(pred, 1)[1] == labels).type(torch.float))
            total = pred.shape[0]
            acc = acc_sum / total

            train_loss_meters.update(loss.item())
            train_acc_meters.update(acc.item())
            train_acc5_meters.update(acc5.item())

            end = time.time()

            GLOBAL_STEP += 1
            if GLOBAL_STEP % args.train_log_every == 0:
                lr = fixed_cnn_optmizer.param_groups[0]['lr']
                display = log_display(epoch=epoch,
                                      global_step=GLOBAL_STEP,
                                      time_elapse=end-start,
                                      loss=loss.item(),
                                      loss_avg=train_loss_meters.avg,
                                      acc=acc.item(),
                                      acc_top1_avg=train_acc_meters.avg,
                                      acc_top5_avg=train_acc5_meters.avg,
                                      lr=lr,
                                      gn=grad_norm)
                tqdm.write(display)
        fixed_cnn_scheduler.step()
        tqdm.write("="*20 + "Eval" + "="*20)
        curr_acc, curr_acc5 = model_eval(epoch, fixed_cnn, data_loader)
        print("curr_acc\t%.4f" % curr_acc)
        print("BEST_ACC\t%.4f" % EVAL_BEST_ACC)
        print("curr_acc_top5\t%.4f" % curr_acc5)
        print("BEST_ACC_top5\t%.4f" % EVAL_BEST_ACC_TOP5)
        payload = '=' * 10 + '\n'
        payload = payload + ("curr_acc: %.4f\n best_acc: %.4f\n" % (curr_acc, EVAL_BEST_ACC))
        payload = payload + ("curr_acc_top5: %.4f\n best_acc_top5: %.4f\n" % (curr_acc5, EVAL_BEST_ACC_TOP5))
        EVAL_BEST_ACC = max(curr_acc, EVAL_BEST_ACC)
        EVAL_BEST_ACC_TOP5 = max(curr_acc5, EVAL_BEST_ACC_TOP5)
        tqdm.write("Model Saved!\n")
    return


def train():
    global GLOBAL_STEP, reduction_arc, cell_arc
    # Dataset
    dataset = cifarDataset(batchSize=args.batch_size,
                           dataPath=args.data_path,
                           numOfWorkers=args.data_nums_workers,
                           noise_rate=args.nr,
                           is_cifar100=args.train_cifar100)
    dataLoader = dataset.getDataLoader()

    if args.train_cifar100:
        num_classes = 100
        fixed_cnn = ResNet34(num_classes=num_classes)
    else:
        num_classes = 10
        fixed_cnn = SCEModel()

    if args.loss == 'SCE':
        if args.train_cifar100:
            criterion = SCELoss(alpha=6.0, beta=0.1, num_classes=num_classes)
        else:
            criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=num_classes)
    elif args.loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        print("Unknown loss")

    print(criterion.__class__.__name__)
    print("Number of Trainable Parameters %.4f" % count_parameters_in_MB(fixed_cnn))
    fixed_cnn = torch.nn.DataParallel(fixed_cnn)
    fixed_cnn.to(device)

    fixed_cnn_optmizer = torch.optim.SGD(params=fixed_cnn.parameters(),
                                         lr=args.lr,
                                         momentum=0.9,
                                         nesterov=True,
                                         weight_decay=args.l2_reg)

    if args.train_cifar100:
        milestone = [80, 120]
    else:
        milestone = [40, 80]
    fixed_cnn_scheduler = MultiStepLR(fixed_cnn_optmizer, milestone, gamma=0.1)

    utilHelper = TrainUtil(checkpoint_path=args.checkpoint_path, version=args.version)
    starting_epoch = 0
    train_fixed(starting_epoch, dataLoader, fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler, utilHelper)


if __name__ == '__main__':
    train()
