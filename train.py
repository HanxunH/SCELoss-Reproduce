import argparse
import torch
import time
import torchvision
from model import SCEModel
from dataset import cifarDataset
from tqdm import tqdm
from utils.utils import AverageMeter, accuracy, count_parameters_in_MB
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from train_util import TrainUtil
from loss import SCELoss
from radam import RAdam

# ArgParse
parser = argparse.ArgumentParser(description='MCTS_NAS')
# Arc
parser.add_argument('--arc_cell_depth', type=int, default=8)
parser.add_argument('--arc_out_filiters', type=int, default=36)
parser.add_argument('--arc_keep_prob', type=float, default=0.8)
parser.add_argument('--arc_num_of_layers', type=int, default=18)
parser.add_argument('--arc_lr', type=float, default=0.01)
parser.add_argument('--arc_l2_reg', type=float, default=5e-3)
parser.add_argument('--arc_grad_bound', type=float, default=5.0)
parser.add_argument('--arc_train_log_every', type=int, default=50)
parser.add_argument('--arc_drop_path_prob', type=float, default=0.2)
parser.add_argument('--arc_num_reduce', type=int, default=2)

# Training
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--train_imagenet', action='store_true', default=False)
parser.add_argument('--use_cutout', action='store_true', default=False)
parser.add_argument('--cutout_length', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_path', default='data', type=str)
parser.add_argument('--checkpoint_path', default='checkpoints', type=str)
parser.add_argument('--data_nums_workers', type=int, default=8)
parser.add_argument('--epoch', type=int, default=120)
parser.add_argument('--fixed_version', type=str, default='sceloss')
parser.add_argument('--search_arc_version', type=str, default='mcts_nas_v2_normal')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--nr', type=float, default=0.4, help='noise_rate')
parser.add_argument('--loss', type=str, default='SCE', help='SCE or CE')


args = parser.parse_args()
tb_writer = SummaryWriter(log_dir='tb_logs/'+args.fixed_version, comment=args.fixed_version)
GLOBAL_STEP, EVAL_STEP, EVAL_BEST_ACC, EVAL_BEST_ACC_TOP5 = 0, 0, 0, 0
cell_arc = None

for arg in vars(args):
    print(arg, getattr(args, arg))

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
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
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    for images, labels in tqdm(data_loader["test_dataset"]):
        start = time.time()
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            pred = fixed_cnn(images)
            loss = criterion(pred, labels)
            acc, acc5 = accuracy(pred, labels, topk=(1, 5))

        valid_loss_meters.update(loss.item())
        valid_acc_meters.update(acc)
        valid_acc5_meters.update(acc5)
        end = time.time()
        tb_writer.add_scalar('TestCNN/Eval/loss', loss.item(), EVAL_STEP)
        tb_writer.add_scalar('TestCNN/Eval/acc_top1', acc, EVAL_STEP)
        tb_writer.add_scalar('TestCNN/Eval/acc_top5', acc5, EVAL_STEP)

        EVAL_STEP += 1
        if EVAL_STEP % args.arc_train_log_every == 0:
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
        fixed_cnn.drop_path_prob = args.arc_drop_path_prob * epoch / args.epoch
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
            grad_norm = torch.nn.utils.clip_grad_norm_(fixed_cnn.parameters(), args.arc_grad_bound)
            fixed_cnn_optmizer.step()

            acc, acc5 = accuracy(pred, labels, topk=(1, 5))
            train_loss_meters.update(loss.item())
            train_acc_meters.update(acc)
            train_acc5_meters.update(acc5)

            end = time.time()
            tb_writer.add_scalar('FixedCNN/Training/loss', loss.item(), GLOBAL_STEP)
            tb_writer.add_scalar('FixedCNN/Training/acc_top1', acc, GLOBAL_STEP)
            tb_writer.add_scalar('FixedCNN/Training/acc_top5', acc5, GLOBAL_STEP)

            GLOBAL_STEP += 1
            if GLOBAL_STEP % args.arc_train_log_every == 0:
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
        utilHelper.save_model_fixed(epoch=epoch,
                                    fixed_cnn=fixed_cnn,
                                    fixed_cnn_optmizer=fixed_cnn_optmizer,
                                    GLOBAL_STEP=GLOBAL_STEP,
                                    cell_arc=cell_arc,
                                    EVAL_STEP=EVAL_STEP,
                                    save_best=curr_acc < EVAL_BEST_ACC,
                                    EVAL_BEST_ACC_TOP5=EVAL_BEST_ACC_TOP5)
        print("curr_acc\t%.4f" % curr_acc)
        print("BEST_ACC\t%.4f" % EVAL_BEST_ACC)
        print("curr_acc_top5\t%.4f" % curr_acc5)
        print("BEST_ACC_top5\t%.4f" % EVAL_BEST_ACC_TOP5)
        payload = '=' * 10 + '\n'
        payload = payload + ("curr_acc: %.4f\n best_acc: %.4f\n" % (curr_acc, EVAL_BEST_ACC))
        payload = payload + ("curr_acc_top5: %.4f\n best_acc_top5: %.4f\n" % (curr_acc5, EVAL_BEST_ACC_TOP5))
        tb_writer.add_text('Eval/best_arc', payload, global_step=epoch)
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
                           use_cutout=args.use_cutout,
                           cutout_length=args.cutout_length,
                           noise_rate=args.nr)
    dataLoader = dataset.getDataLoader()

    num_classes = 10
    # fixed_cnn = torchvision.models.resnet34(num_classes=num_classes)
    fixed_cnn = SCEModel()

    if args.loss == 'SCE':
        criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=num_classes)
    elif args.loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        print("Unknown loss")

    print("Number of Trainable Parameters %.4f" % count_parameters_in_MB(fixed_cnn))
    fixed_cnn = torch.nn.DataParallel(fixed_cnn)
    fixed_cnn.to(device)

    fixed_cnn_optmizer = torch.optim.SGD(params=fixed_cnn.parameters(),
                                          lr=args.arc_lr,
                                          momentum=0.9,
                                          nesterov=True,
                                          weight_decay=args.arc_l2_reg)

    # fixed_cnn_optmizer = RAdam(params=fixed_cnn.parameters(), lr=args.arc_lr, weight_decay=args.arc_l2_reg)

    fixed_cnn_scheduler = MultiStepLR(fixed_cnn_optmizer, [40, 80], gamma=0.1)

    # fixed_cnn_scheduler = CosineAnnealingLR(fixed_cnn_optmizer,
    #                                         float(args.epoch))

    utilHelper = TrainUtil(checkpoint_path=args.checkpoint_path,
                           version=args.fixed_version)
    starting_epoch = 0
    train_fixed(starting_epoch, dataLoader, fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler, utilHelper)


if __name__ == '__main__':
    train()
