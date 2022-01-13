import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

import software.utils as utils
from software.losses import MSELoss
from software.models.models_ivimnet import IVIMNET
from software.data import *
from software.experiments.ivimnet.trainer import IVIMTrainer
import tqdm
import copy
import software.experiments.ivimnet.ivim.simulations as sim
import torch
import torch.utils.data as tutils
import argparse
from datetime import timedelta
import logging


parser = argparse.ArgumentParser("ivimnet")
parser.add_argument('--model', type=str, default='ivimnet', help='the model that we want to train')
parser.add_argument(
    "--learning_rate", type=float, default=3e-5, help="initial learning rate"
)
parser.add_argument("--smoothing", type=float, default=0.0, help="smoothing factor")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
parser.add_argument("--clip", type=float, default=0.0, help="dropout probability")
parser.add_argument("--p", type=float, default=0.1, help="dropout probability")
parser.add_argument("--data", type=str, default="", help="location of the data corpus")
parser.add_argument("--dataset", type=str, default="ivimnet", help="dataset")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument(
    "--dataset_size", type=float, default=1.0, help="portion of the whole training data"
)
parser.add_argument(
    "--valid_portion", type=float, default=0.1, help="portion of training data"
)
parser.add_argument("--epochs", type=int, default=200, help="num of training epochs")
parser.add_argument(
    "--input_size", nargs="+", default=[1, 3, 32, 32], help="input size"
)
parser.add_argument("--output_size", type=int, default=10, help="output size")
parser.add_argument("--samples", type=int, default=10, help="output size")
parser.add_argument("--save", type=str, default="EXP", help="experiment name")
parser.add_argument(
    "--save_last", action="store_true", help="whether to just save the last model"
)
parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument(
    "--debug", action="store_true", help="whether we are currently debugging"
)
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device ids")
parser.add_argument(
    "--q", action="store_true", help="whether to do post training quantisation"
)


def make_model(should_load, bvalues, args):
    model = IVIMNET(bvalues, args)
    if should_load:
        utils.load_model(model, args.save + "/weights.pt")

    logging.info(f"## Model {'re-' if should_load else ''}created: ##")
    logging.info(model.__repr__())
    logging.info("### Loading model to parallel GPUs ###")
    model = utils.model_to_gpus(model, args)
    return model


def normalize(X_train, bvalues, bref=0):
    try:
        S0 = np.mean(X_train[:, bvalues == bref], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        np.delete(X_train, np.isnan(np.mean(X_train, axis=1)), axis=0)
    except:
        S0 = torch.mean(X_train[:, bvalues == bref], axis=1)
        X_train = X_train / S0[:, None]
        np.delete(X_train, np.isnan(torch.mean(X_train, axis=1)), axis=0)
    return X_train


def sim_signals(SNR, bvalues):
    # Parameters taken from hp_example_1() in IVIMNET repo.
    range = ([0.0005, 0.05, 0.01], [0.003, 0.55, 0.1])
    X_train, *rest = sim.sim_signal(SNR, bvalues, sims=int(1e6), rician=False,
                                    Dmin=range[0][0], Dmax=range[1][0],
                                    fmin=range[0][1], fmax=range[1][1],
                                    Dsmin=range[0][2], Dsmax=range[1][2])
    X_train = normalize(X_train, bvalues)
    return X_train, *rest


def train(model, bvalues, SNR, args, writer):
    logging.log("## Simulating signals ##")
    X_train, D, f, Dp = sim_signals(SNR, bvalues)

    logging.log("## Preparing training data ##")
    _split = int(np.floor(len(X_train) * 0.9))
    train_set, val_set = tutils.random_split(torch.from_numpy(X_train.astype(np.float32)),
                                             [_split, len(X_train) - _split])

    _tbs = 128; _vbs = min(len(val_set), 32 * _tbs)
    train_loader = tutils.DataLoader(train_set, batch_size=_tbs, shuffle=True, drop_last=True)
    valid_loader = tutils.DataLoader(val_set, batch_size=_vbs, shuffle=False, drop_last=True)

    logging.info("### Preparing schedulers and optimizers ###")
    criterion = MSELoss(args)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad()],
                                 args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20.0)
    scheduler = None

    logging.info("## Beginning Training ##")
    train = IVIMTrainer(model, criterion, optimizer, scheduler, args)
    best_error, train_time, val_time = train.train_loop(train_loader, valid_loader, writer)

    logging.info(
        f"## Finished training, the best observed validation error: {best_error}, "
        f"total training time: {timedelta(seconds=train_time)}, "
        f"total validation time: {timedelta(seconds=val_time)} ##"
    )

    logging.info("## Beginning Evaluating ##")
    del model
    args.samples = 100


def sim_signal_predict(bvalues, SNR):
    """ Copy paste from the IVIMNET/simulations.py file with minor modifications and removed plots. """
    def fit_ivim(bvalues, Dt, Fp, Dp, S0):
        # regular IVIM function
        return (S0 * (Fp * np.exp(-bvalues * Dp) + (1 - Fp) * np.exp(-bvalues * Dt)))

    rg = np.random.RandomState(123)
    ## define parameter values in the three regions
    S0_region0, S0_region1, S0_region2 = 1, 1, 1
    Dp_region0, Dp_region1, Dp_region2 = 0.03, 0.05, 0.07
    Dt_region0, Dt_region1, Dt_region2 = 0.0020, 0.0015, 0.0010
    Fp_region0, Fp_region1, Fp_region2 = 0.15, 0.3, 0.45
    # image size
    sx, sy, sb = 100, 100, len(bvalues)
    # create image
    dwi_image = np.zeros((sx, sy, sb))
    Dp_truth = np.zeros((sx, sy))
    Dt_truth = np.zeros((sx, sy))
    Fp_truth = np.zeros((sx, sy))

    # fill image with simulated values
    for i in range(sx):
        for j in range(sy):
            if (40 < i < 60) and (40 < j < 60):
                # region 0
                dwi_image[i, j, :] = fit_ivim(bvalues, Dt_region0, Fp_region0, Dp_region0, S0_region0)
                Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region0, Dt_region0, Fp_region0
            elif (20 < i < 80) and (20 < j < 80):
                # region 1
                dwi_image[i, j, :] = fit_ivim(bvalues, Dt_region1, Fp_region1, Dp_region1, S0_region1)
                Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region1, Dt_region1, Fp_region1
            else:
                # region 2
                dwi_image[i, j, :] = fit_ivim(bvalues, Dt_region2, Fp_region2, Dp_region2, S0_region2)
                Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region2, Dt_region2, Fp_region2
    
    # Initialise dwi noise image
    dwi_noise_imag = np.zeros((sx, sy, sb))
    # fill dwi noise image with Gaussian noise
    for i in range(sx):
        for j in range(sy):
            dwi_noise_imag[i, j, :] = rg.normal(0, 1 / SNR, (1, len(bvalues)))
    # Add Gaussian noise to dwi image
    dwi_image_noise = dwi_image + dwi_noise_imag
    # normalise signal
    S0_dwi_noisy = np.mean(dwi_image_noise[:, :, bvalues == 0], axis=2)
    dwi_image_noise_norm = dwi_image_noise / S0_dwi_noisy[:, :, None]

    dwi_image_long = np.reshape(dwi_image_noise_norm, (sx * sy, sb))
    return dwi_image_long, Dt_truth, Fp_truth, Dp_truth


def infer(model, bvalues, SNR, args, writer):
    dwi_image, Dt_truth, Fp_truth, Dp_truth = sim_signal_predict(bvalues, SNR)

    ## normalise the signal to b=0 and remove data with nans
    dwi_image = normalize(dwi_image, bvalues)
    mylist = np.isnan(np.mean(dwi_image, axis=1))    # TODO: Very descriptive name. Change it.
    sels = ~mylist
    # remove data with non-IVIM-like behaviour. Estimating IVIM parameters in these data is meaningless anyways.
    sels = sels & (np.percentile(dwi_image[:, bvalues <  50], 0.95, axis=1) < 1.3) & (
                   np.percentile(dwi_image[:, bvalues >  50], 0.95, axis=1) < 1.2) & (
                   np.percentile(dwi_image[:, bvalues > 150], 0.95, axis=1) < 1.0)
    # we need this for later
    lend = len(dwi_image)
    dwi_image = dwi_image[sels]

    # initialise parameters and data
    Dp = np.array([])
    Dt = np.array([])
    Fp = np.array([])
    S0 = np.array([])

    # initialise dataloader. Batch size can be way larger as we are still training.
    inferloader = utils.DataLoader(torch.from_numpy(dwi_image.astype(np.float32)),
                                   batch_size=2056,
                                   shuffle=False,
                                   drop_last=False)

    # start predicting
    with torch.no_grad():
        model.eval()
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            X_batch = X_batch.to('cuda')
            # here the signal is predicted. Note that we now are interested in the parameters and no longer in the predicted signal decay.
            _, Dtt, Fpt, Dpt, S0t = model(X_batch)
            S0 = np.append(S0, (S0t.cpu()).numpy())
            Dt = np.append(Dt, (Dtt.cpu()).numpy())
            Fp = np.append(Fp, (Fpt.cpu()).numpy())
            Dp = np.append(Dp, (Dpt.cpu()).numpy())

    if np.mean(Dp) < np.mean(Dt):
        Dp22 = copy.deepcopy(Dt)
        Dt = copy.deepcopy(Dp)
        Dp = copy.deepcopy(Dp22)
        Fp = 1 - Fp    # here we correct for the data that initially was removed as it did not have IVIM behaviour, by returning zero

    # estimates
    Dptrue = np.zeros(lend)
    Dttrue = np.zeros(lend)
    Fptrue = np.zeros(lend)
    S0true = np.zeros(lend)
    Dptrue[sels] = Dp
    Dttrue[sels] = Dt
    Fptrue[sels] = Fp
    S0true[sels] = S0
    del inferloader
    torch.cuda.empty_cache()    # TODO: Everywhere I'm just assuming we use cuda. Alright for PoC.

    return Dttrue, Fptrue, Dptrue, S0true

def main():
    args = parser.parse_args()
    load = False
    if args.save.upper() != "EXP":
        load = True
    args, writer = utils.parse_args(args)

    logging.info('# Start Re-training #')

    SNR = 20
    bvalues = np.array([0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700]) # array of b-values
    model = make_model(load, bvalues, args)
    if not load:
        train(model, bvalues, SNR, args, writer)

    Dttrue, Fptrue, Dptrue, S0true = infer(model, bvalues, SNR, args, writer)


if __name__ == "__main__":
    main()
