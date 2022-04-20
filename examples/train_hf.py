from fastai.vision.all import *
from fastai.basics import *
from upit.models.cyclegan import *
from upit.train.cyclegan import *
from upit.data.unpaired import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='horse2zebra')
    parser.add_argument('--dataset_name', type=str, default='huggan/horse2zebra')
    parser.add_argument('--model_name', type=str, default='cyclegan', choices=['cyclegan', 'dualgan','ganilla'])
    parser.add_argument('--fieldA', type=str, default='imageA', help='Name of the column for domain A in dataset')
    parser.add_argument('--fieldB', type=str, default='imageB', help='Name of the column for domain B in dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--load_size', type=int, default=286, help='Load size')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size')
    parser.add_argument('--epochs_flat', type=int, default=100, help='Number of epochs with flat LR')
    parser.add_argument('--epochs_decay', type=int, default=100, help='Number of epochs with linear decay of LR')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    dls = get_dls_from_hf(args.dataset_name, fieldA=args.fieldA, fieldB=args.fieldB, load_size=args.load_size, crop_size=args.crop_size, bs=args.batch_size)
    if args.model_name == 'cyclegan': 
        cycle_gan = CycleGAN()
        learn = cycle_learner(dls, cycle_gan, opt_func=partial(Adam,mom=0.5,sqr_mom=0.999))
    elif args.model_name == 'dualgan':
        dual_gan = DualGAN()
        learn = dual_learner(dls, dual_gan, opt_func=partial(Adam,mom=0.5,sqr_mom=0.999))
    elif args.model_name == 'ganilla':
        ganilla = GANILLA()
        learn = cycle_learner(dls, ganilla, opt_func=partial(Adam,mom=0.5,sqr_mom=0.999))
    learn.fit_flat_lin(args.epochs_flat, args.epochs_decay, args.lr)
    learn.save(args.experiment_name+'_'+args.model_name+'_'+str(args.batch_size)+'_'+str(args.epochs_flat)+'_'+str(args.epochs_decay)+'_'+str(args.lr))
    learn.model.push_to_hub(args.experiment_name+'_'+args.model_name)
