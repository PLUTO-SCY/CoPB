import sys
from utils import *
import os
from DLModel.denoising_diffusion_pytorch import *
from torch.utils.tensorboard import SummaryWriter
from DLModel.dataprepare import *
import warnings
import argparse
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10001)  
    parser.add_argument("--expIndex", type=int, default=999)
    parser.add_argument("--diffusionstep", type=int, default=100)
    parser.add_argument("--denoise", type=str,
                        default='add')
    parser.add_argument("--dataset", type=str, default='foursquare')

    parser.add_argument("--train_batch_size", type=int, default=500)
    parser.add_argument("--train_lr", type=float, default=8e-5)
    parser.add_argument("--save_and_sample_every", type=int, default=2000)
    parser.add_argument("--testSample_batch", type=int, default=4096)
    args = parser.parse_args()

    args_dict = vars(args)

    with open(f'config/exp{args.expIndex}_config.json', 'w') as f:
        json.dump(args_dict, f, indent=4)

    xdataset, numCategory = get_dataset(repeat=30, dataset = args.dataset)

    writer = SummaryWriter(
        log_dir='TensorBoardLogs/exp{}'.format(args.expIndex))  

    d_input = 2  # channel
    d_model = args.modeldim  # Lattent dim
    d_output = d_input  # From dataset
    d_emb = 24*6
    N = 4  # Number of encoder and decoder to stack  
    dropout = 0.1  # Dropout rate
    pe = 'original'  # Positional encoding
    device = torch.device("cuda")

    if args.denoise == 'add':
        dinoisingModel = TransformerDenoisingAdd(
            d_input=d_input,
            d_model=d_model,
            d_output=d_output,
            d_emb=d_emb,  # Embedding size for the additional input variable
            N=N,
            numCategory=numCategory,
            dropout=dropout,
            pe=pe).to(device)
        
    elif args.denoise == 'queryonce':
        dinoisingModel = TransformerDenoisingQueryOnce(  
            d_input=d_input,
            d_model=d_model,
            d_output=d_output,
            d_emb=d_emb,
            N=N,
            numCategory=numCategory,
            dropout=dropout,
            pe=pe).to(device)
    elif args.denoise == 'querymulti':
        dinoisingModel = TransformerDenoisingQueryMulti( 
            d_input=d_input,
            d_model=d_model,
            d_output=d_output,
            d_emb=d_emb,
            N=N,
            numCategory=numCategory,
            dropout=dropout,
            pe=pe).to(device)
        
    elif args.denoise == 'unet':
        print('unet1D')
        dinoisingModel = Unet1D(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = 2,
        )
    elif args.denoise == 'unet_condition':
        dinoisingModel = Unet1D_condition(
            dim = 64, 
            dim_mults = (1, 2, 4, 8),
            channels = 2,
            numCategory=numCategory,
        )
    else:
        print('Wrong denoise network type')  
    

    diffusion = GaussianDiffusion1D(
        dinoisingModel,
        seq_length=24*6,
        timesteps=args.diffusionstep,
        objective='pred_v'
    )

    total_params = count_parameters(diffusion)
    print(f"模型的总参数量: {total_params / 1e6:.2f}M")

    # after a lot of training
    categories = torch.tensor(
        xdataset.processed_categories_init, dtype=torch.long).to(device)
    print('total categories.shape: ', categories.shape)  # (13,144)
    torch.save(categories, f'data/{args.dataset}_condition.pt')


    trainer = Trainer1D(
        diffusion,
        dataset=xdataset,
        train_batch_size=args.train_batch_size,  
        train_lr=args.train_lr, 
        train_num_steps=args.epochs,         # total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                       # turn on mixed precision
        tbwriter=writer,
        save_and_sample_every=args.save_and_sample_every,  
        evalCondition=categories,
        testSample_batch=args.testSample_batch,   
        expIndex=args.expIndex,
    )
    trainer.train()

    sampled_seq = diffusion.sample(categories, batch_size=categories.shape[0])
    print(sampled_seq.shape) 

    torch.save(sampled_seq, f'../Generate_results/samples_{args.expIndex}.pt')
    print(f"Tensor saved")

    eval()


