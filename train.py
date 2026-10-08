import argparse
import torch
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm 
from torchvision.utils import save_image

from utils.utils import *
from utils.model import *



def parse_argument():

    parser = argparse.ArgumentParser()

    # Content dataset
    parser.add_argument(
        '--content_dir',
        type=str,
        default=r"C:\Users\rajub\OneDrive\Documents\Desktop\NST_PROJECT\NST_P\content_data",
        help="Location of content dataset"
    )

    # Style dataset
    parser.add_argument(
        '--style_dir',
        type=str,
        default=r"C:\Users\rajub\OneDrive\Documents\Desktop\NST_PROJECT\NST_P\style_data",
        help="Location of style dataset"
    )

    # Pre-trained VGG model
    parser.add_argument(
        '--vgg',
        type=str,
        default=r"C:\Users\rajub\OneDrive\Documents\Desktop\NST_PROJECT\NST_P\vgg_normalised.pth",
        help="Location of trained VGG model"
    )

    # Experiment name
    parser.add_argument(
        '--experiment',
        type=str,
        default='experiment',
        help="Name of experiment"
    )

    # Final image size
    parser.add_argument(
        '--final_size',
        type=int,
        default=256,
        help="Size of the final image"
    )

    # Style image size
    parser.add_argument(
        '--style_size',
        type=int,
        default=256,
        help="Size of style image"
    )

    # Content image size
    parser.add_argument(
        '--content_size',
        type=int,
        default=256,
        help="Size of content image"
    )

    # Batch size
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help="Batch size"
    )

    # Random crop
    parser.add_argument(
        '--crop',
        action='store_true',
        help="Crop image"
    )

    # Learning rate
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help="Learning rate"
    )

    # Learning rate decay
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=5e-5,
        help="Learning rate decay"
    )
    # start epoch
    parser.add_argument(
    '--start_epoch',
    type=int,
    default=0,
    help="Starting epoch"
)

    # epochs
    parser.add_argument('--epochs', type=int, default=2, help="Epoch value")

    # content weight ke liye 
    parser.add_argument('--content_weight', type=int, default=1, help="content weight")

    # stle weight ke liye

    parser.add_argument('--style_weight', type=int, default=5, help="style weight")

    # bhai log interval dhena bhull gya tha so error aya tha
    parser.add_argument('--log_interval', type=int, default=10, help="log interval")

    # save_interval
    parser.add_argument('--save_interval', type=int, default=10, help="Save interval")

    # decoder path 
    parser.add_argument('--decoder_path', type=str, default=None, help="Decoder path")

    # resumr 
    parser.add_argument('--resume', action='store_true', default=False, help="Resume  ")

    # optimizer path

    parser.add_argument('--optimizer_path', type=str,default=None, help="optimizer path" )

    return parser.parse_args()


def main():

    # Get command-line arguments
    args = parse_argument()

    # Select GPU if available, otherwise CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Create experiment directory
    save_dir = Path('experiment') / args.experiment

    save_dir.mkdir(
        exist_ok=True,
        parents=True
    )

    # Save argument values
    with open(save_dir / 'args.txt', 'w') as args_file:

        for key, value in vars(args).items():
            args_file.write(f'{key} : {value}\n')

    # Create content transformation
    content_transform = get_transform(
        args.content_size,
        args.crop,
        args.final_size
    )

    # Create style transformation
    style_transform = get_transform(
        args.style_size,
        args.crop,
        args.final_size
    )

    # Create content dataset
    content_dataset = ImageFolderDataset(
        args.content_dir,
        content_transform
    )

    # Create style dataset
    style_dataset = ImageFolderDataset(
        args.style_dir,
        style_transform
    )

    # Create content DataLoader
    content_dataloader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # Create style DataLoader
    style_dataloader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    # Print dataset information
    print("Content dataset size:", len(content_dataset))
    print("Style dataset size:", len(style_dataset))

    print("Content batches:", len(content_dataloader))
    print("Style batches:", len(style_dataloader))

    # Check batch shape
    for batch in content_dataloader:
        print(batch.shape)
        break

    # Create encoder
    encoder = VGGEncoder(args.vgg).to(device)

    # Create decoder
    decoder = Decoder().to(device)

    # Optimizer
    optimizer = optim.Adam(
        decoder.parameters(),
        lr=args.lr
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch:
            1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        decoder.load_state_dict(
                torch.load(args.decoder_path, map_location=device)
        )

        optimizer.load_state_dict(
                torch.load(args.optimizer_path, map_location=device)
        )

    print("Resuming from epoch:", args.start_epoch)

    print("training....")

    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    running_loss = None
    running_closs=None
    running_sloss=None

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        progress_bar = tqdm(zip(content_dataloader, style_dataloader), total=min(len(content_dataloader), len(style_dataloader))) # dekh hm ne jab noram ANN train karte hh tb usme kewal tran_loader par kare thh

        running_closs=0
        running_loss=0
        running_sloss=0


        for content_batch, style_batch in progress_bar:

            # before passing the encoder that should be on one device 
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            # print(len(c_feats))
            # print(len(s_feats))

            # print(c_feats[0].shape)

            t= adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            g=decoder(t)

            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight
            loss_s=0

            for g_f, s_f in zip(g_feats,s_feats):

                g_mean, g_std = cal_mean_std(g_f)
                s_mean, s_std = cal_mean_std(s_f)

                loss_s+=mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)


            loss_s = loss_s * args.style_weight

            loss = loss_c + loss_s

            # now optimizer

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'{loss.item():4f}, content loss: {loss_c.item():4f}, style loss:{loss_s.item():4f}')


            running_sloss+=loss_s.item()
            running_loss+=loss.item()
            running_closs+=loss_c.item()

        scheduler.step()

        running_sloss/=len(content_dataloader)
        running_loss/=len(content_dataloader)
        running_closs = len(content_dataloader)

        if(epoch+1) % args.log_interval ==0:
            tqdm.write(f'Iter {epoch+1}: loss:{running_loss:4f}, content loss: {running_closs:4f}, style loss: {running_sloss:4f}')

        if(epoch +1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir/ f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir/ f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir/ f'output_{epoch+1}.png', nrow=args.batch_size)


            



        




if __name__ == '__main__':
    main()