import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=16, help="dimensionality of the latent space")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--n_classes", type=int, default=91)
parser.add_argument("--val_batch_size", type=int, default=128, help="size of the batches")
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--num", type=float, default=2, help="number of nan data")
parser.add_argument('--model_path', type=str, default='C:/data/model/')  # Model Tmp Save
parser.add_argument('--sample_path', type=str, default='C:/data/results/')
parser.add_argument('--train_data_path', type=str, default='train')
parser.add_argument('--validation_data_path', type=str, default='validation')
parser.add_argument('--test_data_path', type=str, default='test')
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")

args = parser.parse_args()
