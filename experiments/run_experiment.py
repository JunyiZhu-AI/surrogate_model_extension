import click
import json
from sme import attack

@click.command()
@click.option("--path_to_data", default="./data")
@click.option("--path_to_res", default="./res")
@click.option("--dataset", type=click.Choice(["CIFAR100", "FEMNIST"]), default="CIFAR100")
# federated learning parameters
@click.option("--model", type=click.Choice(["LeNet", "MLP", "CNNcifar", "CNNmnist", "ResNet8", "ViT"]), default="CNNcifar")
@click.option("--batchsize", default=10)
@click.option("--train_lr", default=0.01)
@click.option("--k", default=10)
@click.option("--epochs", default=20)
# reconstruction attack parameters
@click.option("--eta", default=1e-3, help="Step size of reconstruction.")
@click.option("--beta", default=1e-3, help="Step size of alpha.")
@click.option("--alpha", default=0., help="Interpolation factor.")
@click.option("--iters", default=5000, help="Optimization iterations of reconstruction.")
@click.option("--test_steps", default=500, help="Measure the psnr and save figs every so many steps.")
@click.option("--lamb", default=1e-4, help="Total variation coefficient.")
@click.option("--lr_decay", default=True, help="Use learning rate decay.")
@click.option("--seed", default=0)
@click.option(
    "--config", help="Path to the configuration file.", default=None,
)
def main(**kwargs):
    if kwargs["config"]:
        with open(kwargs["config"]) as f:
            kwargs = json.load(f)
    else:
        del kwargs["config"]

    print(kwargs)
    attack(**kwargs)


if __name__ == "__main__":
    main()

