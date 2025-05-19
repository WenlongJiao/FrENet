import importlib
import torch
import torch.nn as nn
import torch.distributed as dist
import pprint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from utils.data_process import load_and_process_data
from utils.train_and_evaluate import *
from utils.option_util import load_config
from utils.util import set_seed
from utils.log_util import *


def main():
    config = load_config()
    args = config.training_args
    model_args = config.model_params

    set_seed(args.seed)

    is_distributed = dist.is_available()

    if is_distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0  # 对于单卡，使用rank 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = os.path.join(args.experiment_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_log_dir = os.path.join(args.experiment_dir, 'tensorboard_log')
    log_dir = os.path.join(tensorboard_log_dir, f"{args.model}_rank{local_rank}_seed{args.seed}_{current_time}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    config_log_message = "\n--- Training Arguments ---"  # Start with a newline for separation
    config_log_message += f"\n{pprint.pformat(vars(args), indent=2)}"
    config_log_message += "\n--- Model Parameters ---"
    config_log_message += f"\n{pprint.pformat(vars(model_args), indent=2)}"
    config_log_message += "\n--------------------------"
    logger.info(config_log_message)

    writer.add_text("TrainingArguments", pprint.pformat(vars(args), indent=2))
    writer.add_text("ModelParameters", pprint.pformat(vars(model_args), indent=2))

    model_module_name = f'models.{args.model.lower()}'
    try:
        model_module = importlib.import_module(model_module_name)
        model_class = getattr(model_module, args.model)  # Get class by name specified in config
        logger.info(f"Successfully imported model '{args.model}' from '{model_module_name}'.")
    except ModuleNotFoundError:
        logger.error(f"Model module '{model_module_name}.py' not found.")
        raise
    except AttributeError:
        logger.error(f"Model class '{args.model}' not found in module '{model_module_name}.py'.")
        raise
    except Exception as e:
        logger.error(f"Error importing model: {e}")
        raise

    logger.info(f"Loading data.")
    args.crop_size = tuple(args.crop_size) if isinstance(args.crop_size, list) else args.crop_size
    train_loader, test_loader, val_loader = load_and_process_data(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_per_gpu,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
        distributed=is_distributed,
    )
    logger.info(f"Loading end.")

    model_init_params = vars(model_args)
    model = model_class(**model_init_params).to(device)

    if is_distributed:  # 如果有多于一张 GPU
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    logger.info("Train and evaluate process starts.")
    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=test_loader,
        device=device,
        num_epoches=args.epoches,
        eta_min=args.eta_min,
        checkpoint_filename=os.path.join(args.experiment_dir, f'{args.model}_checkpoint_seed{args.seed}.pth'),
        model_name=args.model,
        if_augmentation=args.if_augmentation,
        seed=args.seed,
        args=args,
        logger=logger,
        local_rank=local_rank,
        writer=writer
    )

    logger.info("Program completed.")
    writer.close()


if __name__ == "__main__":
    main()
