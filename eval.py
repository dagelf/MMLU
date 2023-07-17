import argparse
import os
from utils import get_model_tokenizer
from pytorch_lightning import seed_everything, Trainer
from module import MMLUModel, MMLUDataModule
import torch
import warnings
from categories import subcategories, categories
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


def main(args):

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.precision == "bf16":
        torch.set_float32_matmul_precision("high")
        torch_dtype = torch.bfloat16
        trainer_precision = "bf16-mixed"
    elif args.precision == "fp16":
        torch_dtype = torch.float16
        trainer_precision = 32
    else:
        torch_dtype = torch.float32
        trainer_precision = 32

    model, tokenizer = get_model_tokenizer(args.model, torch_dtype)
    
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    model = MMLUModel(model, tokenizer, args)
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="auto",
        precision=trainer_precision,
    )

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    results_df = pd.DataFrame(columns=["subject", "accuracy", "loss", "granularity"])
    for subject in subjects:
        data_module = MMLUDataModule(args.data_dir, args.prompt_dir, tokenizer, args.batch_size, subject, args.ntrain)
        data_module.prepare_data()

        val_dataloader = data_module.val_dataloader()
        test_dataloader = data_module.test_dataloader()

        if "alpaca" not in args.model:
            trainer.validate(model, dataloaders=val_dataloader, verbose=False)
            val_loss = model.get_val_loss()
        else:
            val_loss = 0

        trainer.test(model, dataloaders=test_dataloader, verbose=False)
        acc = model.metric.compute().item()
        print("Average accuracy {:.3f}; Average val_loss {:.4f} - {}".format(acc, val_loss, subject))
        results_df.loc[len(results_df)] = [subject, acc, val_loss, "subject"]
        
        cor = model.metric.correct.item()
        total = model.metric.total.item()

        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append((cor, total))
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append((cor, total))
        all_cors.append((cor, total))

    for subcat in subcat_cors:
        cors, totals = zip(*subcat_cors[subcat])
        subcat_acc = sum(cors) / sum(totals)

        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
        results_df.loc[len(results_df)] = [subcat, subcat_acc, None, "subcategory"]

    for cat in cat_cors:
        cors, totals = zip(*cat_cors[cat])
        cat_acc = sum(cors) / sum(totals)
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
        results_df.loc[len(results_df)] = [cat, cat_acc, None, "category"]

    cors, totals = zip(*all_cors)
    weighted_acc = sum(cors) / sum(totals)

    print("Average accuracy: {:.3f}".format(weighted_acc))
    results_df.loc[len(results_df)] = ["all", weighted_acc, None, "overall"]
    results_df.to_csv(os.path.join(args.save_dir, "results.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument('--precision', default="fp32", choices=["fp16", "fp32", "bf16"], type=str)
    parser.add_argument('--devices', default=[0], type=int, nargs="+")
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--prompt_dir', "-p", type=str, default=None)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results/run_best")

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="../results/run_best/t5-base",
    )
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)