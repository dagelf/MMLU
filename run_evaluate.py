from argparse import ArgumentParser, Namespace
import asyncio
from asyncio import Queue
import os
import sys
from typing import Dict, Mapping, Optional, Sequence

import torch.cuda
from tqdm import tqdm
import yaml


def _get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "-cf",
        "--cf",
        "--config",
        "--config_file",
        dest="config_file",
        type=str,
        required=True
    )

    return parser


async def run_task(
        program: str,
        *args: str,
        device_id: int,
        queue: Queue,
        os_environ: Optional[Dict[str, str]] = None
) -> None:
    # see https://docs.python.org/3/library/asyncio-subprocess.html#examples
    if os_environ is None:
        os_environ = os.environ
    os_environ = os_environ.copy()
    os_environ.update({
        "CUDA_VISIBLE_DEVICES": f"{device_id}",
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
    })

    proc = await asyncio.create_subprocess_exec(
        program,
        *args,
        # stdout=asyncio.subprocess.PIPE,
        # stderr=asyncio.subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os_environ
    )
    print(f"[{proc.pid}] Started")

    await proc.wait()
    await queue.put((device_id, proc))


async def main(args: Namespace) -> None:
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    model_args_list = []
    for model_config in tqdm(config["model_configs"], desc="Reading config"):
        if model_config is None:
            break
        elif isinstance(model_config, (str, int, float)):
            model_args_list.append(["-m", str(model_config)])
        elif isinstance(model_config, Sequence):
            model_args_list.append(model_config)
        elif isinstance(model_config, Mapping):
            model_args_list.append([
                str(m)
                for k, v in model_config.items()
                for m in (k, v)
            ])
        else:
            raise TypeError(f"{type(model_config)} is not a supported model config")

    assert torch.cuda.is_available()
    num_devices = torch.cuda.device_count()
    assert num_devices > 0

    num_initial_tasks = min(len(model_args_list), num_devices)
    num_extra_tasks = max(0, len(model_args_list) - num_devices)

    queue = Queue()
    processes = []
    for i in range(num_initial_tasks):
        processes.append(asyncio.create_task(
            run_task(
                sys.executable,
                "evaluate_llama.py",
                *model_args_list[i],
                device_id=i,
                queue=queue
            )
        ))

    num_finished_tasks = 0
    while num_finished_tasks < len(model_args_list):
        device_id, proc = await queue.get()
        print(f"[{proc.pid}] Finished!", flush=True)

        if num_finished_tasks < num_extra_tasks:
            processes.append(asyncio.create_task(
                run_task(
                    sys.executable,
                    "evaluate_llama.py",
                    *model_args_list[num_initial_tasks + num_finished_tasks],
                    device_id=device_id,
                    queue=queue
                )
            ))

        # update counter
        num_finished_tasks += 1
        queue.task_done()

    print(f"{num_finished_tasks} / {len(model_args_list)} processed. Finishing up", flush=True)
    await queue.join()
    await asyncio.gather(*processes)


if __name__ == '__main__':
    asyncio.run(main(_get_parser().parse_args()))
