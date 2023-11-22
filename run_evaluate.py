from collections import deque
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

import torch.cuda

if __name__ == '__main__':
    model_args_list = [
        (
            "-m", "meta-llama/Llama-2-7b-hf",
        ),
    ]
    output_path = Path(f"reports/eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-2]}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available()
    num_devices = torch.cuda.device_count()
    assert num_devices > 0

    # pid ->
    model_queue = deque(model_args_list)
    process_map = {}

    for cuda_id in range(min(len(model_args_list), num_devices)):
        model_args = model_queue.popleft()
        if isinstance(model_args, str):
            model_args = ("-m", model_args)

        os_environ = os.environ.copy()
        os_environ.update({
            "CUDA_VISIBLE_DEVICES": f"{cuda_id}",
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
        })

        proc = subprocess.Popen(
            [
                sys.executable,
                "evaluate_llama.py",
                *model_args,
                # *sys.argv[1:],
            ],
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=os_environ
        )
        process_map[proc.pid] = (proc, cuda_id)
        print(f"[{proc.pid}] Started")

    while len(process_map) > 0:
        pid, _ = os.wait()
        proc, cuda_id = process_map.pop(pid)
        print(f"[{proc.pid}] Finished")

        if len(model_queue) > 0:
            model_args = model_queue.popleft()

            os_environ = os.environ.copy()
            os_environ.update({
                "CUDA_VISIBLE_DEVICES": f"{cuda_id}",
            })

            proc = subprocess.Popen(
                [
                    sys.executable,
                    "evaluate_llama.py",
                    *model_args,
                    # *sys.argv[1:],
                ],
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=os_environ
            )
            process_map[proc.pid] = (proc, cuda_id)
