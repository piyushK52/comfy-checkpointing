import torch
import os
import json
import safetensors.torch
import aiohttp
import asyncio
from asyncio import Future
import queue
import threading
import logging
import itertools
import hashlib
import time
from enum import Enum

import comfy.samplers
import execution
import server
import heapq

from .utils.checkpoint import get_checkpoint_client
from .utils.file import FetchLoop, FileMethods


class ExecutionResult(Enum):
    SUCCESS = 0
    FAILURE = 1
    PENDING = 2


SAMPLER_NODES = [
    "SamplerCustom",
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustomAdvanced",
]

fetch_loop = None
fetch_loop_initialized = Future()
checkpoint_client = None
file_client = None


async def initialize_fetch_loop():
    global fetch_loop
    fetch_loop = FetchLoop()
    await fetch_loop.initialize()
    fetch_loop_initialized.set_result(True)


async def get_checkpoint_client_async():
    await fetch_loop_initialized
    return get_checkpoint_client(fetch_loop=fetch_loop)


async def get_file_client_async():
    await fetch_loop_initialized
    return FileMethods(fetch_loop=fetch_loop)


async def setup():
    global checkpoint_client
    global file_client
    await initialize_fetch_loop()
    checkpoint_client = await get_checkpoint_client_async()
    file_client = await get_file_client_async()


def run_setup():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(setup())


run_setup()
completion_futures = {}


def add_future(json_data):
    """Add a future to track the completion of a prompt execution."""
    index = max(completion_futures.keys())
    if "extra_data" not in json_data:
        json_data["extra_data"] = {}
    json_data["extra_data"]["completion_future"] = index
    return json_data


server.PromptServer.instance.add_on_prompt_handler(add_future)

# ------------------- replacing the "/prompt" server method ------------
prompt_route = next(
    filter(
        lambda x: x.path == "/prompt" and x.method == "POST",
        server.PromptServer.instance.routes,
    )
)
original_post_prompt = prompt_route.handler


async def post_prompt_remote(request):
    """
    Handle remote prompt execution requests.
    Manages file fetching, execution, and result uploading.
    """
    if "dump_req" in os.environ:
        with open("resp-dump.txt", "wb") as f:
            f.write(await request.read())
        import sys

        sys.exit()

    json_data = await request.json()

    # fetching remote files (checkpoints/input files)
    extra_data = json_data.get("extra_data", {})
    remote_files = extra_data.get("remote_files", [])
    uid = json_data.get("client_id", "local")
    checkpoint_client.uid = uid
    await file_client.download_file_list(remote_files, uid=uid)
    if "prompt" not in json_data:
        return server.web.json_response("PreLoad Complete")

    # generating result
    f = asyncio.Future()
    index = max(completion_futures.keys(), default=0) + 1
    completion_futures[index] = f
    start_time = time.perf_counter()
    base_res = await original_post_prompt(request)
    outputs = await f
    execution_time = time.perf_counter() - start_time
    completion_futures.pop(index)

    # saving outputs remotely
    output_url_list = await file_client.upload_file_list(outputs, uid)

    json_output = json.loads(base_res.text)
    json_output["outputs"] = output_url_list
    json_output["execution_time"] = execution_time
    json_output["machineid"] = os.environ.get("SALAD_MACHINE_ID", "local")

    return server.web.Response(body=json.dumps(json_output))


object.__setattr__(prompt_route, "handler", post_prompt_remote)


# ------------------- replacing execute and recursive execute ---------------


class CheckpointSampler(comfy.samplers.KSAMPLER):
    """
    Custom sampler that implements checkpointing.
    Allows for resuming interrupted sampling processes.
    """

    def sample(self, *args, **kwargs):
        args = list(args)
        # self.unique_id = server.PromptServer.instance.last_node_id
        current_running_gen = (
            server.PromptServer.instance.prompt_queue.get_current_queue()
        )
        current_running_gen = current_running_gen[0]
        if current_running_gen and len(current_running_gen):
            info = current_running_gen[-1][-2]
            if "client_id" in info and info["client_id"]:
                self.unique_id = info["client_id"]
            else:
                self.unique_id = server.PromptServer.instance.last_node_id

        print("------ unique id: ", self.unique_id)
        self.step = None
        data, metadata = checkpoint_client.get(self.unique_id)
        if metadata is not None and "step" in metadata:
            data = data["x"]
            self.step = int(metadata["step"])
            # checkpoint of execution exists
            args[5] = data.to(args[4].device)
            args[1] = args[1][self.step :]
            # disable added noise, as the checkpointed latent is already noised
            args[4][:] = 0
        original_callback = args[3]

        def callback(*args):
            self.callback(*args)
            if original_callback is not None:
                return original_callback(*args)

        args[3] = callback
        res = super().sample(*args, **kwargs)
        return res

    def callback(self, step, denoised, x, total_steps):
        if self.step is not None:
            step += self.step
        data = safetensors.torch.save
        checkpoint_client.store(self.unique_id, {"x": x}, {"step": str(step)})
        if self.step is None and "FORCE_CRASH_AT" in os.environ:
            if step == int(os.environ["FORCE_CRASH_AT"]):
                raise Exception("Simulated Crash")


original_prompt_executor_execute = execution.PromptExecutor.execute
original_execute_method = execution.execute


def execute_injection(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
    """
    Injects checkpoint management and output tracking into the main execution process.
    """
    metadata = checkpoint_client.get("prompt")[1]
    if metadata is None or json.loads(metadata["prompt"]) != prompt:
        checkpoint_client.reset()
        checkpoint_client.store(
            "prompt", {"x": torch.ones(1)}, {"prompt": json.dumps(prompt)}, priority=2
        )

    prev_outputs = {}
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    for item in itertools.chain(os.scandir("output"), os.scandir("temp")):
        if item.is_file():
            prev_outputs[item.path] = item.stat().st_mtime

    # Call the original execute method
    original_prompt_executor_execute(
        self, prompt, prompt_id, extra_data, execute_outputs
    )

    outputs = []
    for item in itertools.chain(os.scandir("output"), os.scandir("temp")):
        if item.is_file() and prev_outputs.get(item.path, 0) < item.stat().st_mtime:
            outputs.append(item.path)

    if "completion_future" in extra_data:
        completion_futures[extra_data["completion_future"]].set_result(outputs)


def execute_node_injection(
    server,
    dynprompt,
    caches,
    current_item,
    extra_data,
    executed,
    prompt_id,
    execution_list,
    pending_subgraph_results,
):
    """
    Injects checkpoint loading and saving logic into the node execution process.
    """
    unique_id = current_item
    class_type = dynprompt.get_node(unique_id)["class_type"]

    if class_type in SAMPLER_NODES:
        data, metadata = checkpoint_client.get(unique_id)
        if metadata is not None and "step" in metadata:
            dynprompt.get_node(unique_id)["inputs"]["latent_image"] = [
                "checkpointed" + unique_id,
                0,
            ]
            caches.outputs.set("checkpointed" + unique_id, [[{"samples": data["x"]}]])
        elif metadata is not None and "completed" in metadata:
            outputs = json.loads(metadata["completed"])
            for x in range(len(outputs)):
                if outputs[x] == "tensor":
                    outputs[x] = list(data[str(x)])
                elif outputs[x] == "latent":
                    outputs[x] = [{"samples": l} for l in data[str(x)]]
            caches.outputs.set(unique_id, outputs)
            return ExecutionResult.SUCCESS, None, None

    # Call the original execute function
    result, error, ex = original_execute_method(
        server,
        dynprompt,
        caches,
        current_item,
        extra_data,
        executed,
        prompt_id,
        execution_list,
        pending_subgraph_results,
    )

    # Conditionally save node output
    if (
        result == ExecutionResult.SUCCESS
        and class_type in SAMPLER_NODES
        and caches.outputs.get(unique_id) is not None
    ):
        data = {}
        outputs = caches.outputs.get(unique_id).copy()
        for x in range(len(outputs)):
            if isinstance(outputs[x][0], torch.Tensor):
                data[str(x)] = torch.stack(outputs[x])
                outputs[x] = "tensor"
            elif isinstance(outputs[x][0], dict):
                data[str(x)] = torch.stack([l["samples"] for l in outputs[x]])
                outputs[x] = "latent"
        print("************  storing metadata in re: ")
        checkpoint_client.store(
            unique_id, data, {"completed": json.dumps(outputs)}, priority=1
        )

    return result, error, ex


print("------------------- replaced comfy methods with checkpointing methods")
comfy.samplers.KSAMPLER = CheckpointSampler
execution.execute = execute_node_injection
execution.PromptExecutor.execute = execute_injection

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


"""
NOTE:

Execution flow

- Client sends a POST request to /prompt.
- post_prompt_remote is called.
- post_prompt_remote calls original_post_prompt.
- original_post_prompt eventually calls execute_injection.
- execute_injection calls execute_node_injection for each node.
- If a node is a sampler, execute_node_injection uses CheckpointSampler.
- CheckpointSampler.sample is called to perform sampling.
- After all nodes are processed, execute_injection sets the future's result.
- post_prompt_remote receives the result and sends the response.
"""
