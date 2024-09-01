import torch
import os
import json
import safetensors.torch
import aiohttp
import asyncio
import queue
import threading
import logging
import itertools
import hashlib
import time

import comfy.samplers
import execution
import server
import heapq

# Define the types of sampler nodes that will be affected by checkpointing
SAMPLER_NODES = [
    "SamplerCustom",
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustomAdvanced",
]

SALAD_TOKEN = None
ORGANIZATION = os.environ.get("SALAD_ORGANIZATION", None)
if ORGANIZATION is not None:
    base_url_path = "/organizations/" + ORGANIZATION + "/files"
    base_url = "https://storage-api.salad.com" + base_url_path


async def get_header():
    """
    Retrieve the appropriate headers for API requests.
    If running locally, use SALAD_API_KEY from environment.
    If deployed, fetch and cache the JWT token from Salad's metadata service
    """
    if "SALAD_API_KEY" in os.environ:
        # NOTE: Only for local testing. Do not add to container
        return {"Salad-Api-Key": os.environ["SALAD_API_KEY"]}
    global SALAD_TOKEN
    if SALAD_TOKEN is None:
        assert (
            "SALAD_MACHINE_ID" in os.environ
        ), "SALAD_API_KEY must be provided if not deployed"
        async with aiohttp.ClientSession() as session:
            async with session.get("http://169.254.169.254:80/v1/token") as r:
                SALAD_TOKEN = (await r.json())["jwt"]
    return {"Authorization": SALAD_TOKEN}


class FetchQueue:
    """
    A priority queue implementation for managing file fetch requests.
    Supports updating priorities and tracking in-flight requests.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.queue = []  # queue contains --> (priority, url, future)
        self.count = 0
        self.consumed = (
            {}
        )  # consumed containrs --> {item_1: (item_future, item_priority)}
        self.new_items = asyncio.Event()

    def update_priority(self, i, priority):
        """Update the priority of an item in the queue"""
        # lock must already be acquired
        future = self.queue[i][3]
        item = self.queue[i][2]
        if priority < self.queue[i][0]:
            # priority is increased, invalidate old
            self.queue[i] = (self.queue[i][0], self.queue[i][1], None, None)
            heapq.heappush(self.queue, (priority, self.count, item, future))
            self.count += 1

    def requeue(self, future, item, dec_priority=1):
        """Requeue an item with decreased priority"""
        with self.lock:
            priority = self.consumed[item][1] - dec_priority
            heapq.heappush(
                self.queue, (priority, self.count, future, None)
            )  # DOUBT: check what will go inplace of the self.count (item ?)
            self.count += 1
            self.new_items.set()

    def enqueue_checked(self, item, priority):
        """Enqueue an item, checking for existing entries and updating priority if needed"""
        with self.lock:
            # DOUBT: return the item if it is already fetched ?
            if item in self.consumed:
                # TODO: Also update in queue
                # TODO: if complete check etag?
                self.consumed[item][1] = min(self.consumed[item][1], priority)
                return self.consumed[item][0]

            # if the item is already in the queue, just update it's priority and return the future
            for i in range(len(self.queue)):
                if self.queue[i][2] == item:
                    future = self.queue[i][3]
                    self.update_priority(i, priority)
                    return future

            # enqueuing a fresh item
            future = asyncio.Future()
            heapq.heappush(self.queue, (priority, self.count, item, future))
            self.count += 1
            self.new_items.set()
            return future

    async def get(self):
        """Get the next item from the queue"""
        while True:
            await self.new_items.wait()
            with self.lock:
                priority, _, item, future = heapq.heappop(self.queue)
                if len(self.queue) == 0:
                    self.new_items.clear()
                if item is not None:
                    if isinstance(item, str):
                        self.consumed[item] = [future, priority]
                        return priority, item, future
                    else:
                        # item is future
                        item.set_result(True)


class FetchLoop:
    """
    Manages asynchronous fetching of remote files.
    Implements a semaphore to limit concurrent downloads.
    """

    def __init__(self):
        self.queue = FetchQueue()
        self.semaphore = asyncio.Semaphore(5)

        self.cs = None

        event_loop = server.PromptServer.instance.loop
        self.process_loop = event_loop.create_task(self.loop())

        os.makedirs("fetches", exist_ok=True)
        
    async def initialize(self):
        self.cs = aiohttp.ClientSession()

    async def loop(self):
        """Main loop for processing fetch requests"""
        event_loop = server.PromptServer.instance.loop
        while True:
            await self.semaphore.acquire()
            event_loop.create_task(self.fetch(*(await self.queue.get())))

    def reset(self, url):
        """Reset the fetch state for a given URL"""
        with self.queue.lock:
            if url in self.queue.consumed:
                self.queue.consumed.pop(url)
        hashloc = os.path.join("fetches", string_hash(url))
        if os.path.exists(hashloc):
            os.remove(hashloc)

    def enqueue(self, url, priority=0):
        """Enqueue a URL for fetching"""
        return self.queue.enqueue_checked(url, priority)

    async def fetch(self, priority, url, future):
        """Fetch a file from the given URL"""
        chunk_size = 2**25  # 32MB
        headers = {}
        if url.startswith(base_url):
            headers.update(await get_header())
        filename = os.path.join("fetches", string_hash(url))
        try:
            async with self.cs.get(url, headers=headers) as r:
                with open(filename, "wb") as f:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        f.write(chunk)
                        if not r.content.is_eof():
                            awaken = asyncio.Future()
                            self.queue.requeue(awaken, url)
                            await awaken
            future.set_result(filename)
        except:
            future.set_result(None)
            raise
        finally:
            self.semaphore.release()
        return


# fetch_loop = FetchLoop()
async def initialize_fetch_loop():
    global fetch_loop
    loop = asyncio.get_event_loop()
    fetch_loop = FetchLoop()
    await fetch_loop.initialize()

server.PromptServer.instance.loop.create_task(initialize_fetch_loop())

async def prepare_file(url, path, priority):
    """Prepare a file by fetching it if necessary and linking it to the desired path"""
    hashloc = os.path.join("fetches", string_hash(url))
    if not os.path.exists(hashloc):
        hashloc = await fetch_loop.enqueue(url, priority)
    if os.path.exists(path):
        os.remove(path)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    # TODO consider if symlinking would be better
    os.link(hashloc, path)


# NOTE: most methods have a RLock on them, meaning this as a whole will mostly be run
# by one thread at a time
class RequestLoop:
    """
    Manages a queue of requests with different priorities.
    Handles high-priority requests immediately and processes low-priority requests when idle.
    """

    def __init__(self):
        self.active_request = None
        self.current_priority = 0
        self.queue_high = queue.Queue()
        self.low = None
        self.mutex = threading.RLock()
        self.reset_uid = False
        # main.py has already created an event loop
        event_loop = server.PromptServer.instance.loop
        self.process_loop = event_loop.create_task(self.process_requests())

    def queue(self, req, prio):
        """
        Add a request to the queue with the specified priority.
        Higher priority requests are processed first.
        """
        with self.mutex:
            if prio == 2:
                self.low = None
                self.queue_high = (
                    queue.Queue()
                )  # DOUBT: we are discarding everything, when we get a priority 2 element ?
                self.queue_high.put(req)
                if self.active_request is not None:
                    pass
                    # self.process_loop.cancel()

            elif prio == 1:
                self.low = None  # DOUBT: why remove the low element ?
                self.queue_high.put(req)
                if self.current_priority == 0:
                    pass
                    # self.process_loop.cancel()

            else:
                self.low = req  # DOUBT: what happens to the request already at low ?

    def reset(self, uid):
        """
        Reset the request queue and mark for resetting files associated with the given UID.
        """
        with self.mutex:
            self.low = None
            self.queue_high = queue.Queue()
            self.reset_uid = uid

    async def delete_file(self, s, url, semaphore):
        """
        Delete a file from remote storage using the provided URL.
        """
        async with semaphore:
            async with s.delete(url, headers=await get_header()) as r:
                await r.text()

    async def _reset(self, s, uid):
        """
        Reset all checkpoint files for a given UID.
        """
        checkpoint_base = "/".join([base_url, uid, "checkpoint"])
        async with s.get(base_url_path, headers=await get_header()) as r:
            js = await r.json()
            files = js["files"]
        checkpoints = list(
            filter(lambda x: x["url"].startswith(checkpoint_base), files)
        )
        for cp in checkpoints:
            cp["url"] = cp["url"][29:]
        semaphore = asyncio.Semaphore(5)
        deletes = [
            asyncio.create_task(self.delete_file(s, f["url"], semaphore))
            for f in checkpoints
        ]
        if len(deletes) > 0:
            await asyncio.gather(*deletes)

    async def process_requests(self):
        """
        Main loop for processing queued requests.
        Handles file uploads and manages the request queue.
        """
        headers = await get_header()
        async with aiohttp.ClientSession("https://storage-api.salad.com") as session:
            try:
                while True:
                    if self.reset_uid != False:
                        await self._reset(session, self.reset_uid)
                        self.reset_uid = False

                    if self.active_request is None:
                        await asyncio.sleep(0.1)
                    else:
                        req = self.active_request
                        fd = aiohttp.FormData({"file": req[1]})
                        async with session.put(req[0], headers=headers, data=fd) as r:
                            # We don't care about result, but must still await it
                            await r.text()

                    with self.mutex:
                        if not self.queue_high.empty():
                            self.active_request = self.queue_high.get()
                        elif self.low is not None:
                            self.active_request = self.low
                            self.low = None
                        else:
                            self.active_request = None
            except:
                # Exceptions from event loop get swallowed and kill the loop
                import traceback

                traceback.print_exc()
                raise


class NetCheckpoint:
    """
    Implements checkpoint management for network-based storage.
    Handles saving and loading of model states and metadata.
    """

    def __init__(self):
        self.upload_loop = RequestLoop()
        self.has_warned_size = False
        assert ORGANIZATION is not None

    def store(self, unique_id, tensors, metadata, priority=0):
        """Store checkpoint data in the network storage"""
        file = "/" + "/".join(
            [
                "organizations",
                ORGANIZATION,
                "files",
                self.uid,
                "checkpoint",
                f"{unique_id}.checkpoint",
            ]
        )
        data = safetensors.torch.save(tensors, metadata)
        if len(data) > 10**8:
            if not self.has_warned_size:
                logging.warning("Checkpoint is too large and has been skipped")
                self.has_warned_size = True
            return
        self.upload_loop.queue((file, data), priority)

    def get(self, unique_id):
        """Retrieve checkpoint data from the network storage"""
        file = f"input/checkpoint/{unique_id}.checkpoint"
        if not os.path.exists(file):
            return None, None
        with safetensors.torch.safe_open(file, framework="pt") as f:
            metadata = f.metadata()
            tensors = {key: f.get_tensor(key) for key in f.keys()}
            return tensors, metadata

    def reset(self, unique_id=None):
        """Reset checkpoint data, optionally for a specific unique_id"""
        self.upload_loop.reset(self.uid)
        if unique_id is not None:
            if os.path.exists(f"input/checkpoint/{unique_id}.checkpoint"):
                os.remove(f"input/checkpoint/{unique_id}.checkpoint")
                fetch_loop.reset(
                    "/".join(
                        [base_url, self.uid, "checkpoint", f"{unique_id}.checkpoint"]
                    )
                )
            return
        os.makedirs("input/checkpoint", exist_ok=True)
        for file in os.listdir("input/checkpoint"):
            os.remove(os.path.join("input/checkpoint", file))
            fetch_loop.reset("/".join([base_url, self.uid, "checkpoint", file]))


class FileCheckpoint:
    """
    Implements checkpoint management for local file-based storage.
    Handles saving and loading of model states and metadata.
    """

    def store(self, unique_id, tensors, metadata, priority=0):
        """Store checkpoint data in local file storage"""
        file = f"checkpoint/{unique_id}.checkpoint"
        if not os.path.exists(f"checkpoint"):
            os.mkdir("checkpoint")
            
        safetensors.torch.save_file(tensors, file, metadata)

    def get(self, unique_id):
        """Retrieve checkpoint data from local file storage"""
        file = f"checkpoint/{unique_id}.checkpoint"
        if not os.path.exists(file):
            return None, None
        with safetensors.torch.safe_open(file, framework="pt") as f:
            metadata = f.metadata()
            tensors = {key: f.get_tensor(key) for key in f.keys()}
            return tensors, metadata

    def reset(self, unique_id=None):
        """Clear all checkpoint information."""
        if unique_id is not None:
            if os.path.exists(f"checkpoint/{unique_id}.checkpoint"):
                os.remove(f"checkpoint/{unique_id}.checkpoint")
            return

        if os.path.exists("checkpoint"):
            for file in os.listdir("checkpoint"):
                os.remove(os.path.join("checkpoint", file))


checkpoint = NetCheckpoint() if "SALAD_ORGANIZATION" in os.environ else FileCheckpoint()


def file_hash(filename):
    """Calculate the SHA256 hash of a file."""
    h = hashlib.sha256()
    b = bytearray(10 * 1024 * 1024)  # read 10 megabytes at a time
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(b):
            h.update(b)
    return h.hexdigest()


def string_hash(s):
    """Calculate the SHA256 hash of a string."""
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def fetch_remote_file(url, filepath, file_hash=None):
    """Fetch a remote file and save it locally."""
    assert filepath.find("..") == -1, "Paths may not contain .."
    return prepare_file(url, filepath, -1)


async def fetch_remote_file_list(remote_files, uid=None):
    """
    Fetch multiple remote files, including checkpoints if a UID is provided.

    URL types
    REMOTE
    base_url - https://storage-api.salad.com/organizations/{ORGANIZATION}/files
    checkpoint file - /{ORGANIZATION}/files/{uid}/checkpoint/{unique_id}.checkpoint
    output file - /{ORGANIZATION}/files/{uid}/outputs/{output_filename}

    LOCAL
    fetches - fetches/{hashed_url}
    checkpoint file - input/checkpoint/{unique_id}.checkpoint
    output file - output/{filename} or temp/{filename}
    """
    # TODO: Add requested support for zip files?
    if uid is not None:
        checkpoint_base = "/".join([base_url_path, uid, "checkpoint"])
        checkpoint_base = "https://storage-api.salad.com" + checkpoint_base
        async with fetch_loop.cs.get(base_url, headers=await get_header()) as r:
            js = await r.json()
        files = js["files"]
        checkpoints = list(
            filter(lambda x: x["url"].startswith(checkpoint_base), files)
        )
        for cp in checkpoints:
            cp["filepath"] = os.path.join(
                "input/checkpoint", cp["url"][len(checkpoint_base) + 1 :]
            )
        remote_files = itertools.chain(remote_files, checkpoints)
    fetches = []
    for f in remote_files:
        fetches.append(
            fetch_remote_file(f["url"], f["filepath"], f.get("file_hash", None))
        )
    if len(fetches) > 0:
        await asyncio.gather(*fetches)


completion_futures = {}


def add_future(json_data):
    """Add a future to track the completion of a prompt execution."""
    index = max(completion_futures.keys())
    if "extra_data" not in json_data:
        json_data["extra_data"] = {}
    json_data["extra_data"]["completion_future"] = index
    return json_data


server.PromptServer.instance.add_on_prompt_handler(add_future)

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
    if "SALAD_ORGANIZATION" in os.environ:
        # TODO: update this to extract the params required by comfy_runner
        extra_data = json_data.get("extra_data", {})
        remote_files = extra_data.get("remote_files", [])
        uid = json_data.get("client_id", "local")
        checkpoint.uid = uid
        await fetch_remote_file_list(remote_files, uid=uid)
        if "prompt" not in json_data:
            return server.web.json_response("PreLoad Complete")

    f = asyncio.Future()
    index = max(completion_futures.keys(), default=0) + 1
    completion_futures[index] = f
    start_time = time.perf_counter()
    base_res = await original_post_prompt(request)
    outputs = await f
    execution_time = time.perf_counter() - start_time
    completion_futures.pop(index)
    if "SALAD_ORGANIZATION" in os.environ:
        async with aiohttp.ClientSession("https://storage-api.salad.com") as s:
            headers = await get_header()
            for i in range(len(outputs)):
                with open(outputs[i], "rb") as f:
                    data = f.read()
                # TODO support uploads > 100MB/ memory optimizations
                fd = {"file": data, "sign": "true"}
                url = "/".join([base_url_path, uid, "outputs", outputs[i]])
                async with s.put(url, headers=headers, data=fd) as r:
                    url = (await r.json())["url"]
                outputs[i] = url
    json_output = json.loads(base_res.text)
    json_output["outputs"] = outputs
    json_output["execution_time"] = execution_time
    json_output["machineid"] = os.environ.get("SALAD_MACHINE_ID", "local")
    return server.web.Response(body=json.dumps(json_output))


# Dangerous
object.__setattr__(prompt_route, "handler", post_prompt_remote)


class CheckpointSampler(comfy.samplers.KSAMPLER):
    """
    Custom sampler that implements checkpointing.
    Allows for resuming interrupted sampling processes.
    """

    def sample(self, *args, **kwargs):
        args = list(args)
        self.unique_id = server.PromptServer.instance.last_node_id
        self.step = None
        data, metadata = checkpoint.get(self.unique_id)
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
        checkpoint.store(self.unique_id, {"x": x}, {"step": str(step)})
        if self.step is None and "FORCE_CRASH_AT" in os.environ:
            if step == int(os.environ["FORCE_CRASH_AT"]):
                raise Exception("Simulated Crash")


original_recursive_execute = execution.recursive_execute


def recursive_execute_injection(*args):
    """
    Injects checkpoint loading and saving logic into the recursive execution process.
    """
    unique_id = args[3]
    class_type = args[1][unique_id]["class_type"]
    extra_data = args[4]
    if class_type in SAMPLER_NODES:
        data, metadata = checkpoint.get(unique_id)
        if metadata is not None and "step" in metadata:
            args[1][unique_id]["inputs"]["latent_image"] = [
                "checkpointed" + unique_id,
                0,
            ]
            args[2]["checkpointed" + unique_id] = [[{"samples": data["x"]}]]
        elif metadata is not None and "completed" in metadata:
            outputs = json.loads(metadata["completed"])
            for x in range(len(outputs)):
                if outputs[x] == "tensor":
                    outputs[x] = list(data[str(x)])
                elif outputs[x] == "latent":
                    outputs[x] = [{"samples": l} for l in data[str(x)]]
            args[2][unique_id] = outputs
            return True, None, None

    res = original_recursive_execute(*args)
    # Conditionally save node output
    # TODO: determine which non-sampler nodes are worth saving
    if class_type in SAMPLER_NODES and unique_id in args[2]:
        data = {}
        outputs = args[2][unique_id].copy()
        for x in range(len(outputs)):
            if isinstance(outputs[x][0], torch.Tensor):
                data[str(x)] = torch.stack(outputs[x])
                outputs[x] = "tensor"
            elif isinstance(outputs[x][0], dict):
                data[str(x)] = torch.stack([l["samples"] for l in outputs[x]])
                outputs[x] = "latent"
        checkpoint.store(
            unique_id, data, {"completed": json.dumps(outputs)}, priority=1
        )
    return res


original_execute = execution.PromptExecutor.execute


def execute_injection(*args, **kwargs):
    """
    Injects checkpoint management and output tracking into the main execution process.
    """
    metadata = checkpoint.get("prompt")[1]
    if metadata is None or json.loads(metadata["prompt"]) != args[1]:
        checkpoint.reset()
        checkpoint.store(
            "prompt", {"x": torch.ones(1)}, {"prompt": json.dumps(args[1])}, priority=2
        )

    prev_outputs = {}
    os.makedirs("temp", exist_ok=True)
    # TODO: Consider subdir recursing?
    for item in itertools.chain(os.scandir("output"), os.scandir("temp")):
        if item.is_file():
            prev_outputs[item.path] = item.stat().st_mtime

    original_execute(*args, **kwargs)

    outputs = []
    for item in itertools.chain(os.scandir("output"), os.scandir("temp")):
        if item.is_file() and prev_outputs.get(item.path, 0) < item.stat().st_mtime:
            outputs.append(item.path)

    if "completion_future" in args[3]:
        completion_futures[args[3]["completion_future"]].set_result(outputs)

print("------------------- replaced the default classes/methods with custom checkpointing code")
comfy.samplers.KSAMPLER = CheckpointSampler
execution.recursive_execute = recursive_execute_injection
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
- execute_injection calls recursive_execute_injection for each node.
- If a node is a sampler, recursive_execute_injection uses CheckpointSampler.
- CheckpointSampler.sample is called to perform sampling.
- After all nodes are processed, execute_injection sets the future's result.
- post_prompt_remote receives the result and sends the response.
"""
