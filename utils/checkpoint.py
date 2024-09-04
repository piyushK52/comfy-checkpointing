import asyncio
import os
import queue
import threading
import safetensors.torch
import aiohttp
import logging

import server

from ..constants import NETWORK_CHECKPOINT_ENABLED, SaladConst, network_data_constants


class RequestLoop:
    """
    Manages a queue of requests with different priorities.
    Handles high-priority requests immediately and processes low-priority requests when idle.
    """

    def __init__(self, network_data_constants):
        self.active_request = None
        self.current_priority = 0
        self.queue_high = queue.Queue()
        self.low = None
        self.mutex = threading.RLock()
        self.reset_uid = False
        self.salad_constants: SaladConst = network_data_constants
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
                self.queue_high = queue.Queue()
                self.queue_high.put(req)
                if self.active_request is not None:
                    pass
                    # self.process_loop.cancel()

            elif prio == 1:
                self.low = None
                self.queue_high.put(req)
                if self.current_priority == 0:
                    pass
                    # self.process_loop.cancel()

            else:
                self.low = req

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
            async with s.delete(
                url, headers=await self.salad_constants.get_header()
            ) as r:
                await r.text()

    async def _reset(self, s, uid):
        """
        Reset all checkpoint files for a given UID.
        """
        checkpoint_base = "/".join([self.salad_constants.base_url, uid, "checkpoint"])
        async with s.get(
            self.salad_constants.base_url_path,
            headers=await self.salad_constants.get_header(),
        ) as r:
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
        headers = await self.salad_constants.get_header()
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


class SaladCheckpoint:
    """
    Implements checkpoint management for network-based storage.
    Handles saving and loading of model states and metadata.
    """

    def __init__(self, fetch_loop):
        self.fetch_loop = fetch_loop
        self.salad_consts: SaladConst = network_data_constants
        self.upload_loop = RequestLoop(constants=network_data_constants)
        self.has_warned_size = False
        assert self.salad_constants.organisation is not None

    def store(self, unique_id, tensors, metadata, priority=0):
        """Store checkpoint data in the network storage"""
        file = "/" + "/".join(
            [
                "organizations",
                self.salad_consts.organisation,
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
                self.fetch_loop.reset(
                    "/".join(
                        [
                            self.salad_consts.base_url,
                            self.uid,
                            "checkpoint",
                            f"{unique_id}.checkpoint",
                        ]
                    )
                )
            return
        os.makedirs("input/checkpoint", exist_ok=True)
        for file in os.listdir("input/checkpoint"):
            os.remove(os.path.join("input/checkpoint", file))
            self.fetch_loop.reset(
                "/".join([self.salad_consts.base_url, self.uid, "checkpoint", file])
            )


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


def get_checkpoint_client(fetch_loop):
    if NETWORK_CHECKPOINT_ENABLED:
        return SaladCheckpoint(fetch_loop=fetch_loop)  # rn only salad is supported
    else:
        return FileCheckpoint()
