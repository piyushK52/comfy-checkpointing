# methods for file management
import itertools
import aiohttp
import os
import heapq
import threading
import asyncio
from ..constants import NETWORK_DATA, network_data_constants, SaladConst
import server
from .common import string_hash


class FileMethods:
    def __init__(self, fetch_loop):
        self.fetch_loop = fetch_loop

    async def prepare_file(self, url, path, priority):
        """Prepare a file by fetching it if necessary and linking it to the desired path"""
        hashloc = os.path.join("fetches", string_hash(url))
        if not os.path.exists(hashloc):
            hashloc = await self.fetch_loop.enqueue(url, priority)
        if os.path.exists(path):
            os.remove(path)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        # TODO consider if symlinking would be better
        os.link(hashloc, path)

    def fetch_remote_file(self, url, filepath, file_hash=None):
        """Fetch a remote file and save it locally."""
        assert filepath.find("..") == -1, "Paths may not contain .."
        return self.prepare_file(url, filepath, -1)

    async def download_file_list(self, remote_files, uid):
        if not NETWORK_DATA:
            return

        if NETWORK_DATA["type"] == "SALAD":
            return await self.download_salad_file_list(remote_files, uid)

    async def upload_file_list(self, outputs, uid):
        if not NETWORK_DATA:
            return

        if NETWORK_DATA["type"] == "SALAD":
            return await self.download_salad_file_list(outputs, uid)

    async def download_salad_file_list(self, remote_files, uid):
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
        salad_constants: SaladConst = network_data_constants

        # TODO: Add requested support for zip files?
        if uid is not None:
            checkpoint_base = "/".join(
                [salad_constants.base_url_path, uid, "checkpoint"]
            )
            checkpoint_base = "https://storage-api.salad.com" + checkpoint_base
            async with self.fetch_loop.cs.get(
                salad_constants.base_url, headers=await salad_constants.get_headers()
            ) as r:
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
                self.fetch_remote_file(
                    f["url"], f["filepath"], f.get("file_hash", None)
                )
            )
        if len(fetches) > 0:
            await asyncio.gather(*fetches)

    async def upload_salad_file_list(self, outputs, uid):
        salad_constants: SaladConst = network_data_constants

        async with aiohttp.ClientSession("https://storage-api.salad.com") as s:
            headers = await salad_constants.get_headers()
            for i in range(len(outputs)):
                with open(outputs[i], "rb") as f:
                    data = f.read()
                # TODO support uploads > 100MB/ memory optimizations
                fd = {"file": data, "sign": "true"}
                url = "/".join(
                    [salad_constants.base_url_path, uid, "outputs", outputs[i]]
                )
                async with s.put(url, headers=headers, data=fd) as r:
                    url = (await r.json())["url"]
                outputs[i] = url

        return outputs


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
            heapq.heappush(self.queue, (priority, self.count, future, None))
            self.count += 1
            self.new_items.set()

    def enqueue_checked(self, item, priority):
        """Enqueue an item, checking for existing entries and updating priority if needed"""
        with self.lock:
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
        if url.startswith(network_data_constants.base_url):
            headers.update(await network_data_constants.get_headers())
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
