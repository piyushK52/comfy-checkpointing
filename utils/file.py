# methods for file management

import os


class FileMethods:
    
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
    
    def fetch_remote_file(self, url, filepath, file_hash=None):
        """Fetch a remote file and save it locally."""
        assert filepath.find("..") == -1, "Paths may not contain .."
        return self.prepare_file(url, filepath, -1)

    async def download_file_list(self, remote_files, uid):
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
                self.fetch_remote_file(f["url"], f["filepath"], f.get("file_hash", None))
            )
        if len(fetches) > 0:
            await asyncio.gather(*fetches)
    
    async def upload_file_list(self, outputs):
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