import os
import aiohttp
from utils.common import get_toml_config

NETWORK_CHECKPOINT_ENABLED = os.getenv("NETWORK_CHECKPOINT_ENABLED", False)


class SaladConst:
    def __init__(self, organization, token):
        self.organisation = organization
        self.base_url_path = "/organizations/" + organization + "/files"
        self.base_url = "https://storage-api.salad.com" + self.base_url_path

        async def get_header():
            if "SALAD_API_KEY" in os.environ:
                # NOTE: Only for local testing. Do not add to container
                return {"Salad-Api-Key": os.environ["SALAD_API_KEY"]}

            if token is None:
                assert (
                    "SALAD_MACHINE_ID" in os.environ
                ), "SALAD_API_KEY must be provided if not deployed"
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://169.254.169.254:80/v1/token") as r:
                        SALAD_TOKEN = (await r.json())["jwt"]
            return {"Authorization": SALAD_TOKEN}

        self.get_headers = get_header


def get_net_const():
    if NETWORK_DATA["type"] == "SALAD":  # only supported provider rn
        return SaladConst()
    raise ("Network provider not supported")
    return None


if NETWORK_CHECKPOINT_ENABLED:
    NETWORK_DATA = get_toml_config()
    network_data_constants = get_net_const()
else:
    NETWORK_DATA = None
    network_data_constants = None
