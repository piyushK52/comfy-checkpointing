import os
import aiohttp
import toml


# TODO: create a abstract interface with base_url and get_headers as mandatory fields
class SaladConst:
    def __init__(self, organization, token):
        self.organisation = organization
        self.base_url_path = "/organizations/" + organization + "/files"
        self.base_url = "https://storage-api.salad.com" + self.base_url_path

        async def get_header():
            if token:
                # NOTE: Only for local testing. Do not add to container
                return {"Salad-Api-Key": token}

            if token is None:
                assert (
                    "SALAD_MACHINE_ID" in os.environ
                ), "SALAD_API_KEY must be provided if not deployed"
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://169.254.169.254:80/v1/token") as r:
                        SALAD_TOKEN = (await r.json())["jwt"]
            return {"Authorization": SALAD_TOKEN}

        self.get_headers = get_header


def get_net_const(data):
    if data["type"] == "SALAD":  # only supported provider rn
        return SaladConst(
            data["organisation"],
            data["salad_api_key"]
        )
    raise ("Network provider not supported")

def get_toml_config(key=None, toml_file="config.toml"):
    toml_config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            toml_file,
        )
    )

    toml_data = {}
    if os.path.exists(toml_config_path):
        with open(toml_config_path, "r") as f:
            toml_data = toml.load(f)

    if key and key in toml_data:
        return toml_data[key]
    return toml_data

NETWORK_DATA = get_toml_config()
if NETWORK_DATA and len(NETWORK_DATA):
    NETWORK_DATA = NETWORK_DATA.get("network_data", {})
NETWORK_CHECKPOINT_ENABLED = NETWORK_DATA.get("NETWORK_CHECKPOINT_ENABLED", False)

if NETWORK_CHECKPOINT_ENABLED:
    network_data_constants = get_net_const(NETWORK_DATA)
else:
    NETWORK_DATA = None
    network_data_constants = None


