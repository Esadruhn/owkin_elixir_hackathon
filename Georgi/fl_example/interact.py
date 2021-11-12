import substra

client = substra.Client.from_config_file(profile_name='node_A')  # profile_name=$PROFILE_NAME from the CLI

# get assets from the platform
dataset = client.get_dataset(dataset_key)
# register assets to the platform
client.add_compute_plan(
    # ...
)