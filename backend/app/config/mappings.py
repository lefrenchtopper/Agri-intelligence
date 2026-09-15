CROP_MAPPINGS = {
    "onion": "c49a8637df4d4d399d27c7756c0ffa4f",
    "bellary onion": "c49a8637df4d4d399d27c7756c0ffa4f",
    "small onion": "c49a8637df4d4d399d27c7756c0ffa4f",
}

def get_crop_id(commodity_name: str) -> str | None:
    if not commodity_name:
        return None
    return CROP_MAPPINGS.get(commodity_name.strip().lower())