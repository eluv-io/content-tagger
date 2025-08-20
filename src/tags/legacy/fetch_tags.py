import os
from loguru import logger
from src.common.content import Content

def _download_missing(q: Content, save_path: str, fabric_path: str) -> list[ValueError | None]:
    """Recursively downloads the given fabric path into save_path only if there is a difference"""
    file_info = q.list_files(path=fabric_path, get_info=True)
    to_download = []
    status = [0, 0, 0]

    def helper(data: dict, sub_path: str):
        for key, value in data.items():
            if key == ".":
                continue
            if "." in value and value["."].get("type", "") == "directory":
                helper(value, "/".join([sub_path, key]) if sub_path != "" else key)
            else:
                fpath = "/".join([sub_path, key]) if sub_path != "" else key
                fsize = value.get(".", {}).get("size", -1)
                if not os.path.exists(os.path.join(save_path, fpath)):
                    to_download.append(fpath)
                    status[0] += 1
                elif fsize != os.path.getsize(os.path.join(save_path, fpath)):
                    to_download.append(fpath)
                    status[1] += 1
                else:
                    status[2] += 1

    helper(file_info, sub_path="")
    new, changed, old = status
    logger.debug(f"{new} new files found on fabric. {changed} have changed on fabric. {old} files already up to date")

    return q.download_files([("/".join([fabric_path, path]), path) for path in to_download], dest_path=save_path)