from __future__ import annotations

from jetpytools import SPath, add_script_path_hook, check_perms, get_script_path, get_user_data_dir, open_file

__all__ = ["check_perms", "get_script_path", "get_user_data_dir", "open_file"]


def _vspreview_script_path() -> SPath | None:
    # TODO: move to vspreview
    try:
        from vspreview import is_preview

        if is_preview():
            from vspreview.core import main_window

            return SPath(main_window().script_path)
    except ModuleNotFoundError:
        ...

    return None


add_script_path_hook(_vspreview_script_path)
