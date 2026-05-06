from __future__ import annotations

try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from scripts.dataset._bootstrap import bootstrap_project_root

bootstrap_project_root()

from scripts.dataset.common import run_open_images_split_cli


def main() -> None:
    run_open_images_split_cli()


if __name__ == "__main__":
    main()
