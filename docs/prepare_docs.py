import argparse
import pathlib
import shutil

DOC_FOLDERS = ["_api", "_build", "auto_examples", "source/backreferences"]


def rebuild_folder(folder: str):
    """Rebuild a folder.

    Parameters
    ----------
    folder : str
        Path to the folder to rebuild.
    """
    shutil.rmtree(folder, ignore_errors=True)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


def main():
    """Parse arguments and rebuild specified folders.

    Use `--folders all` to rebuild all documentation folders.
    Use `--folders <folder1> <folder2> ...` to rebuild specific folders.
    """
    parser = argparse.ArgumentParser(
        description="Rebuild documentation folders."
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        choices=DOC_FOLDERS + ["all"],
        required=True,
        help="Folders to rebuild."
        " Use 'all' to rebuild all documentation folders.",
    )
    args = parser.parse_args()

    folders_to_rebuild = DOC_FOLDERS if "all" in args.folders else args.folders

    for folder in folders_to_rebuild:
        rebuild_folder(folder)


if __name__ == "__main__":
    main()
