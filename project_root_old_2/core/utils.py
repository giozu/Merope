import os
import shutil
from pathlib import Path
from contextlib import contextmanager
from send2trash import send2trash

class ProjectManager:
    @staticmethod
    def cleanup_folder(path):
        """Safely removes a folder if it exists."""
        if os.path.exists(path):
            send2trash(str(path))

    @staticmethod
    @contextmanager
    def cd(newdir):
        """Context manager for changing directory safely."""
        prevdir = os.getcwd()
        os.makedirs(newdir, exist_ok=True)
        os.chdir(os.path.expanduser(newdir))
        try:
            yield
        finally:
            os.chdir(prevdir)

    @staticmethod
    def log_results(file_path, data_dict, header=None):
        """Appends a row of results to a tab-separated text file."""
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a') as f:
            if not file_exists and header:
                f.write("\t".join(header) + "\n")
            values = [f"{v:.6f}" if isinstance(v, float) else str(v) for v in data_dict.values()]
            f.write("\t".join(values) + "\n")