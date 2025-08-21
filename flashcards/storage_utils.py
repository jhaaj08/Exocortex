# flashcards/storage_utils.py
import os
import tempfile
import shutil
from django.core.files.storage import default_storage

def materialize_file_to_tmp(filefield):
    """
    Returns a local temp path containing the file's bytes.
    Works with FileSystemStorage and remote storages (e.g., S3).
    Caller must delete the temp file afterwards.
    
    Returns:
        tuple: (local_path, cleanup_func)
            - local_path: Path to temporary file with the content
            - cleanup_func: Function to call to clean up temp file (None if using local path)
    """
    # Try to use local path if available (dev)
    try:
        return filefield.path, None  # second value: cleanup func
    except Exception:
        pass

    # Remote storage: stream to a temp file
    tmpdir = tempfile.mkdtemp(prefix="pdf_")
    tmppath = os.path.join(tmpdir, os.path.basename(filefield.name) or "upload.pdf")
    
    with default_storage.open(filefield.name, "rb") as fsrc, open(tmppath, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)
    
    def cleanup():
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    return tmppath, cleanup