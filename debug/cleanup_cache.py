import os
import shutil

def clean_python_cache(root_path='.'):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Remove __pycache__ directories
        if '__pycache__' in dirnames:
            pycache_dir = os.path.join(dirpath, '__pycache__')
            shutil.rmtree(pycache_dir)
            print(f"Removed: {pycache_dir}")
            count += 1

        # Remove *.pyc and *.pyo files
        for filename in filenames:
            if filename.endswith('.pyc') or filename.endswith('.pyo'):
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"Removed: {file_path}")
                count += 1

    print(f"\n✅ Cleaned {count} cache files/directories.")

if __name__ == "__main__":
    clean_python_cache()  # or pass a custom path like clean_python_cache('/my/project')