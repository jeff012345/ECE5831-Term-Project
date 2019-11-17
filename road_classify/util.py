import os

def create_dir_or_is_empty(path):
    try:
        os.mkdir(path)
    except OSError:
        if len(os.listdir(path)) != 0:
            print("Error: %s folder is not empty." % path)
            quit()
