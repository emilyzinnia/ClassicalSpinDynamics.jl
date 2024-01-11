import os 
import argh
from argh import arg
import h5py

@arg("stackfile", default="stack", help="Path to stackfile")
@arg("path", default="None", help="Path to add to stackfile")
def push(stackfile, path):
    """Pushes single path with trailing backslash to stackfile."""
    with open(stackfile, "+a") as f:
        print(path+"\n")
        f.write(path+"\n")

@arg("stackfile", default="stack", help="Path to stackfile")
@arg("target_dir", default="None", help="Path to target directory containing files to add to stackfile")
def push_all(stackfile, target_dir):
    """Pushes all files in target directory with trailing backslash to stackfile."""
    f = open(stackfile, "+a")
    for file in os.listdir(target_dir):
        print(os.path.join(target_dir, file))
        f.write(os.path.join(target_dir, file)+"\n")
    f.close()

@arg("stackfile", default="stack", help="Path to stackfile")
@arg("target_dir", default="None", help="Path to target directory containing files to add to stackfile")
def repush(stackfile, target_dir, key="spectroscopy"):
    """Checks to see if group is present in a file. If not, push to stack"""
    f = open(stackfile, "+a")
    for file in os.listdir(target_dir):
        with h5py.File(file, "r") as h5:
            if not(key in h5.keys()):
                push(stackfile, os.path.join(target_dir, file))
    f.close()

parser = argh.ArghParser()
parser.add_commands([push, push_all, repush])

if __name__ == "__main__":
    parser.dispatch()