
# Python Imports
import os
import socket
from contextlib import closing
import time
import subprocess

# Module Imports
import torch
from tqdm import tqdm


class DistributedInfoClass(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
              cls.instance = super(DistributedInfoClass, cls).__new__(cls)
              cls.instance.reset()
        return cls.instance

    def reset(self):
        self.using_distributed = False
        self.rank = None

    def set_rank(self,rank):
        self.rank = rank

    def get_rank(self):
        return self.rank



def distributed_get_open_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def distributed_setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = str(master_port)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # Save the info
    info = DistributedInfoClass()
    info.reset()
    info.using_distributed = True
    info.set_rank(rank)

def distributed_get_rank():
    info = DistributedInfoClass()
    return info.get_rank()

def distributed_is_master():
    info = DistributedInfoClass()

    if(info.using_distributed == False):
        return True

    if(info.get_rank() == 0):
        return True

    return False

def distributed_cleanup():
    torch.distributed.destroy_process_group()


def distributed_wrap_tqdm(iterator, leave, desc, total=None, initial=0):
    if(distributed_is_master()):
        return tqdm(iterator, leave=leave, total=total, initial=initial, desc=desc)
    else:
        return iterator



