# http://mvapich.cse.ohio-state.edu/benchmarks/

import tenpy
from tenpy.linalg import np_conserved as npc
from tenpy.tools import hdf5_io

import numpy as np

import mpi4py
from mpi4py import MPI

def load_env(large=True):
    if large:
        env = hdf5_io.load('state_nu_0.0_mpo_svd_1e-06_mps_chi_8192_mpirank_0.h5')
        LP = env['resume_data']['init_env_data']['init_LP']
    else:
        print('state_4uc_nu_-0.2_mpo_svd_1e-06_mps_chi_4096.h5', flush=True)
        #env = hdf5_io.load('state_nu_0.0_mpo_svd_1e-06_mps_chi_6144_mpirank_0.h5')
        #env = hdf5_io.load('state_nu_0.0_mpo_svd_1e-06_mps_chi_4096.h5')
        env = hdf5_io.load('state_4uc_nu_-0.2_mpo_svd_1e-06_mps_chi_4096.h5')
        LP = env['resume_data']['init_env_data']['init_LP']
    size = LP.size * 16 # 16 bytes per complex128
    #print('Env norm^2:', npc.tensordot(LP, LP.conj(), axes=(['vR*', 'wR', 'vR'], ['vR', 'wR*', 'vR*'])),
    #          flush=True)
    
    return LP, size

def big_send_env(comm, env, dest, tag, max_message=2**31-1):
    env_data_orig = env._data
    t_start = MPI.Wtime()
    env_data_flat = np.concatenate([d.flatten() for d in env._data])
    t1 = MPI.Wtime()
    #print("SEND - Concatenation time:", t1 - t_start, flush=True)
    
    block_shapes = [d.shape for d in env._data]
    if env.dtype == 'complex128':
        dtype_size = 16
    elif env.dtype == 'float64':
        dtype_size = 8
    else:
        raise ValueError('dtype %s of environment not recognized' % env.dtype)

    data_size = env.size * dtype_size # np.complex128 is 16 bytes
    num_messages = 1 + data_size // max_message # Max message is 2^31 - 1 bytes
    #print('SEND - # Messages:', num_messages, flush=True)
    message_size = env.size // num_messages
    message_boundary = [message_size * i for i in range(num_messages)] + [env.size]
    assert message_boundary[-1] - message_boundary[-2] <= 2147483647

    env._data = [block_shapes] + [message_boundary] + [env.dtype]
    comm.send(env, dest=dest, tag=tag)
    env._data = env_data_orig
    t2 = MPI.Wtime()
    #print("SEND - First send:", t2 - t1, flush=True)
    
    for i in range(num_messages):
        t1 = MPI.Wtime()
        comm.Send(env_data_flat[message_boundary[i]:message_boundary[i+1]], dest=dest, tag=tag*10+i)
        t2 = MPI.Wtime()
        #print("SEND - Flat Send:", t2 - t1, flush=True)


def big_recv_env(comm, source, tag):
    t_start = MPI.Wtime()
    env = comm.recv(source=source, tag=tag)
    t1 = MPI.Wtime()
    #print("RECV - First recv:", t1 - t_start, flush=True)
    
    block_shapes = env._data[0]
    message_boundary = env._data[1]
    dtype = env._data[2]
    env_data = np.empty(message_boundary[-1], dtype=dtype)
    
    for i in range(len(message_boundary) - 1):
        t1 = MPI.Wtime()
        comm.Recv(env_data[message_boundary[i]:message_boundary[i+1]], source=source, tag=tag*10+i)
        t2 = MPI.Wtime()
        #print("RECV - Flat Recv:", t2 - t1, flush=True)
    
    t1 = MPI.Wtime()
    env_data = np.split(env_data, np.cumsum([np.prod(d) for d in block_shapes])[:-1])
    t2 = MPI.Wtime()
    #print("RECV - Split data:", t2 - t1, flush=True)
    env._data = [d.reshape(block_shapes[i]) for i, d in enumerate(env_data)]
    t3 = MPI.Wtime()
    #print("RECV - Reshape data:", t3 - t2, flush=True)
    
    return env

def big_Isend_env(comm, env, dest, tag, max_message=2**31-1):
    env_data_orig = env._data
    t_start = MPI.Wtime()
    env_data_flat = np.concatenate([d.flatten() for d in env._data])
    t1 = MPI.Wtime()
    #print("SEND - Concatenation time:", t1 - t_start, flush=True)
    
    block_shapes = [d.shape for d in env._data]
    if env.dtype == 'complex128':
        dtype_size = 16
    elif env.dtype == 'float64':
        dtype_size = 8
    else:
        raise ValueError('dtype %s of environment not recognized' % env.dtype)

    data_size = env.size * dtype_size # np.complex128 is 16 bytes
    num_messages = 1 + data_size // max_message # Max message is 2^31 - 1 bytes
    #print('SEND - # Messages:', num_messages, flush=True)
    message_size = env.size // num_messages
    message_boundary = [message_size * i for i in range(num_messages)] + [env.size]
    assert message_boundary[-1] - message_boundary[-2] <= 2147483647

    env._data = [block_shapes] + [message_boundary] + [env.dtype]
    comm.send(env, dest=dest, tag=tag)
    env._data = env_data_orig
    t2 = MPI.Wtime()
    #print("SEND - First send:", t2 - t1, flush=True)
    
    requests = [MPI.REQUEST_NULL] * num_messages
    
    t1 = MPI.Wtime()
    for i in range(num_messages):
        requests[i] = comm.Isend(env_data_flat[message_boundary[i]:message_boundary[i+1]], dest=dest, tag=tag*10+i)
        
    MPI.Request.Waitall(requests)
    t2 = MPI.Wtime()
    #print("SEND - Flat I Send:", t2 - t1, flush=True)
    
def big_Irecv_env(comm, source, tag):
    t_start = MPI.Wtime()
    env = comm.recv(source=source, tag=tag)
    t1 = MPI.Wtime()
    #print("RECV - First I recv:", t1 - t_start, flush=True)
    
    block_shapes = env._data[0]
    message_boundary = env._data[1]
    dtype = env._data[2]
    env_data = np.empty(message_boundary[-1], dtype=dtype)
    
    requests = [MPI.REQUEST_NULL] * (len(message_boundary) - 1)
    
    t1 = MPI.Wtime()
    for i in range(len(message_boundary) - 1):
        requests[i] = comm.Irecv(env_data[message_boundary[i]:message_boundary[i+1]], source=source, tag=tag*10+i)
    
    MPI.Request.Waitall(requests)    
    t2 = MPI.Wtime()
    #print("RECV - Flat I Recv:", t2 - t1, flush=True)
    
    t1 = MPI.Wtime()
    env_data = np.split(env_data, np.cumsum([np.prod(d) for d in block_shapes])[:-1])
    t2 = MPI.Wtime()
    #print("RECV - Split data:", t2 - t1, flush=True)
    env._data = [d.reshape(block_shapes[i]) for i, d in enumerate(env_data)]
    t3 = MPI.Wtime()
    #print("RECV - Reshape data:", t3 - t2, flush=True)
    
    return env


def big_Isend_all_env(comm, env, dest, tag, max_message=2**31-1):
    env_data_orig = env._data
    t_start = MPI.Wtime()
    env_data_flat = np.concatenate([d.flatten() for d in env._data])
    t1 = MPI.Wtime()
    #print("SEND - Concatenation time:", t1 - t_start, flush=True)
    
    block_shapes = [d.shape for d in env._data]
    if env.dtype == 'complex128':
        dtype_size = 16
    elif env.dtype == 'float64':
        dtype_size = 8
    else:
        raise ValueError('dtype %s of environment not recognized' % env.dtype)

    data_size = env.size * dtype_size # np.complex128 is 16 bytes
    num_messages = 1 + data_size // max_message # Max message is 2^31 - 1 bytes
    #print('SEND - # Messages:', num_messages, flush=True)
    message_size = env.size // num_messages
    message_boundary = [message_size * i for i in range(num_messages)] + [env.size]
    assert message_boundary[-1] - message_boundary[-2] <= 2147483647

    env._data = [block_shapes] + [message_boundary] + [env.dtype]
    request = comm.isend(env, dest=dest, tag=tag)
    env._data = env_data_orig
    
    requests = [MPI.REQUEST_NULL] * num_messages
    
    for i in range(num_messages):
        requests[i] = comm.Isend(env_data_flat[message_boundary[i]:message_boundary[i+1]], dest=dest, tag=tag*10+i)
        
    MPI.Request.Waitall(requests + [request])
    
def big_Irecv_all_env(comm, source, tag):
    t_start = MPI.Wtime()
    request = comm.irecv(bytearray(1<<20), source=source, tag=tag)
    env = request.wait()
    t1 = MPI.Wtime()
    #print("RECV - First Recv:", t1 - t_start, flush=True)
    
    block_shapes = env._data[0]
    message_boundary = env._data[1]
    dtype = env._data[2]
    env_data = np.empty(message_boundary[-1], dtype=dtype)
    
    requests = [MPI.REQUEST_NULL] * (len(message_boundary) - 1)
    
    t1 = MPI.Wtime()
    for i in range(len(message_boundary) - 1):
        requests[i] = comm.Irecv(env_data[message_boundary[i]:message_boundary[i+1]], source=source, tag=tag*10+i)
    
    MPI.Request.Waitall(requests)    
    t2 = MPI.Wtime()
    #print("RECV - Flat I recv:", t2 - t1, flush=True)
    
    t1 = MPI.Wtime()
    env_data = np.split(env_data, np.cumsum([np.prod(d) for d in block_shapes])[:-1])
    t2 = MPI.Wtime()
    #print("RECV - Split data:", t2 - t1, flush=True)
    env._data = [d.reshape(block_shapes[i]) for i, d in enumerate(env_data)]
    t3 = MPI.Wtime()
    #print("RECV - Reshape data:", t3 - t2, flush=True)
    
    return env

def block_Isend_env_orig(comm, env, dest, tag):
    #env_data_orig = env._data
    #print('orig len:', len(env_data_orig), flush=True)
    block_shapes = [d.shape for d in env._data]
    if env.dtype == 'complex128':
        dtype_size = 16
    elif env.dtype == 'float64':
        dtype_size = 8
    else:
        raise ValueError('dtype %s of environment not recognized' % env.dtype)

    send_env = env.copy()
    send_env._data = [block_shapes] + [env.dtype]
    request = comm.isend(send_env, dest=dest, tag=tag)
    # We are touching the memory before the send is complete.
    #env._data = env_data_orig
    #print('orig len:', len(env_data_orig), flush=True)
    requests = [MPI.REQUEST_NULL] * env.stored_blocks
    #requests = [MPI.REQUEST_NULL] * len(env_data_orig)

    #for i in range(env.stored_blocks):
        #requests[i] = comm.Isend(np.ascontiguousarray(env._data[i]), dest=dest, tag=tag*len(block_shapes)+i)
    for i, d in enumerate(env._data):
    #for i, d in enumerate(env_data_orig):
        requests[i] = comm.Isend(np.ascontiguousarray(d), dest=dest, tag=tag*len(block_shapes)+i)

    MPI.Request.Waitall(requests + [request])
    #env._data = env_data_orig
    
    #for d in env._data:
    #    assert type(d) is np.ndarray
    return env

def block_Irecv_env_orig(comm, source, tag):
    t_start = MPI.Wtime()
    request = comm.irecv(bytearray(1<<20), source=source, tag=tag)
    env = request.wait()
    t1 = MPI.Wtime()
    #print("RECV - First Recv:", t1 - t_start, flush=True)
    
    block_shapes = env._data[0]
    dtype = env._data[1]
    env._data = []
    
    requests = [MPI.REQUEST_NULL] * len(block_shapes)
    
    t1 = MPI.Wtime()
    for i, b_s in enumerate(block_shapes):
        env._data.append(np.empty(b_s, dtype=dtype))
        requests[i] = comm.Irecv(env._data[-1], source=source, tag=tag*len(block_shapes)+i)
    
    MPI.Request.Waitall(requests)    
    t2 = MPI.Wtime()
    #print("RECV - Flat I recv:", t2 - t1, flush=True)
    
    #t1 = MPI.Wtime()
    #env_data = np.split(env_data, np.cumsum([np.prod(d) for d in block_shapes])[:-1])
    #t2 = MPI.Wtime()
    #print("RECV - Split data:", t2 - t1, flush=True)
    #env._data = [d.reshape(block_shapes[i]) for i, d in enumerate(env_data)]
    #t3 = MPI.Wtime()
    #print("RECV - Reshape data:", t3 - t2, flush=True)
    
    return env


def block_Isend_env(comm, env, dest, tag):
    env_data_orig = env._data
    block_shapes = [d.shape for d in env._data]
    if env.dtype == 'complex128':
        dtype_size = 16
    elif env.dtype == 'float64':
        dtype_size = 8
    else:
        raise ValueError('dtype %s of environment not recognized' % env.dtype)

    env._data = [block_shapes] + [env.dtype]
    buf = MPI.Alloc_mem(1<<20)
    MPI.Attach_buffer(buf)
    comm.bsend(env, dest=dest, tag=tag)
    MPI.Detach_buffer()
    MPI.Free_mem(buf)
    #comm.send(env, dest=dest, tag=tag)
    env._data = env_data_orig
    
    requests = [MPI.REQUEST_NULL] * env.stored_blocks
    
    for i in range(env.stored_blocks):
        requests[i] = comm.Isend(np.ascontiguousarray(env._data[i]), dest=dest, tag=tag*len(block_shapes)+i)
        
    MPI.Request.Waitall(requests)


# Need to allocate a buffer via MPI.Attach_buffer(mem); MPI.Detach_buffer()
# https://programtalk.com/python-examples/mpi4py.MPI.BSEND_OVERHEAD/
def block_Ibsend_env(comm, env, dest, tag):
    #env_data_orig = env._data
    block_shapes = [d.shape for d in env._data]
    if env.dtype == 'complex128':
        dtype_size = 16
    elif env.dtype == 'float64':
        dtype_size = 8
    else:
        raise ValueError('dtype %s of environment not recognized' % env.dtype)
    send_env = env.copy()
    #env._data = [block_shapes] + [env.dtype]
    send_env._data = [block_shapes] + [env.dtype]    
    buf = MPI.Alloc_mem(1<<20)
    MPI.Attach_buffer(buf)
    #comm.bsend(env, dest=dest, tag=tag)
    comm.bsend(send_env, dest=dest, tag=tag)
    MPI.Detach_buffer()
    MPI.Free_mem(buf)
    #env._data = env_data_orig
    
    requests = [MPI.REQUEST_NULL] * env.stored_blocks
    buf = MPI.Alloc_mem(env.size*16+MPI.BSEND_OVERHEAD)
    MPI.Attach_buffer(MPI.Alloc_mem(env.size*16+MPI.BSEND_OVERHEAD))
    for i in range(env.stored_blocks):
        requests[i] = comm.Ibsend(np.ascontiguousarray(env._data[i]), dest=dest, tag=tag*len(block_shapes)+i)
        
    MPI.Request.Waitall(requests)
    MPI.Detach_buffer()
    MPI.Free_mem(buf)
    
def block_Issend_env(comm, env, dest, tag):
    env_data_orig = env._data
    block_shapes = [d.shape for d in env._data]
    if env.dtype == 'complex128':
        dtype_size = 16
    elif env.dtype == 'float64':
        dtype_size = 8
    else:
        raise ValueError('dtype %s of environment not recognized' % env.dtype)

    env._data = [block_shapes] + [env.dtype]
    comm.ssend(env, dest=dest, tag=tag)
    env._data = env_data_orig
    
    requests = [MPI.REQUEST_NULL] * env.stored_blocks
    
    for i in range(env.stored_blocks):
        requests[i] = comm.Issend(np.ascontiguousarray(env._data[i]), dest=dest, tag=tag*len(block_shapes)+i)
        
    MPI.Request.Waitall(requests)
    
def block_Irecv_env(comm, source, tag):
    t_start = MPI.Wtime()
    env = comm.recv(bytearray(1<<20), source=source, tag=tag)
    #env = request.wait()
    t1 = MPI.Wtime()
    #print("RECV - First Recv:", t1 - t_start, flush=True)
    
    block_shapes = env._data[0]
    dtype = env._data[1]
    env._data = []
    
    requests = [MPI.REQUEST_NULL] * len(block_shapes)
    
    t1 = MPI.Wtime()
    for i, b_s in enumerate(block_shapes):
        env._data.append(np.empty(b_s, dtype=dtype))
        requests[i] = comm.Irecv(env._data[-1], source=source, tag=tag*len(block_shapes)+i)
    
    MPI.Request.Waitall(requests)    
    t2 = MPI.Wtime()
    #print("RECV - Flat I recv:", t2 - t1, flush=True)
    
    #t1 = MPI.Wtime()
    #env_data = np.split(env_data, np.cumsum([np.prod(d) for d in block_shapes])[:-1])
    #t2 = MPI.Wtime()
    #print("RECV - Split data:", t2 - t1, flush=True)
    #env._data = [d.reshape(block_shapes[i]) for i, d in enumerate(env_data)]
    #t3 = MPI.Wtime()
    #print("RECV - Reshape data:", t3 - t2, flush=True)
    
    return env
    
def block_Irsend_env(comm, env, dest, tag):
    env_data_orig = env._data
    block_shapes = [d.shape for d in env._data]
    if env.dtype == 'complex128':
        dtype_size = 16
    elif env.dtype == 'float64':
        dtype_size = 8
    else:
        raise ValueError('dtype %s of environment not recognized' % env.dtype)

    env._data = [block_shapes] + [env.dtype]
    comm.send(env, dest=dest, tag=tag)
    env._data = env_data_orig
    
    comm.recv(None, source=dest, tag=tag) # Dest node has posted all receives
    requests = [MPI.REQUEST_NULL] * env.stored_blocks
    
    for i in range(env.stored_blocks):
        requests[i] = comm.Irsend(np.ascontiguousarray(env._data[i]), dest=dest, tag=tag*len(block_shapes)+i)
        
    MPI.Request.Waitall(requests)
    
def block_Irrecv_env(comm, source, tag):
    t_start = MPI.Wtime()
    env = comm.recv(bytearray(1<<20), source=source, tag=tag)
    #env = request.wait()
    t1 = MPI.Wtime()
    #print("RECV - First Recv:", t1 - t_start, flush=True)
    
    block_shapes = env._data[0]
    dtype = env._data[1]
    env._data = []
    
    requests = [MPI.REQUEST_NULL] * len(block_shapes)
    
    t1 = MPI.Wtime()
    for i, b_s in enumerate(block_shapes):
        env._data.append(np.empty(b_s, dtype=dtype))
        requests[i] = comm.Irecv(env._data[-1], source=source, tag=tag*len(block_shapes)+i)
    comm.send(None, dest=source, tag=tag) # Tell source node that all receives have been posted
    
    MPI.Request.Waitall(requests)    
    t2 = MPI.Wtime()
    #print("RECV - Flat I recv:", t2 - t1, flush=True)
    
    #t1 = MPI.Wtime()
    #env_data = np.split(env_data, np.cumsum([np.prod(d) for d in block_shapes])[:-1])
    #t2 = MPI.Wtime()
    #print("RECV - Split data:", t2 - t1, flush=True)
    #env._data = [d.reshape(block_shapes[i]) for i, d in enumerate(env_data)]
    #t3 = MPI.Wtime()
    #print("RECV - Reshape data:", t3 - t2, flush=True)
    
    return env


def benchmark(send_call, send_kwargs, 
              receive_call, receive_kwargs,
              size, comm, myid):
    comm.Barrier()
    t_start = MPI.Wtime()
    if myid == 0:
        send_call(**send_kwargs)
    elif myid == 1:
        env = receive_call(**receive_kwargs)
    t_end = MPI.Wtime()
    comm.Barrier()
    
    """
    if myid == 1:
        print('Env norm^2:', npc.tensordot(env, env.conj(), axes=(['vR*', 'wR', 'vR'], ['vR', 'wR*', 'vR*'])),
              flush=True)
    """
    s, MB = 1, 0
    if myid == 0:
        MB = size / 1e6
        s = t_end - t_start
        print ('%-10d%20.2f' % (size, MB/s), flush=True)
    
    return s, MB/s
    
def osu_bw(
    BENCHMARH = "NPC MPI Bandwidth Test",
    ):

    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    numprocs = comm.Get_size()

    if myid == 0: 
        print(MPI.Get_version(), flush=True)
        print(mpi4py.get_config(), flush=True)
        print(numprocs, flush=True)

    if numprocs != 2:
        if myid == 0:
            errmsg = "This test requires exactly two processes"
        else:
            errmsg = None
        raise SystemExit(errmsg)
    
    LP, size = load_env(False)
    
    #s_buf = allocate(MAX_MSG_SIZE)
    #r_buf = allocate(MAX_MSG_SIZE)

    if myid == 0:
        print ('# %s' % (BENCHMARH,), flush=True)
    if myid == 0:
        print ('# %-8s%20s' % ("Size [B]", "Bandwidth [MB/s]"), flush=True)

    timings = []
    """
    timings.append([])
    if myid == 0:
        print('\n\nPickle Blocking Send', flush=True)
    for _ in range(10):
        timings[-1].append(benchmark(comm.send, {'obj' : LP, 'dest' : 1, 'tag' : 100},
                  comm.recv, {'source' : 0, 'tag' : 100},
                  size, comm, myid))
    if myid == 0:
        rates = [a[1] for a in timings[-1]]
        print(np.mean(rates), np.std(rates), flush=True)
    
    timings.append([])
    if myid == 0:
        print('\n\nBig Send: 2^31 - 1', flush=True)
    for _ in range(10):
        timings[-1].append(benchmark(big_send_env, {'comm' : comm, 'env' : LP, 'dest' : 1, 'tag' : 100},
                  big_recv_env, {'comm' : comm, 'source' : 0, 'tag' : 100},
                  size, comm, myid))
    if myid == 0:
        rates = [a[1] for a in timings[-1]]
        print(np.mean(rates), np.std(rates), flush=True)
    
    timings.append([])
    if myid == 0:
        print('\n\nBig Send: 2^29 - 1', flush=True)
    for _ in range(10):
        timings[-1].append(benchmark(big_send_env, {'comm' : comm, 'env' : LP, 'dest' : 1, 'tag' : 100, 'max_message' : 2**29-1},
                  big_recv_env, {'comm' : comm, 'source' : 0, 'tag' : 100},
                  size, comm, myid))
    if myid == 0:
        rates = [a[1] for a in timings[-1]]
        print(np.mean(rates), np.std(rates), flush=True)
    
    timings.append([])
    if myid == 0:
        print('\n\nBig I Send: 2^31 - 1', flush=True)
    for _ in range(10):
        timings[-1].append(benchmark(big_Isend_env, {'comm' : comm, 'env' : LP, 'dest' : 1, 'tag' : 100},
                  big_Irecv_env, {'comm' : comm, 'source' : 0, 'tag' : 100},
                  size, comm, myid))
    if myid == 0:
        rates = [a[1] for a in timings[-1]]
        print(np.mean(rates), np.std(rates), flush=True)
    
    timings.append([])
    if myid == 0:
        print('\n\nBig I Send: 2^29 - 1', flush=True)
    for _ in range(10):
        timings[-1].append(benchmark(big_Isend_env, {'comm' : comm, 'env' : LP, 'dest' : 1, 'tag' : 100, 'max_message' : 2**29-1},
                  big_Irecv_env, {'comm' : comm, 'source' : 0, 'tag' : 100},
                  size, comm, myid))
    if myid == 0:
        rates = [a[1] for a in timings[-1]]
        print(np.mean(rates), np.std(rates), flush=True)
    
    timings.append([])
    if myid == 0:
        print('\n\nBig I Send All: 2^31 - 1', flush=True)
    for _ in range(10):
        timings[-1].append(benchmark(big_Isend_all_env, {'comm' : comm, 'env' : LP, 'dest' : 1, 'tag' : 100},
                  big_Irecv_all_env, {'comm' : comm, 'source' : 0, 'tag' : 100},
                  size, comm, myid))
    if myid == 0:
        rates = [a[1] for a in timings[-1]]
        print(np.mean(rates), np.std(rates), flush=True)
    
    timings.append([])
    if myid == 0:
        print('\n\nBig I Send All: 2^29 - 1', flush=True)
    for _ in range(10):
        timings[-1].append(benchmark(big_Isend_all_env, {'comm' : comm, 'env' : LP, 'dest' : 1, 'tag' : 100, 'max_message' : 2**29-1},
                  big_Irecv_all_env, {'comm' : comm, 'source' : 0, 'tag' : 100},
                  size, comm, myid))
    if myid == 0:
        rates = [a[1] for a in timings[-1]]
        print(np.mean(rates), np.std(rates), flush=True)
    """
    
    timings.append([])
    if myid == 0:
        print('\n\nBlock I Send Orig All', flush=True)
    for _ in range(10):
        timings[-1].append(benchmark(block_Isend_env_orig, {'comm' : comm, 'env' : LP, 'dest' : 1, 'tag' : 100},
                  block_Irecv_env_orig, {'comm' : comm, 'source' : 0, 'tag' : 100},
                  size, comm, myid))
    if myid == 0:
        rates = [a[1] for a in timings[-1]]
        print(np.mean(rates), np.std(rates), flush=True)
    
    
    timings.append([])
    if myid == 0:
        print('\n\nBlock I Send All', flush=True)
    for _ in range(10):
        timings[-1].append(benchmark(block_Isend_env, {'comm' : comm, 'env' : LP, 'dest' : 1, 'tag' : 100},
                  block_Irecv_env, {'comm' : comm, 'source' : 0, 'tag' : 100},
                  size, comm, myid))
    if myid == 0:
        rates = [a[1] for a in timings[-1]]
        print(np.mean(rates), np.std(rates), flush=True)
        
    timings.append([])
    if myid == 0:
        print('\n\nBlock I Bsend All', flush=True)
    for _ in range(10):
        timings[-1].append(benchmark(block_Ibsend_env, {'comm' : comm, 'env' : LP, 'dest' : 1, 'tag' : 100},
                  block_Irecv_env, {'comm' : comm, 'source' : 0, 'tag' : 100},
                  size, comm, myid))
    if myid == 0:
        rates = [a[1] for a in timings[-1]]
        print(np.mean(rates), np.std(rates), flush=True)
    
    timings.append([])
    if myid == 0:
        print('\n\nBlock I Ssend All', flush=True)
    for _ in range(10):
        timings[-1].append(benchmark(block_Issend_env, {'comm' : comm, 'env' : LP, 'dest' : 1, 'tag' : 100},
                  block_Irecv_env, {'comm' : comm, 'source' : 0, 'tag' : 100},
                  size, comm, myid))
    if myid == 0:
        rates = [a[1] for a in timings[-1]]
        print(np.mean(rates), np.std(rates), flush=True)
        
    timings.append([])
    if myid == 0:
        print('\n\nBlock I Rsend All', flush=True)
    for _ in range(10):
        timings[-1].append(benchmark(block_Irsend_env, {'comm' : comm, 'env' : LP, 'dest' : 1, 'tag' : 100},
                  block_Irrecv_env, {'comm' : comm, 'source' : 0, 'tag' : 100},
                  size, comm, myid))
    if myid == 0:
        rates = [a[1] for a in timings[-1]]
        print(np.mean(rates), np.std(rates), flush=True)
    
    """
    if myid == 0:
        print('Big Send - 2^31 - 1', flush=True)
    comm.Barrier()
    if myid == 0:
        t_start = MPI.Wtime()
        big_Isend_env(comm, LP, dest=1, tag=100)
        t_end = MPI.Wtime()
    elif myid == 1:
        big_Irecv_env(comm, source=0, tag=100)
    comm.Barrier()
    
    if myid == 0:
        MB = size / 1e6
        s = t_end - t_start
        print ('%-10d%20.2f\n\n' % (size, MB/s), flush=True)
    """
    
    """
    if myid == 0:
        print('Big Send - 2^29 - 1', flush=True)
    comm.Barrier()
    if myid == 0:
        t_start = MPI.Wtime()
        big_Isend_env(comm, LP, dest=1, tag=100, max_message=2**29-1)
        t_end = MPI.Wtime()
    elif myid == 1:
        big_Irecv_env(comm, source=0, tag=100)
    comm.Barrier()
    
    if myid == 0:
        MB = size / 1e6
        s = t_end - t_start
        print ('%-10d%20.2f\n\n' % (size, MB/s), flush=True)    
    """
def allocate(n):
    try:
        import mmap
        return mmap.mmap(-1, n)
    except (ImportError, EnvironmentError):
        try:
            from numpy import zeros
            return zeros(n, 'B')
        except ImportError:
            from array import array
            return array('B', [0]) * n


if __name__ == '__main__':
    osu_bw()
