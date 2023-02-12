# http://mvapich.cse.ohio-state.edu/benchmarks/

import mpi4py
from mpi4py import MPI

def osu_bw(
    BENCHMARH = "MPI Bandwidth Test",
    skip = 10,
    loop = 100,
    window_size = 64,
    skip_large = 2,
    loop_large = 20,
    window_size_large = 64,
    large_message_size = 2**22,
    MAX_MSG_SIZE = 1<<32,
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

    s_buf = allocate(MAX_MSG_SIZE)
    r_buf = allocate(MAX_MSG_SIZE)

    if myid == 0:
        print ('# %s' % (BENCHMARH,))
    if myid == 0:
        print ('# %-8s%20s' % ("Size [B]", "Bandwidth [MB/s]"))

    message_sizes = [2**i for i in range(32)]
    for size in message_sizes:
        if size > MAX_MSG_SIZE:
            break
        if size > large_message_size:
            skip = skip_large
            loop = loop_large
            window_size = window_size_large

        iterations = list(range(loop+skip))
        window_sizes = list(range(window_size))
        requests = [MPI.REQUEST_NULL] * window_size
        #
        comm.Barrier()
        if myid == 0:
            #s_msg = [s_buf, size, MPI.BYTE]
            #r_msg = [r_buf,    4, MPI.BYTE]
            s_msg = allocate(size)
            for i in iterations:
                if i == skip:
                    t_start = MPI.Wtime()
                for j in window_sizes:
                    requests[j] = comm.isend(s_msg, dest=1, tag=100)
                MPI.Request.Waitall(requests)
                comm.recv(source=1, tag=101)
            t_end = MPI.Wtime()
        elif myid == 1:
            #s_msg = [s_buf,    4, MPI.BYTE]
            r_msg = [r_buf, size, MPI.BYTE]
            s_msg = allocate(4)
            for i in iterations:
                for j in window_sizes:
                    requests[j] = comm.irecv(r_buf, source=0, tag=100)
                MPI.Request.Waitall(requests)
                comm.send(s_msg, dest=0, tag=101)
        #
        if myid == 0:
            MB = size / 1e6 * loop * window_size
            s = t_end - t_start
            print ('%-10d%20.2f' % (size, MB/s), flush=True)


def allocate(n):
    from numpy import zeros
    return zeros(n, 'B')
    
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
