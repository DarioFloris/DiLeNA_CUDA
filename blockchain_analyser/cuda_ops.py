from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


@cuda.jit
def compute_degree_distribution(n, count, percentage):
    """
    - n                 number of vertices
    - count             input cudf series
    - percentage        output cudf series
    """
    i = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    
    if i < count.size:
        percentage[i] = (count[i] / n) * 100

@cuda.jit
def undirected_adj_list(nodes, src, dst, out):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    dx = cuda.blockDim.x
    tid = dx * bx + tx
    pos = 0
    if tid < len(nodes):
        for j in range(len(src)):
            if nodes[tid] == src[j]:
                out[tid, pos] = dst[j]
                pos += 1
            elif nodes[tid] == dst[j]:
                out[tid, pos] = src[j]
                pos += 1   

@cuda.jit
def directed_adj_list(nodes, src, dst, out):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    dx = cuda.blockDim.x
    tid = dx * bx + tx
    pos = 0
    if tid < len(nodes):
        node = nodes[tid]
        for j in range(len(src)):
            if node == src[j]:
                out[tid, pos] = dst[j]
                pos += 1

@cuda.jit
def reciprocal_count(A, nodes, src, dst, M, N, out):
    ty = cuda.threadIdx.y; tx = cuda.threadIdx.x
    by = cuda.blockIdx.y; bx = cuda.blockIdx.x
    dy = cuda.blockDim.y; dx = cuda.blockDim.x
    row = dy * by + ty
    column = dx * bx + tx
    
    if row < M and column < N and A[row, column] != -1:
        ngbr = A[row, column]
        node = nodes[row]
        for j in range(len(src)):
            if ngbr == src[j] and node == dst[j]:
                cuda.atomic.add(out, row, 1)
                  
@cuda.jit
def undirected_ngbr_edges(A, src, dst, M, N, out):
    ty = cuda.threadIdx.y; tx = cuda.threadIdx.x
    by = cuda.blockIdx.y; bx = cuda.blockIdx.x
    dy = cuda.blockDim.y; dx = cuda.blockDim.x
    row = dy * by + ty
    column = dx * bx + tx

    if row < M and column < N:
        ngbr = A[row, column]
        for j in range(N):
            second_ngbr = A[row, j]
            if ngbr != -1 and second_ngbr != -1 and ngbr != second_ngbr:
                for k in range(len(src)):
                    if ngbr == src[k] and second_ngbr == dst[k]:
                        cuda.atomic.add(out, row, 1)
                    elif ngbr == dst[k] and second_ngbr == src[k]:
                        cuda.atomic.add(out, row, 1)

@cuda.jit
def directed_ngbr_edges(A, src, dst, M, N, out):
    ty = cuda.threadIdx.y; tx = cuda.threadIdx.x
    by = cuda.blockIdx.y; bx = cuda.blockIdx.x
    dy = cuda.blockDim.y; dx = cuda.blockDim.x
    row = dy * by + ty
    column = dx * bx + tx
    
    if row < M and column < N and A[row, column] != -1:
        ngbr = A[row, column]
        j = 0
        while j < N:
            second_ngbr = A[row, j]
            if A[row, j] != -1 and ngbr != second_ngbr:
                k = 0
                while k < len(src):
                    if ngbr == src[k] and second_ngbr == dst[k]:
                        cuda.atomic.add(out, row, 1)
                    k += 1
            j += 1

@cuda.jit
def lcc_directed(nodes, edges, df_degree, recip, lcc_array):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < len(nodes):
        lcc = 0.0
        item = edges[tid]
        if item > 0:
            node_deg = df_degree[tid]
            lcc = item / (node_deg * (node_deg - 1) - (2 * recip[tid]))
        cuda.atomic.add(lcc_array, 0, lcc)

@cuda.jit
def lcc_undirected(nodes, edges, df_degree, lcc_array):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < len(nodes):
        lcc = 0.0
        item = edges[tid]
        if item > 0:
            node_deg = df_degree[tid]
            lcc = item / (node_deg * (node_deg - 1))
        cuda.atomic.add(lcc_array, 0, lcc)

@cuda.jit
def gnp_erdos_renyi(p, rng_states, M, N, matrix):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = 0
    if tid < M:
        for _ in range(N):
            rnd = xoroshiro128p_uniform_float32(rng_states, tid)
            if rnd <= p:
                matrix[tid, pos] = 1
            pos += 1


@cuda.jit
def align(src, const):
    tid = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if tid < len(src):
        src[tid] = src[tid] + const