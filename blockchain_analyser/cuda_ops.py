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
def adj_list(nodes, src, dst, undirected, out):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    dx = cuda.blockDim.x
    tid = dx * bx + tx
    pos = 0
    if tid < len(nodes):
        u_node = nodes[tid]
        for j in range(len(src)):
            if u_node == src[j]:
                out[tid, pos] = dst[j]
                pos += 1
            if undirected:
                if u_node == dst[j]:
                    out[tid, pos] = src[j]
                    pos += 1

@cuda.jit
def reciprocal_count(A, nodes, src, dst, M, N, out):
    ty = cuda.threadIdx.y; tx = cuda.threadIdx.x
    by = cuda.blockIdx.y; bx = cuda.blockIdx.x
    dy = cuda.blockDim.y; dx = cuda.blockDim.x
    row = dy * by + ty
    column = dx * bx + tx
    
    if row < M and column < N and A[row, column] != -1:
        v_node = A[row, column]
        u_node = nodes[row]
        for j in range(len(src)):
            if v_node == src[j] and u_node == dst[j]:
                cuda.atomic.add(out, row, 1)


@cuda.jit
def find_uv_edges(A, src, dst, M, N, out):
    ty = cuda.threadIdx.y; tx = cuda.threadIdx.x
    by = cuda.blockIdx.y; bx = cuda.blockIdx.x
    dy = cuda.blockDim.y; dx = cuda.blockDim.x
    row = dy * by + ty
    column = dx * bx + tx
    
    if row < M and column < N and A[row, column] != -1:
        u_node = A[row, column]
        for j in range(N):
            v_node = A[row, j]
            if v_node != -1 and u_node != v_node:
                common = explore_edges(u_node, v_node, src, dst)
                cuda.atomic.add(out, row, common)
                
#            j += 1


@cuda.jit(device=True)
def explore_edges(u, v, src, dst):
    result, k = 0, 0
    while k < src.size:
        if u == src[k] and v == dst[k]:
            result += 1
        k += 1

    return result


@cuda.jit
def lcc(nodes, edges, df_degree, undirected, lcc_array):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < len(nodes):
        lcc = 0.0
        item = edges[tid]
        if item > 0:
            node_deg = df_degree[tid]
            if undirected:
                lcc = 2 * (item / (node_deg * (node_deg - 1)))
            else:
                lcc = item / (node_deg * (node_deg - 1))

        cuda.atomic.add(lcc_array, 0, lcc)

@cuda.jit
def gnp_erdos_renyi(p, rng_states, M, N, matrix):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < M:
        for pos in range(N):
            rnd = xoroshiro128p_uniform_float32(rng_states, tid)
            if rnd <= p:
                matrix[tid, pos] = 1


@cuda.jit
def align(src, const):
    tid = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if tid < len(src):
        src[tid] = src[tid] + const