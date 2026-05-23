#include "compiled.cuh"
#include "ibgda_device.cuh"

namespace deep_ep::legacy {

__global__ void ping_pong_nvshmem_kernel(
    int64_t* sym_mailbox,
    const int rank,
    const int peer_rank,
    int64_t* timestamps,
    const int num_iters) {

    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    if (peer_rank < 0)
        return;

    const bool is_pinger = (rank < peer_rank);

    for (int i = 0; i < num_iters; ++i) {
        const int64_t val = static_cast<int64_t>(i + 1);

        if (is_pinger) {
            timestamps[i * 2] = clock64();
            nvshmem_int64_p(&sym_mailbox[peer_rank], val, peer_rank);
            nvshmem_int64_wait_until(&sym_mailbox[rank], NVSHMEM_CMP_GE, val);
            timestamps[i * 2 + 1] = clock64();
        } else {
            timestamps[i * 2] = clock64();
            nvshmem_int64_wait_until(&sym_mailbox[rank], NVSHMEM_CMP_GE, val);
            nvshmem_int64_p(&sym_mailbox[peer_rank], val, peer_rank);
            timestamps[i * 2 + 1] = clock64();
        }
    }
}

void launch_ping_pong_nvshmem(
    void* rdma_buffer_ptr,
    const int rank,
    const int peer_rank,
    int64_t* timestamps,
    const int num_iters,
    const cudaStream_t& stream) {

    if (peer_rank < 0)
        return;

    auto* sym_mailbox = static_cast<int64_t*>(rdma_buffer_ptr);
    cudaMemsetAsync(&sym_mailbox[rank], 0, sizeof(int64_t), stream);
    cudaStreamSynchronize(stream);

    ping_pong_nvshmem_kernel<<<1, 1, 0, stream>>>(
        sym_mailbox, rank, peer_rank, timestamps, num_iters);
}

}  // namespace deep_ep::legacy
