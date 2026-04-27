"""
CuTeDSL Dense GEMM with Pair-UMMA + TMA Store:

| Parameter              | Value         |
|------------------------|---------------|
| MMA Instruction Shape  | (128, 256, 16)|
| MMA Tiler             | (256, 256, 64)|
| Threads per CTA        | 128           |
| Pipeline Stages        | 7 (AB), 1 (acc)|
| Cluster Shape          | (2, 1) - default |
| CtaGroup               | TWO (pair-UMMA) |
| TMA Store              | Enabled |

Step 1: Added cluster support for parallel CTA execution.
Step 2: Added pair-UMMA (CtaGroup.TWO) for 2-CTA MMA.
Step 3: Added TMA Store for direct SMEM->GMEM stores.
"""

import argparse
from typing import Tuple, Optional, Union
from functools import lru_cache

import cutlass
import cutex as cute
import cutex.utils as utils
import cutex.pipeline as pipeline

from cutex import cpasync, tcgen05, from_dlpack, testing, runtime
import cutex.utils.sm100 as sm100_utils

import cuda.bindings.driver as cuda

# types, tiler, cluster, warp ids, stages
io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (
    256,
    256,
    64,
)

epilogue_warp_id = (0, 1, 2, 3)
mma_warp_id = 4
tma_warp_id = 5
threads_per_cta = 32 * len((mma_warp_id, tma_warp_id, *epilogue_warp_id))

epilog_sync_bar_id = 1
tmem_alloc_sync_bar_id = 2
tmem_dealloc_sync_bar_id = 3

ab_stages = 6
acc_stage = 1
num_c_stage = 2

cluster_shape_mn = (2, 1)

@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_dealloc_mbar_ptr: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mk: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nk: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    cluster_layout_vmnk: cute.Layout,
    num_mcast_ctas_a: int,
    num_mcast_ctas_b: int,
    is_a_mcast: cutlass.Constexpr,
    is_b_mcast: cutlass.Constexpr,
    num_tma_producer: cutlass.Constexpr,
    cta_tile_shape_mnk: cutlass.Constexpr,
    c_layout: cutlass.Constexpr,
    epi_tiler,
    use_2cta_instrs: cutlass.Constexpr,
    c_smem_layout,
    tma_atom_c,
    mC_mn: cute.Tensor,
    tile_sched_params: utils.PersistentTileSchedulerParams,
):

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, bidz = cute.arch.block_idx()

    tile_sched = utils.StaticPersistentTileScheduler.create(
        tile_sched_params,
        cute.arch.block_idx(),
        cute.arch.grid_dim(),
    )
    work_tile = tile_sched.initial_work_tile_info()

    mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
    is_leader_cta = mma_tile_coord_v == 0

    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
        cta_rank_in_cluster
    )

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    sC = smem.allocate_tensor(
        element_type=io_dtype,
        layout=c_smem_layout.outer,
        byte_alignment=128,
        swizzle=c_smem_layout.inner,
    )

    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=0,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
        is_two_cta=use_2cta_instrs,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
    )

    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stage))
    num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake, arch="sm_100")
    tmem.allocate(num_tmem_alloc_cols)

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)

    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_c)

    num_tma_copy_bytes = cute.size_in_bytes(
        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))

    num_tma_copy_bytes *= cute.size(tiled_mma.thr_id.shape)

    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            num_tma_producer,
        ),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cluster_layout_vmnk,
    ).make_participants()

    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            (2 if use_2cta_instrs else 1) * len(epilogue_warp_id),
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cluster_layout_vmnk,
    )
    acc_producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, acc_stage
    )
    acc_consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, acc_stage
    )

    c_producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, threads_per_cta
    )
    c_pipeline = pipeline.PipelineTmaStore.create(
        num_stages=num_c_stage, producer_group=c_producer_group
    )

    gA = cute.local_tile(
        mA_mk, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None)
    )

    gB = cute.local_tile(
        mB_nk, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None)
    )

    gC_mn = cute.local_tile(
        mC_mn, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None)
    )

    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

    tCgA = thr_mma.partition_A(gA)

    tCgB = thr_mma.partition_B(gB)

    tCgC = thr_mma.partition_C(gC_mn)

    tCrA = tiled_mma.make_fragment_A(sA)

    tCrB = tiled_mma.make_fragment_B(sB)

    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])

    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    a_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
    )
    b_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
    )
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        block_in_cluster_coord_vmnk[2],
        a_cta_layout,
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        block_in_cluster_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)


    epi_smem_layout = cute.slice_(c_smem_layout, (None, None, 0))

    tmem_copy_atom = sm100_utils.get_tmem_load_op(
        cta_tile_shape_mnk,
        c_layout,
        io_dtype,
        acc_dtype,
        epi_tiler,
        use_2cta_instrs,
    )

    tCtAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tiler)
    tmem_tiled_copy = tcgen05.make_tmem_copy(
        tmem_copy_atom, tCtAcc_epi[None, None, 0, 0]
    )
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    tTR_tAcc = tmem_thr_copy.partition_S(tCtAcc_epi)

    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

    tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None)], epi_tiler)

    tTR_gC = tmem_thr_copy.partition_D(tCgC_epi)

    tTR_rAcc = cute.make_rmem_tensor(
        tTR_gC[(None, None, None, 0, 0, 0, 0)].shape, acc_dtype
    )
    tTR_rC = cute.make_rmem_tensor(
        tTR_gC[(None, None, None, 0, 0, 0, 0)].shape, io_dtype
    )

    copy_atom_r2s = sm100_utils.get_smem_store_op(
        c_layout, io_dtype, acc_dtype, tmem_tiled_copy
    )
    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tmem_tiled_copy)
    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)

    tRS_rC = tiled_copy_r2s.retile(tTR_rC)

    tRS_sC = thr_copy_r2s.partition_D(sC)

    bSG_sC, bSG_gC = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )

    a_full_mcast_mask = None
    b_full_mcast_mask = None
    if cutlass.const_expr(is_a_mcast or is_b_mcast or use_2cta_instrs):
        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

    num_k_tiles = cute.size(gA, mode=[3])

    if warp_idx == tma_warp_id:
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_coord_mnk = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                None,
            )

            tAgA_slice = tAgA[(None, mma_coord_mnk[0], mma_coord_mnk[2])]

            tBgB_slice = tBgB[(None, mma_coord_mnk[1], mma_coord_mnk[2])]

            ab_producer.reset()

            for k_tile in cutlass.range(num_k_tiles, unroll=1):
                ab_empty = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=a_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=b_full_mcast_mask,
                )
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        ab_producer.tail()

    if warp_idx == mma_warp_id:
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_coord_mnk = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                None,
            )

            if is_leader_cta:
                acc_pipeline.producer_acquire(acc_producer_state)

            ab_consumer.reset()

            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

            for k_tile_idx in cutlass.range(num_k_tiles):
                if is_leader_cta:
                    ab_full = ab_consumer.wait_and_advance()
                    num_k_blocks = cute.size(tCrA, mode=[2])
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, ab_full.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    ab_full.release()

            if is_leader_cta:
                acc_pipeline.producer_commit(acc_producer_state)
            acc_producer_state.advance()

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        acc_pipeline.producer_tail(acc_producer_state)

    if warp_idx < mma_warp_id:
        epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=epilog_sync_bar_id,
            num_threads=32 * len(epilogue_warp_id),
        )

        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_coord_mnk = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                None,
            )

            acc_pipeline.consumer_wait(acc_consumer_state)

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

            bSG_gC_tile = bSG_gC[
                None, None, None, mma_coord_mnk[0], mma_coord_mnk[1]
            ]
            bSG_gC_tile = cute.group_modes(bSG_gC_tile, 1, cute.rank(bSG_gC_tile))

            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            for subtile_idx in cutlass.range(subtile_cnt):
                tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                cute.copy(tmem_tiled_copy, tTR_tAcc_mn, tTR_rAcc)

                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                tRS_rC.store(acc_vec.to(io_dtype))

                num_tiles_executed = tile_sched.num_tiles_executed
                num_prev_subtiles = num_tiles_executed * subtile_cnt
                c_buffer = (num_prev_subtiles + subtile_idx) % num_c_stage
                cute.copy(
                    tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)]
                )

                cute.arch.fence_proxy("async.shared", space="cta")

                epilog_sync_barrier.arrive_and_wait()

                if warp_idx == 0:
                    cute.copy(
                        tma_atom_c,
                        bSG_sC[(None, c_buffer)],
                        bSG_gC_tile[(None, subtile_idx)],
                    )

                    c_pipeline.producer_commit()
                    c_pipeline.producer_acquire()

                epilog_sync_barrier.arrive_and_wait()


            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

        c_pipeline.producer_tail()

        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

# tiled_mma, c_layout/epi_tile, smem_layouts, tma_atoms, tma_tensors, tile_scheduler
@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):

    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        io_dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        acc_dtype,
        tcgen05.CtaGroup.TWO,
        mma_tiler_mnk[:2],
    )

    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a.element_type,
        ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b.element_type,
        ab_stages,
    )
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    c_layout = utils.LayoutEnum.from_tensor(c) # LayoutEnum.ROW_MAJOR
    epi_tile = (cute.make_layout(128), cute.make_layout(64))

    c_smem_layout = sm100_utils.make_smem_layout_epi(
        io_dtype,
        c_layout,
        epi_tile,
        num_c_stage,
    )
    epi_smem_layout = cute.select(c_smem_layout, mode=[0, 1])

    tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        c,
        epi_smem_layout,
        epi_tile,
    )

    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((*cluster_shape_mn, 1)),
        (tiled_mma.thr_id.shape,),
    )

    num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2])
    num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1])
    is_a_mcast = num_mcast_ctas_a > 1
    is_b_mcast = num_mcast_ctas_b > 1

    num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1

    a_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma.thr_id)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        a,
        a_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(cluster_shape_mn, tiled_mma.thr_id)
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        b_op,
        b,
        b_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )

    use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

    cluster_shape_mnl = (*cluster_shape_mn, 1)

    cta_tile_shape_mnk = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )

    c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
    gc = cute.zipped_divide(c, tiler=c_shape)
    num_ctas_mn = gc[(0, (None, None))].shape
    num_ctas_mnl = (*num_ctas_mn, 1)

    tile_sched_params = utils.PersistentTileSchedulerParams(
        num_ctas_mnl, cluster_shape_mnl
    )
    grid_shape = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params, max_active_clusters
    )

    print(f"7MIN tested")

    print(f"cluster_layout_vmnk shape: {cluster_layout_vmnk.shape}")
    print(f"num_mcast_ctas_a: {num_mcast_ctas_a}, num_mcast_ctas_b: {num_mcast_ctas_b}")
    print(f"is_a_mcast: {is_a_mcast}, is_b_mcast: {is_b_mcast}")
    print(f"grid_shape: {grid_shape}")

    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        a_smem_layout,
        b_smem_layout,
        cluster_layout_vmnk,
        num_mcast_ctas_a,
        num_mcast_ctas_b,
        is_a_mcast,
        is_b_mcast,
        num_tma_producer,
        cta_tile_shape_mnk,
        c_layout,
        epi_tile,
        use_2cta_instrs,
        c_smem_layout,
        tma_atom_c,
        tma_tensor_c,
        tile_sched_params,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        cluster=cluster_shape_mnl,
        stream=stream,
    )


def run_dense_gemm(
    mnk: Tuple[int, int, int],
    tolerance: float,
    warmup_iterations=10,
    iterations=100,
    skip_ref_check=False,
    init_mode: str = "gaussian",
    normal_mean: float = 0.0,
    normal_std: float = 1.0,
):
    global torch, cutlass_torch
    import torch
    import cutlass.torch as cutlass_torch

    print("===================================================================")
    print("Running Blackwell fp16 GEMM with Pair-UMMA + TMA Store:")
    print(f"  mnk:       {mnk}")
    print(f"  tolerance: {tolerance}")
    print(f"  init_mode: {init_mode}")
    if init_mode == "gaussian":
        print(f"  normal_mean/std: {normal_mean}/{normal_std}")
    print("===================================================================")
    print()

    m, n, k = mnk
    l = 1
    torch.manual_seed(1111)
    ab_dtype = cutlass.Float16
    c_dtype = cutlass.Float16
    a_major = "k"
    b_major = "k"
    c_major = "n"

    def make_tensors(mn, k, dtype):
        shape = (mn, k)
        t = torch.empty(*shape, dtype=torch.float32)
        if init_mode == "randint":
            t.random_(-2, 3)
        elif init_mode == "gaussian":
            t.normal_(mean=normal_mean, std=normal_std)
        else:
            raise ValueError(f"Unsupported init_mode: {init_mode}")
        return t.to(dtype=dtype, device="cuda")

    def create_tensors(l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype):
        import torch
        import cutlass.torch as cutlass_torch

        torch.manual_seed(1111)

        a_torch_cpu = make_tensors(m, k, cutlass_torch.dtype(io_dtype))
        b_torch_cpu = make_tensors(n, k, cutlass_torch.dtype(io_dtype))
        c_torch_cpu = make_tensors(m, n, cutlass_torch.dtype(io_dtype))

        a_tensor, _ = cutlass_torch.cute_tensor_like(
            a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_tensor, _ = cutlass_torch.cute_tensor_like(
            b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
            c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
        )

        return (
            a_tensor,
            b_tensor,
            c_tensor,
            a_torch_cpu,
            b_torch_cpu,
            c_torch_cpu,
            c_torch_gpu,
        )

    def generate_tensors():
        import cutlass.torch as cutlass_torch

        a_tensor, _ = cutlass_torch.cute_tensor_like(
            a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_tensor, _ = cutlass_torch.cute_tensor_like(
            b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, _ = cutlass_torch.cute_tensor_like(
            c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
        )
        return testing.JitArguments(a_tensor, b_tensor, c_tensor, current_stream)

    @lru_cache(maxsize=1)
    def compile_mm(
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr = None,
    ):
        from cutex.runtime import make_fake_stream

        stream = runtime.make_fake_stream()
        return cute.compile(host_function, a, b, c, max_active_clusters, stream)

    a_tensor, b_tensor, c_tensor, a_torch_cpu, b_torch_cpu, c_torch_cpu, c_torch_gpu = (
        create_tensors(l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype)
    )

    torch_stream = torch.cuda.current_stream()

    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    compiled_gemm = compile_mm(a_tensor, b_tensor, c_tensor, max_active_clusters)

    def compare(a_torch_cpu, b_torch_cpu, c_torch_gpu, c_dtype, tolerance):
        import torch
        import cutlass.torch as cutlass_torch

        kernel_result = c_torch_gpu.cpu()

        ref = torch.einsum(
            "mk,nk->mn",
            a_torch_cpu.to(dtype=torch.float16),
            b_torch_cpu.to(dtype=torch.float16),
        )

        _, ref_torch_gpu = cutlass_torch.cute_tensor_like(
            ref, c_dtype, is_dynamic_layout=True, assumed_align=16
        )
        ref_result = ref_torch_gpu.cpu()

        print("AssertClose")
        torch.testing.assert_close(
            kernel_result, ref_result, atol=tolerance, rtol=1e-05
        )

    if not skip_ref_check:
        compiled_gemm(a_tensor, b_tensor, c_tensor, current_stream)
        compare(a_torch_cpu, b_torch_cpu, c_torch_gpu, c_dtype, tolerance)

    workspace_count = 1
    exec_time = testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    return exec_time
