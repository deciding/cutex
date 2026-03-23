# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline

# [CLUSTER] Import pipeline_init functions for cluster support
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

import cuda.bindings.driver as cuda
import cutlass.cute.testing as testing

"""
CuTeDSL Dense GEMM with Cluster and Pair-UMMA support.

This kernel demonstrates:
- Cluster support for parallel CTA execution
- Pair-UMMA (CtaGroup.TWO) for 2-CTA MMA instructions
- TMA Store for direct SMEM->GMEM stores
"""

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (
    256,
    256,
    64,
)  # [PAIR-UMMA] Changed from (128, 256, 64) to use CtaGroup.TWO
threads_per_cta = 128

# Pipeline stage configuration
ab_stages = 6  # TODO: don't hardcode this
acc_stage = 1
num_c_stage = 2  # [TMEM_STORE] Number of stages for C SMEM buffer

# Cluster configuration
cluster_shape_mn = (2, 1)

# [TMEM_STORE] Enable TMA Store
use_tma_store = True


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_dealloc_mbar_ptr: cutlass.Int64  # [PAIR-UMMA] Added for pair-UMMA
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
    # [CLUSTER] Cluster parameters for TMA multicast
    cluster_layout_vmnk: cute.Layout,
    num_mcast_ctas_a: int,
    num_mcast_ctas_b: int,
    is_a_mcast: cutlass.Constexpr,
    is_b_mcast: cutlass.Constexpr,
    num_tma_producer: cutlass.Constexpr,
    # [PAIR-UMMA] Parameters for TMEM load
    cta_tile_shape_mnk: cutlass.Constexpr,
    c_layout: cutlass.Constexpr,
    epi_tiler,
    use_2cta_instrs: cutlass.Constexpr,
    # [TMEM_STORE] New parameters for TMA Store
    c_smem_layout_staged,  # Can be None if use_tma_store is False
    tma_atom_c,  # Can be None if use_tma_store is False
    mC_mnl: cute.Tensor,
    # [PERSISTENT] Persistent tile scheduler parameters
    tile_sched_params: utils.PersistentTileSchedulerParams,
):
    # Current thread/warp/block coordinates
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, _, _ = cute.arch.block_idx()

    # [PERSISTENT] 1. Initialize the scheduler
    tile_sched = utils.StaticPersistentTileScheduler.create(
        tile_sched_params, cluster_layout_vmnk
    )
    work_tile = tile_sched.get_initial_tile()

    # [PAIR-UMMA] mma_tile_coord_v 1. for is_leader_cta, 2. for slice tiled_mma
    mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
    is_leader_cta = mma_tile_coord_v == 0

    # [CLUSTER] Get block's cluster coordinates
    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
        cta_rank_in_cluster
    )

    #
    # 1. Prepare args
    #

    # Allocate SMEM
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # [TMEM_STORE] Allocate SMEM C tensor for TMA Store
    sC = None
    if cutlass.const_expr(use_tma_store):
        sC = smem.allocate_tensor(
            element_type=io_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
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

    # Allocate all TMEM columns
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

    # Prefetch tma descriptor
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        if cutlass.const_expr(use_tma_store):
            cpasync.prefetch_descriptor(tma_atom_c)

    # Pipeline configuration
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

    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            threads_per_cta,
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cluster_layout_vmnk,
    ).make_participants()

    # [TMEM_STORE] Setup for epilogue with TMA Store
    epi_smem_layout = None
    bSG_sC, bSG_gC = None, None
    c_pipeline = None
    tiled_copy_r2s, tRS_sC = None, None
    tmem_tiled_copy, tmem_thr_copy = None, None

    if cutlass.const_expr(use_tma_store):
        epi_smem_layout = cute.slice_(c_smem_layout_staged, (None, None, 0))
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            cta_tile_shape_mnk,
            c_layout,
            io_dtype,
            acc_dtype,
            epi_tiler,
            use_2cta_instrs,
        )
        # Use dummy layout for initialization
        dummy_acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        dummy_acc = cute.make_tensor(acc_dtype, dummy_acc_shape)
        dummy_acc_epi = cute.flat_divide(dummy_acc, epi_tiler)
        tmem_tiled_copy = tcgen05.make_tmem_copy(
            tmem_copy_atom, dummy_acc_epi[None, None, 0, 0]
        )
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

        copy_atom_r2s = sm100_utils.get_smem_store_op(
            c_layout, io_dtype, acc_dtype, tmem_tiled_copy
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tmem_tiled_copy)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)

        dummy_gC = cute.make_tensor(io_dtype, mma_tiler_mnk[:2])
        dummy_gC_epi = cute.flat_divide(dummy_gC, epi_tiler)
        bSG_sC, _ = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(dummy_gC_epi, 0, 2),
        )

        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, threads_per_cta
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=num_c_stage, producer_group=c_producer_group
        )
    else:
        epi_tiler_dummy = cta_tile_shape_mnk[:2]
        dummy_acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        dummy_acc = cute.make_tensor(acc_dtype, dummy_acc_shape)
        dummy_acc_epi = cute.flat_divide(dummy_acc, epi_tiler_dummy)
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            cta_tile_shape_mnk,
            c_layout,
            io_dtype,
            acc_dtype,
            epi_tiler_dummy,
            use_2cta_instrs,
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(
            tmem_copy_atom, dummy_acc_epi[None, None, 0, 0]
        )
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    # [PERSISTENT] 2. Outer loop: Persist and process multiple tiles
    while work_tile.is_valid_tile:
        mma_coord_mnk = work_tile.get_tile_coord()

        # Partition tensors for MMA
        gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
        gC_mn = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
        
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC_mn)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc)
        tmem.allocate(num_tmem_cols)

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

        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    # [TMEM_STORE] Setup for epilogue with TMA Store
    epi_smem_layout = None
    bSG_sC, bSG_gC = None, None
    c_pipeline = None
    tiled_copy_r2s, tRS_rC, tRS_sC = None, None, None

    if cutlass.const_expr(use_tma_store):
        # Compute epi_smem_layout for TMA Store
        epi_smem_layout = cute.slice_(c_smem_layout_staged, (None, None, 0))

        # [TMEM_STORE] 1. get tmem_copy

        # TMEM load atom for accumulator
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            cta_tile_shape_mnk,
            c_layout,
            io_dtype,
            acc_dtype,
            epi_tiler,
            use_2cta_instrs,
        )
        # [TMEM_STORE] (MMA, MMA_M, MMA_N) -> (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        tCtAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tiler)
        tmem_tiled_copy = tcgen05.make_tmem_copy(
            tmem_copy_atom, tCtAcc_epi[None, None, 0, 0]
        )
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

        # [TMEM_STORE] tmem copy src: tTR_tAcc
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_tAcc = tmem_thr_copy.partition_S(tCtAcc_epi)
        # (T2R, T2R_M, T2R_N, EPI_MN)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        # [TMEM_STORE] tmem copy dst: tTR_rAcc
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        tCgC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0)], epi_tiler
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_gC = tmem_thr_copy.partition_D(tCgC_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0)].shape, acc_dtype
        )
        tTR_rC = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0)].shape, io_dtype
        )

        # [TMEM_STORE] 2. get smem_copy

        copy_atom_r2s = sm100_utils.get_smem_store_op(
            c_layout, io_dtype, acc_dtype, tmem_tiled_copy
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tmem_tiled_copy)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        # (T2R, T2R_M, T2R_N) -> (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        tRS_sC = thr_copy_r2s.partition_D(sC)

        # [TMEM_STORE] 3. get gmem copy
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0, # cluster coord
            cute.make_layout(1), # cluster layout
            cute.group_modes(sC, 0, 2), # (EPI_TILE_M, EPI_TILE_N), EPI_M, EPI_N, PIPE
            cute.group_modes(tCgC_epi, 0, 2), # (EPI_TILE_M, EPI_TILE_N), EPI_M, EPI_N
        )
        # (EPI_TILE_M, EPI_TILE_N), (EPI_M, EPI_N)
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

        # Initialize TMA store pipeline
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, threads_per_cta
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=num_c_stage, producer_group=c_producer_group
        )
    else:
        # [SIMT-STORE] Original TMEM load setup
        epi_tiler = cta_tile_shape_mnk[:2]
        tCtAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tiler)
        # tCgC shape: (MMA, MMA_M, MMA_N, RestM, RestN), indexing gives (RestM, RestN)
        gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tiler)

        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            cta_tile_shape_mnk,
            c_layout,
            io_dtype,
            acc_dtype,
            epi_tiler,
            use_2cta_instrs,
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(
            tmem_copy_atom, tCtAcc_epi[None, None, 0, 0]
        )
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

        tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
        tDgC = tmem_thr_copy.partition_D(gC_epi)
        tCrAcc = cute.make_rmem_tensor(tDgC[None, None, None, 0, 0].shape, acc_dtype)
        tCrC = cute.make_rmem_tensor(tDgC[None, None, None, 0, 0].shape, io_dtype)

    #
    # 2. Main loop
    #

    # [CLUSTER] Create multicast masks, 1. tma: whom to mcast 2. mma: whom to arrive
    # [PAIR-UMMA] Also enable for use_2cta_instrs
    a_full_mcast_mask = None
    b_full_mcast_mask = None
    if cutlass.const_expr(is_a_mcast or is_b_mcast or use_2cta_instrs):
        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        # Wait for a empty accumulator buffer
        acc_empty = acc_producer.acquire_and_advance()

        # MMA mainloop
        for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
            # Issue TMA loads
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a,
                tAgA[(None, ab_empty.count)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=a_full_mcast_mask,  # mcast
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, ab_empty.count)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=b_full_mcast_mask,  # mcast
            )

            if is_leader_cta:
                # Execute one K-block worth of MMA instructions
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

                # Signal that the A/B buffers have been consumed and are ready for the next load
                ab_full.release()

        # Signal that the accumulator is fully computed
        if is_leader_cta:
            acc_empty.commit()

        #
        # 3. Epilogue
        #

        # Release TMEM allocation lock
        tmem.relinquish_alloc_permit()

        # Wait for the accumulator buffer to be full
        acc_full = acc_consumer.wait_and_advance()

        if cutlass.const_expr(use_tma_store):
            # [TMEM_STORE] Epilogue setup for THIS tile
            tCtAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tiler)
            tTR_tAcc = tmem_thr_copy.partition_S(tCtAcc_epi)
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            tTR_rAcc = cute.make_rmem_tensor(tTR_tAcc[(None, None, None, 0)].shape, acc_dtype)
            tTR_rC = cute.make_rmem_tensor(tTR_tAcc[(None, None, None, 0)].shape, io_dtype)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)
            
            tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tiler)
            _, bSG_gC = cpasync.tma_partition(
                tma_atom_c, 0, cute.make_layout(1),
                cute.group_modes(sC, 0, 2),
                cute.group_modes(tCgC_epi, 0, 2),
            )
            bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            for subtile_idx in cutlass.range(subtile_cnt):
                # TMEM -> Register
                tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                cute.copy(tmem_tiled_copy, tTR_tAcc_mn, tTR_rAcc)

                # Apply epilogue op
                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                tRS_rC.store(acc_vec.to(io_dtype))

                # Register -> SMEM
                c_buffer = subtile_idx % num_c_stage
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline.sync(barrier_id=1)

                # TMA Store C
                if warp_idx == 0:
                    cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)])
                    c_pipeline.producer_commit()
                    c_pipeline.producer_acquire()
                pipeline.sync(barrier_id=1)

            # Wait for C store complete
            c_pipeline.producer_tail()
        else:
            # [SIMT-STORE] SIMT Epilogue for THIS tile
            tCtAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tiler)
            gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tiler)
            tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
            tDgC = tmem_thr_copy.partition_D(gC_epi)
            tCrAcc = cute.make_rmem_tensor(tDgC[None, None, None, 0, 0].shape, acc_dtype)
            tCrC = cute.make_rmem_tensor(tDgC[None, None, None, 0, 0].shape, io_dtype)
            
            simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), io_dtype)
            tDtC = cute.group_modes(tDtC, 3, cute.rank(tDtC))
            tDgC = cute.group_modes(tDgC, 3, cute.rank(tDgC))
            for i in cutlass.range(cute.size(tDtC, mode=[3])):
                cute.copy(tmem_tiled_copy, tDtC[None, None, None, i], tCrAcc)
                tCrC.store(tCrAcc.load().to(io_dtype))
                cute.copy(simt_atom, tCrC, tDgC[(None, None, None, i)])

        acc_full.release()

        # [PERSISTENT] 4. Advance and Reset state for next tile
        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)
        
        ab_producer.advance_stage_after_tile()
        ab_consumer.advance_stage_after_tile()
        acc_producer.advance_stage_after_tile()
        acc_consumer.advance_stage_after_tile()
        if cutlass.const_expr(use_tma_store):
            c_pipeline.advance_stage_after_tile()

        work_tile = tile_sched.get_next_tile()


@cute.jit
def host_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream):
    # Construct tiled MMA
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        io_dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        acc_dtype,
        tcgen05.CtaGroup.TWO,
        mma_tiler_mnk[:2],
    )

    use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

    cta_tile_shape_mnk = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )

    c_layout = utils.LayoutEnum.from_tensor(c)

    if cutlass.const_expr(use_tma_store):
        epi_tile = sm100_utils.compute_epilogue_tile_shape(
            cta_tile_shape_mnk,
            use_2cta_instrs,
            c_layout,
            io_dtype,
        )
    else:
        epi_tile = cta_tile_shape_mnk[:2]

    # Construct SMEM layouts
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, a.element_type, ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, b.element_type, ab_stages,
    )
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    c_smem_layout_staged = None
    tma_atom_c = None
    tma_tensor_c = None
    if cutlass.const_expr(use_tma_store):
        c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            io_dtype, c_layout, epi_tile, num_c_stage,
        )
        epi_smem_layout = cute.slice_(c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, epi_tile,
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
        a_op, a, a_smem_layout_one_stage, mma_tiler_mnk, tiled_mma, cluster_layout_vmnk.shape,
    )
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(cluster_shape_mn, tiled_mma.thr_id)
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        b_op, b, b_smem_layout_one_stage, mma_tiler_mnk, tiled_mma, cluster_layout_vmnk.shape,
    )

    # [PERSISTENT] Compute parameters
    tile_sched_params = utils.PersistentTileSchedulerParams(
        mnkl=(*c.layout.shape, 1),
        mma_tiler_mn=(mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape), mma_tiler_mnk[1]),
        cluster_shape_mn=cluster_shape_mn,
    )
    grid_shape = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params)

    cluster_shape_mnl = (*cluster_shape_mn, 1)
    kernel(
        tiled_mma, a_tma_atom, a_tma_tensor, b_tma_atom, b_tma_tensor,
        a_smem_layout, b_smem_layout, cluster_layout_vmnk,
        num_mcast_ctas_a, num_mcast_ctas_b, is_a_mcast, is_b_mcast, num_tma_producer,
        cta_tile_shape_mnk, c_layout, epi_tile, use_2cta_instrs,
        c_smem_layout_staged, tma_atom_c, tma_tensor_c if use_tma_store else c,
        tile_sched_params,
    ).launch(
        grid=grid_shape, block=(threads_per_cta, 1, 1), cluster=cluster_shape_mnl, stream=stream,
    )


def run_dense_gemm(
    mnk: Tuple[int, int, int],
    tolerance: float,
    warmup_iterations=10,
    iterations=100,
    skip_ref_check=False,
):
    global torch, cutlass_torch
    import torch
    import cutlass.torch as cutlass_torch

    m, n, k = mnk
    l = 1
    torch.manual_seed(1111)
    ab_dtype = cutlass.Float16
    c_dtype = cutlass.Float16

    def make_tensors(mn, k, dtype):
        return torch.empty(mn, k, dtype=torch.int32).random_(-2, 2).to(dtype=dtype, device="cuda")

    def create_tensors():
        a_torch_cpu = make_tensors(m, k, cutlass_torch.dtype(io_dtype))
        b_torch_cpu = make_tensors(n, k, cutlass_torch.dtype(io_dtype))
        c_torch_cpu = make_tensors(m, n, cutlass_torch.dtype(io_dtype))

        a_tensor, _ = cutlass_torch.cute_tensor_like(a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16)
        b_tensor, _ = cutlass_torch.cute_tensor_like(b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16)
        c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16)

        return a_tensor, b_tensor, c_tensor, a_torch_cpu, b_torch_cpu, c_torch_cpu, c_torch_gpu

    a_tensor, b_tensor, c_tensor, a_torch_cpu, b_torch_cpu, c_torch_cpu, c_torch_gpu = create_tensors()

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled_gemm = cute.compile(host_function, a_tensor, b_tensor, c_tensor, current_stream)

    if not skip_ref_check:
        compiled_gemm(a_tensor, b_tensor, c_tensor, current_stream)
        kernel_result = c_torch_gpu.cpu()
        ref = torch.einsum("mk,nk->mn", a_torch_cpu.to(dtype=torch.float32), b_torch_cpu.to(dtype=torch.float32))
        torch.testing.assert_close(kernel_result, ref.to(dtype=kernel_result.dtype), atol=tolerance, rtol=1e-05)

    exec_time = testing.benchmark(
        compiled_gemm,
        workspace_generator=lambda: testing.JitArguments(*create_tensors()[:3], current_stream),
        workspace_count=1,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    return exec_time


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str):
        try:
            return [int(x.strip()) for x in s.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    from cuda.bindings import driver as cu_driver

    cu_driver.cuInit(0)
    err, device_count = cu_driver.cuDeviceGetCount()
    if err != cu_driver.CUresult.CUDA_SUCCESS or device_count < 1:
        raise RuntimeError("A GPU is required to run this example")

    parser = argparse.ArgumentParser(
        description="Blackwell fp16 GEMM with Pair-UMMA + TMA Store"
    )
    parser.add_argument(
        "--mnk",
        type=parse_comma_separated_ints,
        default=[8192, 8192, 8192],
        help="MNK dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    args = parser.parse_args()
    if len(args.mnk) != 3:
        parser.error("--mnk must contain exactly 3 values")
    if args.mnk[0] % mma_tiler_mnk[0] != 0 or args.mnk[1] % mma_tiler_mnk[1] != 0:
        parser.error("m n must be divisible by mma_tiler_mn")

    run_dense_gemm(
        args.mnk,
        args.tolerance,
    )
    print("PASS")
