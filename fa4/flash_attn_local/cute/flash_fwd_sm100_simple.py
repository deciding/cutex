# Simplified Flash Attention Forward for SM100 (Blackwell)
# Only supports: batch=4, heads=16, seqlen=8192, head_dim=128, causal=False, bfloat16

import enum
import math
from typing import Type, Tuple, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL
import flash_attn.cute.pipeline as pipeline_custom

from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import blackwell_helpers as sm100_utils
from quack import copy_utils, layout_utils
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    StaticPersistentTileScheduler,
)


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()
    TmemPtr = enum.auto()


class FlashAttentionForwardSm100Simple:
    def __init__(
        self,
        head_dim: int = 128,
        is_causal: bool = False,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: int = 2,
    ):
        hdim_multiple_of = 16
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        self.head_dim_v_padded = self.head_dim_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = q_stage
        self.use_2cta_instrs = False
        self.is_causal = is_causal
        self.is_local = False
        self.is_varlen_q = False

        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f

        self.cta_group_size = 2 if self.use_2cta_instrs else 1
        self.cta_tiler = (
            self.q_stage * m_block_size,
            n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_qk = (
            self.cta_group_size * m_block_size,
            n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_pv = (
            self.cta_group_size * m_block_size,
            self.head_dim_v_padded,
            n_block_size,
        )
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)

        self.softmax_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.epilogue_warp_id = 5
        self.load_warp_id = 6

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax_warp_ids,
                self.mma_warp_id,
                self.epilogue_warp_id,
                self.load_warp_id,
            )
        )

        self.tmem_s_offset = [0, self.n_block_size]
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded

        self.num_regs_softmax = 192
        self.num_regs_other = 48

    def _setup_attributes(self):
        smem_size_q = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_padded
            * self.q_dtype.width
            // 8
        )
        smem_size_o = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_v_padded
            * self.o_dtype.width
            // 8
        )
        smem_size_q_o = smem_size_q + smem_size_o
        smem_size_k_per_stage = (
            self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        )
        smem_size_v_per_stage = (
            self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        )
        smem_size_kv_per_stage = max(smem_size_k_per_stage, smem_size_v_per_stage)
        kv_stage = (224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage
        self.kv_stage = max(kv_stage, 1)
        self.s_stage = self.q_stage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors=None,
        aux_tensors=None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type

        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 3, 2, 0]))
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 3, 2, 0]))
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[1, 0, 2, 3]))
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=[1, 3, 2, 0]))

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE
        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN

        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K

        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )

        self.tma_copy_bytes = {}
        for name, mX, layout in [
            ("Q", mQ, sQ_layout),
            ("K", mK, sK_layout),
            ("V", mV, sV_layout),
        ]:
            self.tma_copy_bytes[name] = (
                cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
                * self.cta_group_size
            )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
            cta_layout_vmnk.shape,
        )

        tma_atom_O, mO = cpasync.make_tiled_tma_atom(
            tma_store_op, mO, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
        )

        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            1,
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[0],
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=True,
            lpt=False,
            is_split_kv=False,
            cluster_shape_mn=self.cluster_shape_mn,
        )
        tile_sched_params = StaticPersistentTileScheduler.to_underlying_arguments(
            tile_sched_args
        )
        grid_dim = StaticPersistentTileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout)
        sQ_size = cute.cosize(sQ_layout)

        @cute.struct
        class SharedStorage:
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
            tmem_dealloc_mbar_ptr: Int64
            tmem_holding_buf: Int32
            sO: cute.struct.Align[cute.struct.MemRange[self.o_dtype, sO_size], 1024]
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sQ_size], 1024]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)], 1024
            ]

        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk
            if cute.size(self.cluster_shape_mnk) > 1
            else None,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_O)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = 0
        is_leader_cta = True

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=cute.arch.WARP_SIZE * 5,
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        ThreadCooperativeGroup = partial(
            pipeline.CooperativeGroup, pipeline.Agent.Thread
        )
        mma_warp = ThreadCooperativeGroup(1)
        tma_warp = ThreadCooperativeGroup(1)
        softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * 4)
        epilogue_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)

        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_s_p = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=softmax_threads,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=epilogue_threads,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_o_epi = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_O_epi.data_ptr(),
            num_stages=self.q_stage,
            producer_group=epilogue_threads,
            consumer_group=epilogue_threads,
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer
        )
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        tP_width_ratio = Float32.width // self.v_dtype.width
        tP_stage_stride = (
            self.tmem_o_offset[1] - self.tmem_o_offset[0]
        ) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_o_offset[0] * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )

        block_info = BlockInfo(
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            False,
            None,
            None,
            1,
        )

        TileSchedulerCls = partial(
            StaticPersistentTileScheduler.create, tile_sched_params
        )

        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # ============================================================
        # LOAD WARP: Load Q, K, V from GMEM to SMEM
        # ============================================================
        if const_expr(warp_idx == self.load_warp_id):
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_kv,
                thr_mma_qk,
                thr_mma_pv,
                TileSchedulerCls,
            )

        # ============================================================
        # MMA WARP: Compute Q*K^T and P*V
        # ============================================================
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStS,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p,
                pipeline_o_acc,
                is_leader_cta,
                TileSchedulerCls,
            )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ============================================================
        # EPILOGUE WARP: Store O to GMEM
        # ============================================================
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.epilogue(
                mO,
                sO,
                tma_atom_O,
                pipeline_o_acc,
                pipeline_o_epi,
                TileSchedulerCls,
            )

        # ============================================================
        # SOFTMAX WARPS: Compute softmax
        # ============================================================
        if (
            warp_idx >= self.softmax_warp_ids[0]
            and warp_idx <= self.softmax_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.softmax(
                softmax_scale_log2,
                thr_mma_qk,
                tStS,
                pipeline_s_p,
                TileSchedulerCls,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q,
        pipeline_kv,
        thr_mma_qk,
        thr_mma_pv,
        TileSchedulerCls,
    ):
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        producer_phase = Int32(1)

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            tiler_gQ = ((self.q_stage * self.m_block_size), self.head_dim_padded)
            gQ = cute.local_tile(
                mQ[None, None, head_idx, batch_idx], tiler_gQ, (m_block, 0)
            )
            gQ = layout_utils.select(
                cute.flat_divide(gQ, (self.m_block_size,)), mode=[0, 2, 1]
            )

            gK = cute.local_tile(
                mK[None, None, head_idx, batch_idx],
                cute.select(self.mma_tiler_qk, mode=[1, 2]),
                (None, 0),
            )
            gV = cute.local_tile(
                mV[None, None, head_idx, batch_idx],
                cute.select(self.mma_tiler_pv, mode=[1, 2]),
                (0, None),
            )

            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)

            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )

            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )
            load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), tKgK, tKsK
            )
            load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V, 0, cute.make_layout(1), tVgV, tVsV
            )

            n_block_max = cute.ceil_div(mK.shape[0], self.n_block_size)

            pipeline_q.producer_acquire_w_index_phase(0, producer_phase)
            tma_bar_ptr = pipeline_q.sync_object_full.get_barrier(0)
            load_Q_fn(src_idx=0, dst_idx=0, tma_bar_ptr=tma_bar_ptr)
            if self.q_stage == 2:
                pipeline_q.producer_acquire_w_index_phase(1, producer_phase)
                tma_bar_ptr = pipeline_q.sync_object_full.get_barrier(1)
                load_Q_fn(src_idx=1, dst_idx=1, tma_bar_ptr=tma_bar_ptr)
            producer_phase ^= 1

            for i in cutlass.range(n_block_max):
                pipeline_kv.producer_acquire()
                tma_bar_ptr = pipeline_kv.sync_object_full.get_barrier()
                load_K_fn(src_idx=0, dst_idx=0, tma_bar_ptr=tma_bar_ptr)
                load_V_fn(src_idx=0, dst_idx=0, tma_bar_ptr=tma_bar_ptr)
                pipeline_kv.producer_commit()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_qk,
        tiled_mma_pv,
        sQ,
        sK,
        sV,
        tStS,
        tOtO,
        tOrP,
        pipeline_q,
        pipeline_kv,
        pipeline_s_p,
        pipeline_o_acc,
        is_leader_cta,
        TileSchedulerCls,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)

        qk_mma_op = tiled_mma_qk.op
        pv_mma_op = tiled_mma_pv.op

        q_smem_start = [
            cute.make_tensor(
                sQ[None, None, None, stage].iterator, cute.make_layout(1)
            ).iterator
            for stage in range(self.q_stage)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline_custom.PipelineUserType.Consumer, self.kv_stage
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            if is_leader_cta:
                for stage in cutlass.range_constexpr(self.q_stage):
                    pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)

                    sm100_utils.gemm(
                        tiled_mma_qk,
                        tStS[None, None, None, stage],
                        tSrQ[None, None, None, stage],
                        tSrK[None, None, None, mma_kv_consumer_state.index],
                        zero_init=True,
                    )

                    pipeline_s_p.producer_commit_w_index(stage)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()

                mma_q_consumer_phase ^= 1

                block_loop_count = mK.shape[0] // self.n_block_size - 1
                for i in cutlass.range(block_loop_count):
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Vi_index = mma_kv_consumer_state.index

                    for stage in cutlass.range_constexpr(self.q_stage):
                        pipeline_s_p.producer_acquire_w_index_phase(stage, Int32(0))

                        sm100_utils.gemm(
                            tiled_mma_pv,
                            tOtO[None, None, None, stage],
                            tOrP[None, None, None, stage],
                            tOrV[None, None, None, Vi_index],
                            zero_init=(i == 0),
                        )

                    pipeline_o_acc.producer_commit_w_index(0)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()

                for stage in cutlass.range(self.q_stage):
                    pipeline_q.consumer_release_w_index(stage)

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def softmax(
        self,
        softmax_scale_log2: Float32,
        thr_mma_qk,
        tStS,
        pipeline_s_p,
        TileSchedulerCls,
    ):
        tidx = cute.arch.thread_idx()[0]

        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tSAcc = tStS[(None, None), 0, 0, 0]

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )

        softmax = SoftmaxSm100.create(
            softmax_scale_log2,
            rescale_threshold=8.0,
            softmax_scale=None,
        )
        softmax.reset()

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            pipeline_s_p.consumer_wait(Int32(0))

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                tSAcc=tSAcc,
            )

            softmax_step(is_first=True, mask_causal=False)
            softmax_step(mask_causal=False)

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def softmax_step(
        self,
        softmax,
        thr_mma_qk,
        tSAcc,
        is_first,
        mask_causal,
    ):
        tidx = cute.arch.thread_idx()[0]

        softmax.reduce()

    @cute.jit
    def epilogue(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O,
        pipeline_o_acc,
        pipeline_o_epi,
        TileSchedulerCls,
    ):
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            pipeline_o_acc.consumer_wait(Int32(0))

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        pipeline_o_epi.producer_tail()


def run_flash_attn_fwd(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    softmax_scale: float = 1.0,
    causal: bool = False,
):
    import torch

    head_dim = q.shape[-1]
    batch_size = q.shape[0]
    nheads = q.shape[2]
    seqlen_q = q.shape[1]

    fa_kernel = FlashAttentionForwardSm100Simple(
        head_dim=head_dim,
        is_causal=causal,
    )

    stream = torch.cuda.current_stream()
    cu_stream = cuda.CUstream(stream.cuda_stream)

    o = torch.empty_like(q)
    lse = torch.empty(batch_size, nheads, seqlen_q, dtype=torch.float32, device="cuda")

    fa_kernel(q, k, v, o, lse, Float32(softmax_scale), cu_stream)

    return o
