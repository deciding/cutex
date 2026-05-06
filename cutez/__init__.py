from typing import List, Type, Union, Tuple

import cutlass.cute as cute

from cutlass.cute.arch import numeric_conversion
from cutlass.cute.nvgpu.common import CopyUniversalOp
from cutlass.cute.nvgpu.warp import StMatrix8x8x16bOp, StMatrix16x8x8bOp
from cutlass.cute.nvgpu.tcgen05 import (
    MmaF16BF16Op,
    MmaTF32Op,
    MmaI8Op,
    MmaFP8Op,
    MmaMXF8Op,
    MmaMXF4Op,
    MmaMXF4NVF4Op,
    OperandSource as Tcgen05OperandSource,
    OperandMajorMode,
    CtaGroup,
    Ld16x64bOp,
    Ld16x128bOp,
    Ld16x256bOp,
    Ld16x32bx2Op,
    Ld32x32bOp,
    Repetition,
    Pack,
    SmemLayoutAtomKind,
    make_smem_layout_atom,
    tile_to_mma_shape,
    is_tmem_load,
    get_tmem_copy_properties,
)
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkTensorTileG2SMulticastOp,
    CopyBulkTensorTileG2SOp,
)
from cutlass.utils.layout import LayoutEnum
from cutlass.cutlass_dsl import (
    Float16,
    BFloat16,
    TFloat32,
    Float32,
    Uint8,
    Int8,
    Float8E4M3FN,
    Float8E5M2,
    Float4E2M1FN,
    Numeric,
    NumericMeta,
    dsl_user_op,
)
from torch._prims_common import number_type

# Public autotune entrypoints live alongside the broader cutez package exports.
from .autotune import Config, autotune
from .compiler import compile


@dsl_user_op
def get_smem_store_op(
    layout_d: LayoutEnum,
    elem_ty_d: Type[Numeric],
    elem_ty_acc: Type[Numeric],
    tiled_tmem_load: cute.TiledCopy,
    *,
    verbose: bool = True,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    """Selects the largest vectorized smem store atom available subject to
    constraint of gmem layout and chosen TMEM_LOAD's thread-value ownership.

    :param layout_d: The layout enum of the output tensor D.
    :type layout_d: LayoutEnum
    :param elem_ty_d: The element type for output tensor D.
    :type elem_ty_d: Type[Numeric]
    :param elem_ty_acc: The element type for accumulator.
    :type elem_ty_acc: Type[Numeric]
    :param tiled_tmem_load: An instance of TiledCopy that represents the tmem load operation.
    :type tiled_tmem_load: cute.TiledCopy

    :return: Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.
    :rtype: cute.CopyAtom
    """

    def validate_type(ty, ty_name):
        if not isinstance(ty, NumericMeta):
            raise TypeError(f"{ty_name} must be a Numeric, but got {ty}")

    validate_type(elem_ty_d, "elem_ty_d")
    validate_type(elem_ty_acc, "elem_ty_acc")

    is_m_major = layout_d.is_m_major_c()
    is_n_major = layout_d.is_n_major_c()

    if not is_tmem_load(tiled_tmem_load):
        return cute.make_copy_atom(CopyUniversalOp(), elem_ty_d, loc=loc, ip=ip)

    num_dp, num_bits, num_rep, pack = get_tmem_copy_properties(tiled_tmem_load)

    use_stmatrix_m8n8_4x = (
        all(
            [
                elem_ty_acc.width == 32,
                elem_ty_d.width == 32,
                is_n_major,
                num_dp == 16,
                num_bits == 128,
                num_rep in (2, 4, 8, 16, 32, 64),
                pack == Pack.NONE,
            ]
        )
        or all(
            [
                elem_ty_acc.width == 32,
                elem_ty_d.width == 16,
                num_dp == 16,
                num_bits == 256,
                num_rep in (2, 4, 8, 16, 32),
                pack == Pack.NONE,
            ]
        )
        or all(
            [
                elem_ty_acc.width == 16,
                elem_ty_d.width == 16,
                num_dp == 16,
                num_bits == 128,
                num_rep in (2, 4, 8, 16, 32, 64),
                pack == Pack.PACK_16b_IN_32b,
            ]
        )
    )
    use_stmatrix_m16n8_4x = all(
        [
            elem_ty_acc.width == 32,
            elem_ty_d.width == 8,
            is_m_major,
            num_dp == 16,
            num_bits == 256,
            num_rep in (4, 8, 16, 32),
            pack == Pack.NONE,
        ]
    )
    use_stmatrix_m8n8_2x = (
        all(
            [
                elem_ty_acc.width == 32,
                elem_ty_d.width == 32,
                is_n_major,
                num_dp == 16,
                num_bits == 128,
                num_rep == 1,
                pack == Pack.NONE,
            ]
        )
        or all(
            [
                elem_ty_acc.width == 32,
                elem_ty_d.width == 16,
                num_dp == 16,
                num_bits == 256,
                num_rep == 1,
                pack == Pack.NONE,
            ]
        )
        or all(
            [
                elem_ty_acc.width == 16,
                elem_ty_d.width == 16,
                num_dp == 16,
                num_bits == 128,
                num_rep == 1,
                pack == Pack.PACK_16b_IN_32b,
            ]
        )
    )
    use_stmatrix_m16n8_2x = all(
        [
            elem_ty_acc.width == 32,
            elem_ty_d.width == 8,
            is_m_major,
            num_dp == 16,
            num_bits == 256,
            num_rep == 2,
            pack == Pack.NONE,
        ]
    )
    use_stmatrix_m16n8_1x = all(
        [
            elem_ty_acc.width == 32,
            elem_ty_d.width == 8,
            is_m_major,
            num_dp == 16,
            num_bits == 256,
            num_rep == 1,
            pack == Pack.NONE,
        ]
    )
    if verbose:
        print("===========================================")
        print("Explain get_smem_store_op:")
        if num_dp != 16 or num_bits < 128:
            print(
                "NOTE: to directly use stmatrix, tcgen05.ld must have 16 lanes and 128/256b instruction"
            )
        print(f"elem_ty_acc.width: {elem_ty_acc.width}")
        print(f"elem_ty_d.width: {elem_ty_d.width}")
        print(f"is_n_major: {is_n_major}")
        print(f"num_dp: {num_dp}")
        print(f"num_bits: {num_bits}")
        print(f"num_rep: {num_rep}")
        print(f"pack: {pack}")

    if use_stmatrix_m8n8_4x:
        op = StMatrix8x8x16bOp(is_m_major, 4)
    elif use_stmatrix_m8n8_2x:
        op = StMatrix8x8x16bOp(is_m_major, 2)
    elif use_stmatrix_m16n8_4x:
        op = StMatrix16x8x8bOp(transpose=True, num_matrices=4)
    elif use_stmatrix_m16n8_2x:
        op = StMatrix16x8x8bOp(transpose=True, num_matrices=2)
    elif use_stmatrix_m16n8_1x:
        op = StMatrix16x8x8bOp(transpose=True, num_matrices=1)
    else:
        op = CopyUniversalOp()

    if verbose:
        print(op)
        print("===========================================")

    return cute.make_copy_atom(op, elem_ty_d, loc=loc, ip=ip)
