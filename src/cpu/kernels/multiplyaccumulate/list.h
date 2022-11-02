#ifndef SRC_CORE_KERNELS_MULTIPLY_ACCUMULATE_LIST_H
#define SRC_CORE_KERNELS_MULTIPLY_ACCUMULATE_LIST_H

#include "src/cpu/kernels/multiplyaccumulate/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
#define DECLARE_MULTIPLY_ACCUMULATE_KERNEL(func_name) void func_name(const ITensor *src0, const ITensor *src1, const ITensor *src2, ITensor *dst, const Window &window)

DECLARE_MULTIPLY_ACCUMULATE_KERNEL(multiply_accumulate_qasymm8_signed_neon);
DECLARE_MULTIPLY_ACCUMULATE_KERNEL(multiply_accumulate_f16_neon);
DECLARE_MULTIPLY_ACCUMULATE_KERNEL(multiply_accumulate_f32_neon);

#undef DECLARE_MULTIPLY_ACCUMULATE_KERNEL
}
}

#endif
