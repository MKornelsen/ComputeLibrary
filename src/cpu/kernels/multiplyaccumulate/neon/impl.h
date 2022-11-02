#ifndef SRC_CORE_NEON_KERNELS_MULTIPLY_ACCUMULATE_IMPL_H
#define SRC_CORE_NEON_KERNELS_MULTIPLY_ACCUMULATE_IMPL_H

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace cpu
{
template <typename ScalarType>
void multiply_accumulate_float(const ITensor *src0, const ITensor *src1, const ITensor *src2, ITensor *dst, const Window &window);
}
}

#endif
