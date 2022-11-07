#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include "src/cpu/kernels/multiplyaccumulate/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
void multiply_accumulate_f16_neon(const ITensor *src0, const ITensor *src1, const ITensor *src2, ITensor *dst, const Window &window)
{
    return multiply_accumulate_float<float16_t>(src0, src1, src2, dst, window);
}
}
}

#endif
