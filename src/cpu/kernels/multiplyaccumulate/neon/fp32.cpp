#include "src/cpu/kernels/multiplyaccumulate/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
void multiply_accumulate_f32_neon(const ITensor *src0, const ITensor *src1, const ITensor *src2, ITensor *dst, const Window &window)
{
    return multiply_accumulate_float<float>(src0, src1, src2, dst, window);
}
}
}
