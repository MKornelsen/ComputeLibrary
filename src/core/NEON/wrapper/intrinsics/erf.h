/* 
 * Murray Kornelsen
 */

#ifndef ARM_COMPUTE_WRAPPER_ERF_H
#define ARM_COMPUTE_WRAPPER_ERF_H

#include "src/core/NEON/NEMath.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VERF_IMPL(vtype, prefix, postfix) \
    inline vtype verf(const vtype &a)     \
    {                                     \
        return prefix##_##postfix(a);     \
    }

#define VERF_IMPL_INT(vtype, prefix, postfix) \
    inline vtype verf(const vtype &a)         \
    {                                         \
        ARM_COMPUTE_UNUSED(a);                \
        ARM_COMPUTE_ERROR("Not supported");   \
    }

VERF_IMPL(float32x4_t, verfq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VERF_IMPL(float16x8_t, verfq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
// VERF_IMPL_INT(int32x4_t, verfq, s32)

#undef VLOG_IMPL

} // namespace wrapper
} // namespace arm_compute

#endif
