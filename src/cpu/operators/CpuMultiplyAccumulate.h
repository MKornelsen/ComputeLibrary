#ifndef ARM_COMPUTE_CPU_MULTIPLY_ACCUMULATE_H
#define ARM_COMPUTE_CPU_MULTIPLY_ACCUMULATE_H

#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{

class CpuMultiplyAccumulate : public ICpuOperator
{
public:
    
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, ITensorInfo *dst);

    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst);
};
}
}
#endif