#ifndef ARM_COMPUTE_CPU_IBERT_OPERATIONS_H
#define ARM_COMPUTE_CPU_IBERT_OPERATIONS_H

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/experimental/Types.h"
#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{

class CpuIBERTGELU : public ICpuOperator
{
public:

    void configure(const ITensorInfo *src, ITensorInfo *dst);
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
};

class CpuIBERTSoftmax : public ICpuOperator
{
public:

    CpuIBERTSoftmax();

    void configure(const ITensorInfo *src, ITensorInfo *dst);
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
    void run(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:

    TensorInfo _tmp;

    experimental::MemoryRequirements _aux_mem;
};

class CpuIBERTLayerNorm : public ICpuOperator
{
public:

    void configure(const ITensorInfo *src, ITensorInfo *dst);
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
};

class CpuCharlesSoftmax : public ICpuOperator
{
public:

    void configure(const ITensorInfo *src, ITensorInfo *dst, int offset);
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
};

}
}
#endif
