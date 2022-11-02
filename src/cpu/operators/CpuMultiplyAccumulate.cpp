#include "src/cpu/operators/CpuMultiplyAccumulate.h"
#include "src/cpu/kernels/CpuMultiplyAccumulateKernel.h"
#include "src/common/utils/Log.h"

namespace arm_compute
{
namespace cpu
{

void CpuMultiplyAccumulate::configure(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, ITensorInfo *dst)
{
    auto k = std::make_unique<kernels::CpuMultiplyAccumulateKernel>();
    k->configure(src0, src1, src2, dst);
    _kernel = std::move(k);
}

Status CpuMultiplyAccumulate::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst)
{
    return kernels::CpuMultiplyAccumulateKernel::validate(src0, src1, src2, dst);
}

}
}