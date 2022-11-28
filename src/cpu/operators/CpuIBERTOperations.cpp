#include "src/cpu/operators/CpuIBERTOperations.h"
#include "src/cpu/kernels/CpuIBERTKernels.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/common/utils/Log.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

using namespace arm_compute::experimental;

namespace arm_compute
{
namespace cpu
{

void CpuIBERTGELU::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst);
    auto k = std::make_unique<kernels::CpuIBERTGELUKernel>();
    k->configure(src, dst);
    _kernel = std::move(k);
}

Status CpuIBERTGELU::validate(const ITensorInfo *src, const ITensorInfo *dst) 
{
    return kernels::CpuIBERTGELUKernel::validate(src, dst);
}

CpuIBERTSoftmax::CpuIBERTSoftmax()
    :   _tmp(),
        _aux_mem(1)
{
}

void CpuIBERTSoftmax::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(CpuIBERTSoftmax::validate(src, dst));

    _tmp = TensorInfo(src->clone()->set_data_type(DataType::S16));
    _aux_mem[0] = MemoryInfo(offset_int_vec(0), MemoryLifetime::Temporary, _tmp.total_size());

    auto k = std::make_unique<kernels::CpuIBERTSoftmaxKernel>();
    k->configure(src, dst);
    _kernel = std::move(k);
}

Status CpuIBERTSoftmax::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    return kernels::CpuIBERTSoftmaxKernel::validate(src, dst);
}

void CpuIBERTSoftmax::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    CpuAuxTensorHandler tmp(offset_int_vec(0), _tmp, tensors, true);
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
}

experimental::MemoryRequirements CpuIBERTSoftmax::workspace() const 
{
    return _aux_mem;
}

void CpuIBERTLayerNorm::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst);
    auto k = std::make_unique<kernels::CpuIBERTLayerNormKernel>();
    k->configure(src, dst);
    _kernel = std::move(k);
}

Status CpuIBERTLayerNorm::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    return kernels::CpuIBERTLayerNormKernel::validate(src, dst);
}

void CpuCharlesSoftmax::configure(const ITensorInfo *src, ITensorInfo *dst, float offset)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst);
    auto k = std::make_unique<kernels::CpuCharlesSoftmaxKernel>();
    k->configure(src, dst, offset);
    _kernel = std::move(k);
}

Status CpuCharlesSoftmax::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    return kernels::CpuCharlesSoftmaxKernel::validate(src, dst);
}

}
}
