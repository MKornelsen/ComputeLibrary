#ifndef ARM_COMPUTE_CPU_MULTIPLY_ACCUMULATE_KERNEL_H
#define ARM_COMPUTE_CPU_MULTIPLY_ACCUMULATE_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute 
{
namespace cpu
{
namespace kernels
{

class CpuMultiplyAccumulateKernel : public ICpuKernel<CpuMultiplyAccumulateKernel>
{
private:
    using MultiplyAccumulateKernelPtr = std::add_pointer<void(const ITensor *, const ITensor *, const ITensor *, ITensor *, const Window &)>::type;

public:
    struct MultiplyAccumulateKernel
    {
        const char *name;
        const DataTypeISASelectorPtr is_selected;
        MultiplyAccumulateKernelPtr ukernel;
    };

    CpuMultiplyAccumulateKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuMultiplyAccumulateKernel);

    void configure(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, ITensorInfo *dst);

    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst);

    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

    const char *name() const override;

    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

    static const std::vector<MultiplyAccumulateKernel> &get_available_kernels();

private:
    MultiplyAccumulateKernelPtr _run_method { nullptr };
    std::string _name{};
};

}
}
}

#endif
