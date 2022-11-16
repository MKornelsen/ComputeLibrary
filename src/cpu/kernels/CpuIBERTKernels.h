#ifndef ARM_COMPUTE_CPU_IBERT_KERNELS_H
#define ARM_COMPUTE_CPU_IBERT_KERNELS_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"


namespace arm_compute
{
namespace cpu
{
namespace kernels
{

class CpuIBERTGELUKernel : public ICpuKernel<CpuIBERTGELUKernel>
{
public:

    CpuIBERTGELUKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuIBERTGELUKernel);

    void configure(const ITensorInfo *src, ITensorInfo *dst);
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
    
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

private:
    std::string _name{};
};

class CpuIBERTSoftmaxKernel: public ICpuKernel<CpuIBERTSoftmaxKernel>
{
public:

    CpuIBERTSoftmaxKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuIBERTSoftmaxKernel);

    void configure(const ITensorInfo *src, ITensorInfo *dst);
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
    
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

private:
    std::string _name{};
};

class CpuIBERTLayerNormKernel : public ICpuKernel<CpuIBERTLayerNormKernel>
{
public:
    CpuIBERTLayerNormKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuIBERTLayerNormKernel);

    void configure(const ITensorInfo *src, ITensorInfo *dst);
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
    
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

private:
    std::string _name{};
};

}
}
}

#endif
