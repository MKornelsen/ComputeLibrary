#include "arm_compute/runtime/NEON/functions/NEMultiplyAccumulate.h"

#include "arm_compute/core/Validate.h"
#include "src/cpu/operators/CpuMultiplyAccumulate.h"

#include <utility>

namespace arm_compute
{
struct NEMultiplyAccumulate::Impl
{
    const ITensor *src0{ nullptr };
    const ITensor *src1{ nullptr };
    const ITensor *src2{ nullptr };
    ITensor *dst{ nullptr };
    std::unique_ptr<cpu::CpuMultiplyAccumulate> op{ nullptr };
};

NEMultiplyAccumulate::NEMultiplyAccumulate()
    : _impl(std::make_unique<Impl>())
    {}

NEMultiplyAccumulate::NEMultiplyAccumulate(NEMultiplyAccumulate &&) = default;
NEMultiplyAccumulate &NEMultiplyAccumulate::operator=(NEMultiplyAccumulate &&) = default;
NEMultiplyAccumulate::~NEMultiplyAccumulate() = default;

Status NEMultiplyAccumulate::validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output) \
{
    return cpu::CpuMultiplyAccumulate::validate(input0, input1, input2, output);
}

void NEMultiplyAccumulate::configure(const ITensor *input0, const ITensor *input1, const ITensor *input2, ITensor *output)
{
    _impl->src0 = input0;
    _impl->src1 = input1;
    _impl->src2 = input2;
    _impl->dst = output;
    _impl->op = std::make_unique<cpu::CpuMultiplyAccumulate>();
    _impl->op->configure(_impl->src0->info(), _impl->src1->info(), _impl->src2->info(), _impl->dst->info());
}

void NEMultiplyAccumulate::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src1);
    pack.add_tensor(TensorType::ACL_SRC_2, _impl->src2);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}
}