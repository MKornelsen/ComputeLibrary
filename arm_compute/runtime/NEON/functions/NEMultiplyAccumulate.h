#ifndef ARM_COMPUTE_NEMULTIPLYACCUMULATE_H
#define ARM_COMPUTE_NEMULTIPLYACCUMULATE_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;

class NEMultiplyAccumulate : public IFunction
{
public:

    NEMultiplyAccumulate();
    ~NEMultiplyAccumulate();
    
    NEMultiplyAccumulate(const NEMultiplyAccumulate &) = delete;

    NEMultiplyAccumulate(NEMultiplyAccumulate &&);

    NEMultiplyAccumulate &operator=(const NEMultiplyAccumulate &) = delete;

    NEMultiplyAccumulate &operator=(NEMultiplyAccumulate &&);

    void configure(const ITensor *input0, const ITensor *input1, const ITensor *input2, ITensor *output);
    
    static Status validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);

    void run() override;

private:

    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}

#endif
