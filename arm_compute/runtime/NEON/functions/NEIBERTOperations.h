#ifndef ARM_COMPUTE_IBERT_OPERATIONS_H
#define ARM_COMPUTE_IBERT_OPERATIONS_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;

class NEIBERTGELU: public IFunction
{
public:
    NEIBERTGELU();
    ~NEIBERTGELU();

    NEIBERTGELU(const NEIBERTGELU &) = delete;
    NEIBERTGELU(NEIBERTGELU &&);

    NEIBERTGELU &operator=(const NEIBERTGELU &) = delete;
    NEIBERTGELU &operator=(NEIBERTGELU &&);

    void configure(const ITensor *input, ITensor *output);
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

class NEIBERTSoftmax: public IFunction
{
public:
    NEIBERTSoftmax();
    ~NEIBERTSoftmax();

    NEIBERTSoftmax(const NEIBERTSoftmax &) = delete;
    NEIBERTSoftmax(NEIBERTSoftmax &&);

    NEIBERTSoftmax &operator=(const NEIBERTSoftmax &) = delete;
    NEIBERTSoftmax &operator=(NEIBERTSoftmax &&);

    void configure(const ITensor *input, ITensor *output);
    void run() override;
private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

class NEIBERTLayerNorm: public IFunction
{
public:
    NEIBERTLayerNorm();
    ~NEIBERTLayerNorm();

    NEIBERTLayerNorm(const NEIBERTLayerNorm &) = delete;
    NEIBERTLayerNorm(NEIBERTLayerNorm &&);

    NEIBERTLayerNorm &operator=(const NEIBERTLayerNorm &) = delete;
    NEIBERTLayerNorm &operator=(NEIBERTLayerNorm &&);

    void configure(const ITensor *input, ITensor *output);
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

class NECharlesSoftmax: public IFunction
{
public:

    NECharlesSoftmax();
    ~NECharlesSoftmax();

    NECharlesSoftmax(const NECharlesSoftmax &) = delete;
    NECharlesSoftmax(NECharlesSoftmax &&);

    NECharlesSoftmax &operator=(const NECharlesSoftmax &) = delete;
    NECharlesSoftmax &operator=(NECharlesSoftmax &&);

    void configure(const ITensor *input, ITensor *output, int offset);
    void run();

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

}

#endif
