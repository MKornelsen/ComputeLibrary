#include "arm_compute/runtime/NEON/functions/NEIBERTOperations.h"

#include "arm_compute/core/Validate.h"
#include "src/cpu/operators/CpuIBERTOperations.h"

#include <utility>

namespace arm_compute
{

struct NEIBERTGELU::Impl {
    const ITensor *src{ nullptr };
    ITensor *dst{ nullptr };
    std::unique_ptr<cpu::CpuIBERTGELU> op { nullptr };
};

NEIBERTGELU::NEIBERTGELU()
    : _impl(std::make_unique<Impl>()) 
{
}

NEIBERTGELU::NEIBERTGELU(NEIBERTGELU &&) = default;
NEIBERTGELU &NEIBERTGELU::operator=(NEIBERTGELU &&) = default;
NEIBERTGELU::~NEIBERTGELU() = default;

void NEIBERTGELU::configure(const ITensor *src, ITensor *dst)
{
    _impl->src = src;
    _impl->dst = dst;
    _impl->op = std::make_unique<cpu::CpuIBERTGELU>();
    _impl->op->configure(_impl->src->info(), _impl->dst->info());
}

void NEIBERTGELU::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct NEIBERTSoftmax::Impl {
    const ITensor *src{ nullptr };
    ITensor *dst{ nullptr };
    std::unique_ptr<cpu::CpuIBERTSoftmax> op { nullptr };
};

NEIBERTSoftmax::NEIBERTSoftmax()
    : _impl(std::make_unique<Impl>()) 
{
}

NEIBERTSoftmax::NEIBERTSoftmax(NEIBERTSoftmax &&) = default;
NEIBERTSoftmax &NEIBERTSoftmax::operator=(NEIBERTSoftmax &&) = default;
NEIBERTSoftmax::~NEIBERTSoftmax() = default;

void NEIBERTSoftmax::configure(const ITensor *src, ITensor *dst)
{
    _impl->src = src;
    _impl->dst = dst;
    _impl->op = std::make_unique<cpu::CpuIBERTSoftmax>();
    _impl->op->configure(_impl->src->info(), _impl->dst->info());
}

void NEIBERTSoftmax::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct NEIBERTLayerNorm::Impl {
    const ITensor *src{ nullptr };
    ITensor *dst{ nullptr };
    std::unique_ptr<cpu::CpuIBERTLayerNorm> op { nullptr };
};

NEIBERTLayerNorm::NEIBERTLayerNorm()
    : _impl(std::make_unique<Impl>()) 
{
}

NEIBERTLayerNorm::NEIBERTLayerNorm(NEIBERTLayerNorm &&) = default;
NEIBERTLayerNorm &NEIBERTLayerNorm::operator=(NEIBERTLayerNorm &&) = default;
NEIBERTLayerNorm::~NEIBERTLayerNorm() = default;

void NEIBERTLayerNorm::configure(const ITensor *src, ITensor *dst)
{
    _impl->src = src;
    _impl->dst = dst;
    _impl->op = std::make_unique<cpu::CpuIBERTLayerNorm>();
    _impl->op->configure(_impl->src->info(), _impl->dst->info());
}

void NEIBERTLayerNorm::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct NECharlesSoftmax::Impl {
    const ITensor *src{ nullptr };
    const ITensor *offset{ nullptr };
    ITensor *dst{ nullptr };
    std::unique_ptr<cpu::CpuCharlesSoftmax> op{ nullptr };
};

NECharlesSoftmax::NECharlesSoftmax()
    : _impl(std::make_unique<Impl>())
{
}

NECharlesSoftmax::NECharlesSoftmax(NECharlesSoftmax &&) = default;
NECharlesSoftmax &NECharlesSoftmax::operator=(NECharlesSoftmax &&) = default;
NECharlesSoftmax::~NECharlesSoftmax() = default;

void NECharlesSoftmax::configure(const ITensor *src, ITensor *dst, float offset)
{
    _impl->src = src;
    _impl->dst = dst;
    _impl->op = std::make_unique<cpu::CpuCharlesSoftmax>();
    _impl->op->configure(_impl->src->info(), _impl->dst->info(), offset);
}

void NECharlesSoftmax::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

}
