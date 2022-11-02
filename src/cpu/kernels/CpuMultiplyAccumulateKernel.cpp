#include "src/cpu/kernels/CpuMultiplyAccumulateKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/multiplyaccumulate/list.h"
#include <array>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
static const std::vector<CpuMultiplyAccumulateKernel::MultiplyAccumulateKernel> available_kernels =
{
    {
        "neon_fp32_mla",
        [](const DataTypeISASelectorData &data)
        {
            return data.dt == DataType::F32;
        },
        REGISTER_FP32_NEON(arm_compute::cpu::multiply_accumulate_f32_neon)
    },
    {
        "neon_fp16_mla",
        [](const DataTypeISASelectorData &data) 
        {
            return data.dt == DataType::F16 && data.isa.fp16;
        },
        REGISTER_FP16_NEON(arm_compute::cpu::multiply_accumulate_f16_neon)
    },
    {
        "neon_qs8_add",
        [](const DataTypeISASelectorData &data)
        {
            return data.dt == DataType::QASYMM8_SIGNED;
        },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::multiply_accumulate_qasymm8_signed_neon)
    }
};

Status validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &src2, const ITensorInfo &dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src0, &src1, &src2, &dst);

    const TensorShape out_shape = TensorShape::broadcast_shape(src0.tensor_shape(), src1.tensor_shape(), src2.tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");
    
    if (dst.total_size() > 0) {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst.tensor_shape(), 0), "Wrong shape for dst");
    }

    const auto *uk = CpuMultiplyAccumulateKernel::get_implementation(DataTypeISASelectorData { src0.data_type(), CPUInfo::get().get_isa() });
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);
    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &src2, ITensorInfo &dst)
{
    const TensorShape &out_shape = TensorShape::broadcast_shape(src0.tensor_shape(), src1.tensor_shape(), src2.tensor_shape());

    set_shape_if_empty(dst, out_shape);
    set_data_type_if_unknown(dst, src0.data_type());

    Window win = calculate_max_window(out_shape, Steps());

    return std::make_pair(Status{}, win);
}
}

void CpuMultiplyAccumulateKernel::configure(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, ITensorInfo *dst) 
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, src2, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *src2, *dst));

    const auto uk = CpuMultiplyAccumulateKernel::get_implementation(DataTypeISASelectorData { src0->data_type(), CPUInfo::get().get_isa() });

    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);

    _run_method = uk->ukernel;
    _name = std::string("CpuMultiplyAccumulateKernel").append("/").append(uk->name);

    auto win_config = validate_and_configure_window(*src0, *src1, *src2, *dst);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICpuKernel::configure(win_config.second);
}

Status CpuMultiplyAccumulateKernel::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, src2, dst);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *src2, *dst));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(*src0->clone(), *src1->clone(), *src2->clone(), *dst->clone()).first);

    return Status{};
}

void CpuMultiplyAccumulateKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(tensors.empty());
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const ITensor *src0 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *src1 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const ITensor *src2 = tensors.get_const_tensor(TensorType::ACL_SRC_2);

    ITensor *dst = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src0, src1, src2, dst, window);
}

const char *CpuMultiplyAccumulateKernel::name() const
{
    return _name.c_str();
}

const std::vector<CpuMultiplyAccumulateKernel::MultiplyAccumulateKernel> &CpuMultiplyAccumulateKernel::get_available_kernels()
{
    return available_kernels;
}

size_t CpuMultiplyAccumulateKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);
    ARM_COMPUTE_UNUSED(platform);

    return ICPPKernel::default_mws;
}
}
}
}
