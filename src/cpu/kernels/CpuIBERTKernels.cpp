#include "src/cpu/kernels/CpuIBERTKernels.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/core/NEON/NEAsymm.h"
#include <array>

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace {

Status validate_arguments(const ITensorInfo &src, const ITensorInfo &dst) 
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(&src, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &dst);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src.quantization_info().uniform().offset != 0, "IBERT requires offsets of 0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst.quantization_info().uniform().offset != 0, "IBERT requires offsets of 0");
    
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(src.tensor_shape(), dst.tensor_shape(), 0), "IBERT Operation input and output must have same size");

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo &src, ITensorInfo &dst)
{
    const TensorShape out_shape = src.tensor_shape();

    set_shape_if_empty(dst, out_shape);
    set_data_type_if_unknown(dst, src.data_type());

    Window win = calculate_max_window(out_shape, Steps());
    return std::make_pair(Status{}, win);
}
}

// GELU

void CpuIBERTGELUKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src, *dst));

    _name = std::string("CpuIBERTGELUKernel");

    auto win_config = validate_and_configure_window(*src, *dst);

    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICpuKernel::configure(win_config.second);
}

Status CpuIBERTGELUKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src, *dst));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(*src->clone(), *dst->clone()).first);

    return Status{};
}

void CpuIBERTGELUKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());
    
    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC);
    const ITensor *dst = tensors.get_tensor(TensorType::ACL_DST);

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int window_step_x = 16;
    const int window_start_x = static_cast<int>(window.x().start());
    const int window_end_x = static_cast<int>(window.x().end());

    const float src_scale = src->info()->quantization_info().uniform().scale;
    const float dst_scale = dst->info()->quantization_info().uniform().scale;
    // S / sqrt(2)
    const float erf_input_scale = src_scale / 1.41421356237f;

    // I-ERF Params
    // a = -0.2888
    // b = -1.769 
    // c = 1

    int clipping_max = (int) 1.769f / erf_input_scale;
    // std::cout << "clipping max = " << clipping_max << std::endl;
    if (clipping_max > 127) {
        clipping_max = 127;
    }
    const int16x8_t clip_max_vec = vmovq_n_s16(clipping_max);
    
    const int qb = (int) (-1.769f / erf_input_scale);
    // std::cout << "qb = " << qb << std::endl;
    const int16x8_t qbvec = vmovq_n_s16(qb);

    const float erf_scale = -0.2888f * erf_input_scale * erf_input_scale;

    const int qc = (int) (1.0f / erf_scale);
    // std::cout << "qc = " << qc << std::endl;
    const int32x4_t qcvec = vmovq_n_s32(qc);

    // std::cout << "erf output scale = " << erf_scale << std::endl;

    const int q1 = (int) (1.0f / erf_scale);
    // std::cout << "q1 = " << q1 << std::endl;
    const int32x4_t q1vec = vmovq_n_s32(q1);

    const float out_scale = src_scale * erf_scale / 2.0f;
    // const float32x4_t out_scale_vec = vmovq_n_f32(out_scale);

    const float requant_scale = out_scale / dst_scale;
    // const float32x4_t requant_scale_vec = vmovq_n_f32(requant_scale);
    Iterator input(src, win);
    Iterator output(dst, win);

    // std::cout << "Running IBERT GELU Kernel" << std::endl;

    execute_window_loop(win, [&](const Coordinates &)
    {
        // std::cout << "(" << coords.x() << ", " << coords.y() << ")" << std::endl;
        const auto input_ptr = reinterpret_cast<const int8_t *>(input.ptr());
        const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

        int x = window_start_x;
        for (; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const int8x16_t a = vld1q_s8(input_ptr + x);
            const int8x16_t isneg = (int8x16_t) vcltzq_s8(a);

            // const int8x16_t clipped = vminq_s8(vabsq_s8(a), clip_max_vec);
            const int16x8_t a_low = vminq_s16(vabsq_s16(vmovl_s8(vget_low_s8(a))), clip_max_vec);
            const int16x8_t a_high = vminq_s16(vabsq_s16(vmovl_high_s8(a)), clip_max_vec);
            // I-poly
            const int16x8_t sum_low = vaddq_s16(a_low, qbvec);
            const int16x8_t sum_high = vaddq_s16(a_high, qbvec);

            // std::cout << "sum: " << sum_low[0] << std::endl;
            // const int32x4x4_t ipolyout = {
            //     {
            //         vaddq_s32(vmull_s16(vget_low_s16(sum_low), vget_low_s16(sum_low)), qcvec),
            //         vaddq_s32(vmull_high_s16(sum_low, sum_low), qcvec),
            //         vaddq_s32(vmull_s16(vget_low_s16(sum_high), vget_low_s16(sum_high)), qcvec),
            //         vaddq_s32(vmull_high_s16(sum_high, sum_high), qcvec)
            //     }
            // };
            const int32x4x4_t ipolyout = {
                {
                    vmlal_s16(qcvec, vget_low_s16(sum_low), vget_low_s16(sum_low)),
                    vmlal_high_s16(qcvec, sum_low, sum_low),
                    vmlal_s16(qcvec, vget_low_s16(sum_high), vget_low_s16(sum_high)),
                    vmlal_high_s16(qcvec, sum_high, sum_high)
                }
            };

            const uint32x4x4_t neg_uints = {
                {
                    vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(isneg)))),
                    vmovl_high_s16(vmovl_s8(vget_low_s8(isneg))),
                    vmovl_s16(vget_low_s16(vmovl_high_s8(isneg))),
                    vmovl_high_s16(vmovl_high_s8(isneg))
                }
            };

            const int32x4x4_t erf_selected = {
                {
                    vbslq_s32(neg_uints.val[0], vnegq_s32(ipolyout.val[0]), ipolyout.val[0]),
                    vbslq_s32(neg_uints.val[1], vnegq_s32(ipolyout.val[1]), ipolyout.val[1]),
                    vbslq_s32(neg_uints.val[2], vnegq_s32(ipolyout.val[2]), ipolyout.val[2]),
                    vbslq_s32(neg_uints.val[3], vnegq_s32(ipolyout.val[3]), ipolyout.val[3])
                }
            };

            const int32x4x4_t long_input = {
                {
                    vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(a)))),
                    vmovl_high_s16(vmovl_s8(vget_low_s8(a))),
                    vmovl_s16(vget_low_s16(vmovl_high_s8(a))),
                    vmovl_high_s16(vmovl_high_s8(a))
                }
            };

            const int32x4x4_t qout = {
                {
                    vmulq_s32(long_input.val[0], vaddq_s32(erf_selected.val[0], q1vec)),
                    vmulq_s32(long_input.val[1], vaddq_s32(erf_selected.val[1], q1vec)),
                    vmulq_s32(long_input.val[2], vaddq_s32(erf_selected.val[2], q1vec)),
                    vmulq_s32(long_input.val[3], vaddq_s32(erf_selected.val[3], q1vec))
                }
            };

            const int32x4x4_t requantized = {
                {
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(qout.val[0]), requant_scale)),
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(qout.val[1]), requant_scale)),
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(qout.val[2]), requant_scale)),
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(qout.val[3]), requant_scale))
                }
            };

            const int16x8_t result_low_shorts = vqmovn_high_s32(vqmovn_s32(requantized.val[0]), requantized.val[1]);
            const int16x8_t result_high_shorts = vqmovn_high_s32(vqmovn_s32(requantized.val[2]), requantized.val[3]);
            const int8x16_t result = vqmovn_high_s16(vqmovn_s16(result_low_shorts), result_high_shorts);
            
            vst1q_s8(output_ptr + x, result);
        }

        for (; x < window_end_x; ++x)
        {
            const int a = static_cast<int32_t>(*(input_ptr + x));
            
            int clipped = a;
            if (a < 0) {
                clipped = -clipped;
            }

            if (clipped > clipping_max) {
                clipped = clipping_max;
            }
            const int sum = clipped + qb;
            int erf = (sum * sum) + qc;
            
            if (a < 0) {
                erf = -erf;
            }

            // std::cout << a * src_scale << " - " << erf << " - " << erf * erf_scale << " " << std::endl;
            int qout = a * (erf + q1);

            // std::cout << a * src_scale << " - " << qout * out_scale << " - " << qout << std::endl;;
            int qdst = (int) ((float) qout * requant_scale);

            if (qdst < -128) {
                qdst = -128;
            }
            if (qdst > 127) {
                qdst = 127;
            }
            // std::cout << qout << " -> " << qdst << std::endl;
            *(output_ptr + x) = (int8_t) qdst;
        }
    }, 
    input, output);
}

const char *CpuIBERTGELUKernel::name() const
{
    return _name.c_str();
}

size_t CpuIBERTGELUKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);
    ARM_COMPUTE_UNUSED(platform);

    return ICPPKernel::default_mws;
}

// SOFTMAX
void CpuIBERTSoftmaxKernel::configure(const ITensorInfo *src, ITensorInfo *dst) 
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src, *dst));

    _name = std::string("CpuIBERTSoftmaxKernel");

    auto win_config = validate_and_configure_window(*src, *dst);

    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICpuKernel::configure(win_config.second);
}

Status CpuIBERTSoftmaxKernel::validate(const ITensorInfo *src, const ITensorInfo *dst) 
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src, *dst));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(*src->clone(), *dst->clone()).first);

    return Status{};
}

void CpuIBERTSoftmaxKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) 
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC);
    const ITensor *dst = tensors.get_tensor(TensorType::ACL_DST);
    const ITensor *tmp = tensors.get_tensor(offset_int_vec(0));

    // std::cout << "unpacked tmp tensor: " << tmp->info()->dimension(0) << " " << tmp->info()->dimension(1) << std::endl;

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int window_step_x = 16;
    const int window_start_x = static_cast<int>(window.x().start());
    const int window_end_x = static_cast<int>(window.x().end());

    const float src_scale = src->info()->quantization_info().uniform().scale;
    const float dst_scale = dst->info()->quantization_info().uniform().scale;

    // I-EXP Params
    // a = 0.3585
    // b = 1.353
    // c = 0.344

    const int qb = (int) (1.353f / src_scale);
    const float exp_scale = 0.3585f * src_scale * src_scale;
    const int qc = (int) (0.344f / exp_scale);
    const int qln2 = (int) (M_LN2 / src_scale);
    // std::cout << "qln2 = " << qln2 << std::endl;

    const int32x4_t qbvec = vmovq_n_s32(qb);
    const int32x4_t qcvec = vmovq_n_s32(qc);

    const int inv_qln2_factor = (1 << 14) / qln2;
    // const int neg_inv_qln2_factor = -inv_qln2_factor;
    // const int32x4_t neg_inv_qln2_vector = vmovq_n_s32(neg_inv_qln2_factor);
    // const int32x4_t qln2_vector = vmovq_n_s32(qln2);

    const float requant_scale = (1.0f / 127.0f) / dst_scale;

    Iterator input(src, win);
    Iterator output(dst, win);
    Iterator tmp_itr(tmp, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto input_ptr = reinterpret_cast<const int8_t *>(input.ptr());
        const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

        const auto tmp_ptr = reinterpret_cast<int16_t *>(tmp_itr.ptr());

        int x = window_start_x;
        
        // FIND MAX IN ROW
        int8x16_t maxvec = vmovq_n_s8(-128);
        for (; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const int8x16_t a = vld1q_s8(input_ptr + x);
            maxvec = vmaxq_s8(maxvec, a);
        }

        int max = (int) vmaxvq_s8(maxvec);

        // std::cout << "max = " << max << std::endl;
        for (; x < window_end_x; ++x)
        {
            int a = static_cast<int32_t>(*(input_ptr + x));
            if (a > max) {
                max = a;
            }
        }

        // COMPUTE EXPONENTIALS AND SUM OF ROW
        const int16x8_t max_vec = vmovq_n_s16((int16_t) max);

        // std::cout << "max: " << max_vec[0] << std::endl;
        int32x4_t sum_vec = vmovq_n_s32(0);

        x = window_start_x;
        
        for (; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const int8x16_t a = vld1q_s8(input_ptr + x);

            const int16x8_t a_low = vsubq_s16(vmovl_s8(vget_low_s8(a)), max_vec);
            const int16x8_t a_high = vsubq_s16(vmovl_high_s8(a), max_vec);

            // std::cout << "a_low: ";
            // for (int i = 0; i < 8; i++) {
            //     std::cout << a_low[i] << " ";
            // }
            // std::cout << std::endl;
            const int16x8_t neg_a_low = vnegq_s16(a_low);
            const int16x8_t neg_a_high = vnegq_s16(a_high);

            const int32x4x4_t z = {
                {
                    vshrq_n_s32(vmull_n_s16(vget_low_s16(neg_a_low), inv_qln2_factor), 14),
                    vshrq_n_s32(vmull_high_n_s16(neg_a_low, inv_qln2_factor), 14),
                    vshrq_n_s32(vmull_n_s16(vget_low_s16(neg_a_high), inv_qln2_factor), 14),
                    vshrq_n_s32(vmull_high_n_s16(neg_a_high, inv_qln2_factor), 14)
                }
            };

            // std::cout << "test z: " << z.val[0][0] << std::endl;

            // const int16x8_t qp_low = vaddq_s16(a_low, vmulq_s16(z_low, qln2_vector));
            // const int16x8_t qp_high = vaddq_s16(a_high, vmulq_s16(z_high, qln2_vector)); 
            const int32x4x4_t qp = {
                {
                    vmlaq_n_s32(vmovl_s16(vget_low_s16(a_low)), z.val[0], qln2),
                    vmlaq_n_s32(vmovl_high_s16(a_low), z.val[1], qln2),
                    vmlaq_n_s32(vmovl_s16(vget_low_s16(a_high)), z.val[2], qln2),
                    vmlaq_n_s32(vmovl_high_s16(a_high), z.val[3], qln2)
                }
            };

            const int32x4x4_t ipsums = {
                {
                    vaddq_s32(qp.val[0], qbvec),
                    vaddq_s32(qp.val[1], qbvec),
                    vaddq_s32(qp.val[2], qbvec),
                    vaddq_s32(qp.val[3], qbvec)
                }
            };
            const int32x4x4_t qexp = {
                {
                    vshlq_s32(vmlaq_s32(qcvec, ipsums.val[0], ipsums.val[0]), vnegq_s32(z.val[0])),
                    vshlq_s32(vmlaq_s32(qcvec, ipsums.val[1], ipsums.val[1]), vnegq_s32(z.val[1])),
                    vshlq_s32(vmlaq_s32(qcvec, ipsums.val[2], ipsums.val[2]), vnegq_s32(z.val[2])),
                    vshlq_s32(vmlaq_s32(qcvec, ipsums.val[3], ipsums.val[3]), vnegq_s32(z.val[3]))
                }
            };

            // std::cout << "test qexp: " << qexp.val[0][0] << std::endl;
            // vst1q_s32_x4(tmp_ptr + x, qexp);
            const int16x8x2_t qexp_short = {
                {
                    vqmovn_high_s32(vqmovn_s32(qexp.val[0]), qexp.val[1]),
                    vqmovn_high_s32(vqmovn_s32(qexp.val[2]), qexp.val[3])
                }
            };
            
            vst1q_s16_x2(tmp_ptr + x, qexp_short);

            sum_vec = vaddq_s32(sum_vec, qexp.val[0]);
            sum_vec = vaddq_s32(sum_vec, qexp.val[1]);
            sum_vec = vaddq_s32(sum_vec, qexp.val[2]);
            sum_vec = vaddq_s32(sum_vec, qexp.val[3]);
        }

        int sum = vaddvq_s32(sum_vec);
        
        // int sum = 0;
        // std::cout << 
        for (; x < window_end_x; ++x) 
        {
            int a = static_cast<int32_t>(*(input_ptr + x)) - max;
            // std::cout << "a = " << a << std::endl;
            int z = -a * inv_qln2_factor >> 14;
            // int z = (a * neg_inv_qln2_factor) >> 14;
            int qp = a + z * qln2;
            // std::cout << "z = " << z << std::endl;
            // I-POLY(qp, src_scale)
            int qexp = (qp + qb) * (qp + qb) + qc;
            qexp = qexp >> z;
            // std::cout << "qexp = " << qexp << std::endl;
            *(tmp_ptr + x) = (int16_t) qexp;
            sum += qexp;
        }
        
        // NORMALIZE
        int factor = (1 << 30) / sum;

        // std::cout << "sum = " << sum << std::endl;
        // std::cout << "factor = " << factor << std::endl;
        x = window_start_x;

        for (; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            int16x8x2_t qexp_short = vld1q_s16_x2(tmp_ptr + x);

            int32x4x4_t qexp = {
                {
                    vmovl_s16(vget_low_s16(qexp_short.val[0])),
                    vmovl_high_s16(qexp_short.val[0]),
                    vmovl_s16(vget_low_s16(qexp_short.val[1])),
                    vmovl_high_s16(qexp_short.val[1]),
                }
            };

            int32x4x4_t qout = {
                {
                    vshrq_n_s32(vmulq_n_s32(qexp.val[0], factor), 23),
                    vshrq_n_s32(vmulq_n_s32(qexp.val[1], factor), 23),
                    vshrq_n_s32(vmulq_n_s32(qexp.val[2], factor), 23),
                    vshrq_n_s32(vmulq_n_s32(qexp.val[3], factor), 23)
                }
            };

            int32x4x4_t requantized = {
                {
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(qout.val[0]), requant_scale)),
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(qout.val[1]), requant_scale)),
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(qout.val[2]), requant_scale)),
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(qout.val[3]), requant_scale))
                }
            };

            const int16x8_t result_low_shorts = vqmovn_high_s32(vqmovn_s32(requantized.val[0]), requantized.val[1]);
            const int16x8_t result_high_shorts = vqmovn_high_s32(vqmovn_s32(requantized.val[2]), requantized.val[3]);
            const int8x16_t result = vqmovn_high_s16(vqmovn_s16(result_low_shorts), result_high_shorts);
            
            vst1q_s8(output_ptr + x, result);
        }

        for (; x < window_end_x; ++x)
        {
            // I-POLY(qp, src_scale)
            int qexp = (int) *(tmp_ptr + x);

            // int qout = qexp / sum;
            int qout = (qexp * factor) >> 23;
            // std::cout << "qexp = " << qexp << "\nqout = " << qout << std::endl;
            int qdst = (int) ((float) qout * requant_scale);
            if (qdst < -128) {
                qdst = -128;
            }
            if (qdst > 127) {
                qdst = 127;
            }
            *(output_ptr + x) = (int8_t) qdst;
        }
    },
    input, output, tmp_itr);
}

const char *CpuIBERTSoftmaxKernel::name() const 
{
    return _name.c_str();
}

size_t CpuIBERTSoftmaxKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);
    ARM_COMPUTE_UNUSED(platform);

    return ICPPKernel::default_mws;
}

void CpuIBERTLayerNormKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src, *dst));

    _name = std::string("CpuIBERTLayerNormKernel");

    auto win_config = validate_and_configure_window(*src, *dst);

    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICpuKernel::configure(win_config.second);
}

Status CpuIBERTLayerNormKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src, *dst));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(*src->clone(), *dst->clone()).first);

    return Status{};
}

void CpuIBERTLayerNormKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());
    
    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC);
    const ITensor *dst = tensors.get_tensor(TensorType::ACL_DST);

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int window_step_x = 16;
    const int window_start_x = static_cast<int>(window.x().start());
    const int window_end_x = static_cast<int>(window.x().end());
    
    const float dst_scale = dst->info()->quantization_info().uniform().scale;

    Iterator input(src, win);
    Iterator output(dst, win);

    const float requant_scale = (1.0f / 127.0f) / dst_scale;

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto input_ptr = reinterpret_cast<const int8_t *>(input.ptr());
        const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

        int32x4_t sum_vec = vmovq_n_s32(0);
        int32x4_t sum_sq_vec = vmovq_n_s32(0);

        int x = window_start_x;
        for (; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const int8x16_t a = vld1q_s8(input_ptr + x);

            sum_vec = vaddq_s32(sum_vec, vpaddlq_s16(vpaddlq_s8(a)));

            const int16x8_t sq_low = vmull_s8(vget_low_s8(a), vget_low_s8(a));
            const int16x8_t sq_high = vmull_high_s8(a, a);
            
            sum_sq_vec = vaddq_s32(sum_sq_vec, vpaddlq_s16(sq_low));
            sum_sq_vec = vaddq_s32(sum_sq_vec, vpaddlq_s16(sq_high));
        }

        int sum = vaddvq_s32(sum_vec);
        int sum_sq = vaddvq_s32(sum_sq_vec);

        for (; x < window_end_x; ++x) 
        {
            const int a = *(input_ptr + x);
            sum += a;
            sum_sq += a * a;
        }

        int mean = sum / (int) src->info()->dimension(0);
        int var = sum_sq / (int) src->info()->dimension(0);

        // std::cout << "sum = " << sum << " mean = " << mean << std::endl;

        int stddev = (1 << 16);
        while (1) {
            int next = (stddev + (var / stddev)) / 2;
            if (next >= stddev) {
                break;
            } else {
                stddev = next;
            }
        }

        // std::cout << "sqrt(" << var << ") = " << stddev << std::endl;

        int stddev_inv_factor = (1 << 24) / stddev;

        int8x16_t mean_vec = vmovq_n_s8(mean);
        // int16x8_t stddev_inv_vec = vmovq_n_s16(stddev_inv_factor);

        x = window_start_x;
        for (; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            int8x16_t a = vld1q_s8(input_ptr + x);

            const int16x8_t diff_low = vsubl_s8(vget_low_s8(a), vget_low_s8(mean_vec));
            const int16x8_t diff_high = vsubl_high_s8(a, mean_vec);

            const int32x4x4_t normed = {
                {
                    vshrq_n_s32(vmulq_n_s32(vmovl_s16(vget_low_s16(diff_low)), stddev_inv_factor), 17),
                    vshrq_n_s32(vmulq_n_s32(vmovl_high_s16(diff_low), stddev_inv_factor), 17),
                    vshrq_n_s32(vmulq_n_s32(vmovl_s16(vget_low_s16(diff_high)), stddev_inv_factor), 17),
                    vshrq_n_s32(vmulq_n_s32(vmovl_high_s16(diff_high), stddev_inv_factor), 17)
                }
            };

            int32x4x4_t requantized = {
                {
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(normed.val[0]), requant_scale)),
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(normed.val[1]), requant_scale)),
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(normed.val[2]), requant_scale)),
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(normed.val[3]), requant_scale))
                }
            };

            const int16x8_t result_low_shorts = vqmovn_high_s32(vqmovn_s32(requantized.val[0]), requantized.val[1]);
            const int16x8_t result_high_shorts = vqmovn_high_s32(vqmovn_s32(requantized.val[2]), requantized.val[3]);
            const int8x16_t result = vqmovn_high_s16(vqmovn_s16(result_low_shorts), result_high_shorts);
            
            vst1q_s8(output_ptr + x, result);
        }

        for (; x < window_end_x; ++x)
        {
            int a = (int) *(input_ptr + x);
            int norm = ((a - mean) * stddev_inv_factor) >> 17;

            int rq = (int) ((float) norm * requant_scale);
            if (rq < -128) {
                rq = -128;
            }
            if (rq > 127) {
                rq = 127;
            }
            *(output_ptr + x) = (int8_t) rq;
        }
    },
    input, output);
}

const char *CpuIBERTLayerNormKernel::name() const
{
    return _name.c_str();
}

size_t CpuIBERTLayerNormKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);
    ARM_COMPUTE_UNUSED(platform);

    return ICPPKernel::default_mws;
}

void CpuCharlesSoftmaxKernel::configure(const ITensorInfo *src, ITensorInfo *dst, int offset)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src, *dst));

    _name = std::string("CpuCharlesSoftmaxKernel");
    _offset = offset;
    
    auto win_config = validate_and_configure_window(*src, *dst);

    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICpuKernel::configure(win_config.second);
}

Status CpuCharlesSoftmaxKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src, *dst));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(*src->clone(), *dst->clone()).first);

    return Status{};
}

void CpuCharlesSoftmaxKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC);
    const ITensor *dst = tensors.get_tensor(TensorType::ACL_DST);

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int window_step_x = 16;
    const int window_start_x = static_cast<int>(window.x().start());
    const int window_end_x = static_cast<int>(window.x().end());
    
    const float dst_scale = dst->info()->quantization_info().uniform().scale;

    Iterator input(src, win);
    Iterator output(dst, win);

    const float requant_scale = (1.0f / 127.0f) / dst_scale;

    const int8x16_t offset_vec = vmovq_n_s8(_offset);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto input_ptr = reinterpret_cast<const int8_t *>(input.ptr());
        const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

        int32x4_t sum_vec = vmovq_n_s32(0);
        
        int x = window_start_x;
        for (; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const int8x16_t a = vld1q_s8(input_ptr + x);
            const int16x8_t offset_low = vaddl_s8(vget_low_s8(a), vget_low_s8(offset_vec));
            const int16x8_t offset_high = vaddl_high_s8(a, offset_vec);
            
            sum_vec = vaddq_s32(sum_vec, vpaddlq_s16(offset_low));
            sum_vec = vaddq_s32(sum_vec, vpaddlq_s16(offset_high));
        }

        int sum = vaddvq_s32(sum_vec);
        // std::cout << "sum = " << sum << std::endl;

        for (; x < window_end_x; ++x)
        {
            int a = static_cast<int32_t>(*(input_ptr + x));
            sum += a;
        }

        int factor = (1 << 30) / sum;

        x = window_start_x;
        for (; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            int8x16_t a = vld1q_s8(input_ptr + x);
            int16x8_t offset_low = vaddl_s8(vget_low_s8(a), vget_low_s8(offset_vec));
            int16x8_t offset_high = vaddl_high_s8(a, offset_vec);

            int32x4x4_t offset_vals = {
                {
                    vmovl_s16(vget_low_s16(offset_low)),
                    vmovl_high_s16(offset_low),
                    vmovl_s16(vget_low_s16(offset_high)),
                    vmovl_high_s16(offset_high)
                }
            };

            int32x4x4_t qout = {
                {
                    vshrq_n_s32(vmulq_n_s32(offset_vals.val[0], factor), 23),
                    vshrq_n_s32(vmulq_n_s32(offset_vals.val[1], factor), 23),
                    vshrq_n_s32(vmulq_n_s32(offset_vals.val[2], factor), 23),
                    vshrq_n_s32(vmulq_n_s32(offset_vals.val[3], factor), 23)
                }
            };

            int32x4x4_t requantized = {
                {
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(qout.val[0]), requant_scale)),
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(qout.val[1]), requant_scale)),
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(qout.val[2]), requant_scale)),
                    vcvtq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(qout.val[3]), requant_scale))
                }
            };

            const int16x8_t result_low_shorts = vqmovn_high_s32(vqmovn_s32(requantized.val[0]), requantized.val[1]);
            const int16x8_t result_high_shorts = vqmovn_high_s32(vqmovn_s32(requantized.val[2]), requantized.val[3]);
            const int8x16_t result = vqmovn_high_s16(vqmovn_s16(result_low_shorts), result_high_shorts);
            
            vst1q_s8(output_ptr + x, result);
        }

        for (; x < window_end_x; ++x)
        {
            int a = static_cast<int32_t>(*(input_ptr + x));
            int qout = ((a + _offset) * factor) >> 23;
            int qdst = (int) ((float) qout * requant_scale);
            if (qdst < -128) {
                qdst = -128;
            }
            if (qdst > 127) {
                qdst = 127;
            }
            *(output_ptr + x) = (int8_t) qdst;
        }

    },
    input, output);
}

const char *CpuCharlesSoftmaxKernel::name() const
{
    return _name.c_str();
}

size_t CpuCharlesSoftmaxKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);
    ARM_COMPUTE_UNUSED(platform);
    
    return ICPPKernel::default_mws;
}

}
}
}
 