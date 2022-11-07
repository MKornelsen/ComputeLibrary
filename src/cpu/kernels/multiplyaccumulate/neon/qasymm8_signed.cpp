#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
void multiply_accumulate_qasymm8_signed_neon(const ITensor *src0, const ITensor *src1, const ITensor *src2, ITensor *dst, const Window &window)
{
    Window src0_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
    Window src1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window src2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    Window win = window;

    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    constexpr int window_step_x = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x = static_cast<int>(window.x().end());
    
    src0_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    src1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    src2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const UniformQuantizationInfo src0_qinfo = src0->info()->quantization_info().uniform();
    const UniformQuantizationInfo src1_qinfo = src1->info()->quantization_info().uniform();
    const UniformQuantizationInfo src2_qinfo = src2->info()->quantization_info().uniform();
    const UniformQuantizationInfo dst_qinfo = dst->info()->quantization_info().uniform();

    Iterator input0(src0, src0_win);
    Iterator input1(src1, src1_win);
    Iterator input2(src2, src2_win);
    Iterator output(dst, win);

    const float32x4_t vscale_0 = vdupq_n_f32(src0_qinfo.scale);
    const float32x4_t vscale_1 = vdupq_n_f32(src1_qinfo.scale);
    const float32x4_t vscale_2 = vdupq_n_f32(src2_qinfo.scale);

    const int32x4_t voffset_0 = vdupq_n_s32(src0_qinfo.offset);
    const int32x4_t voffset_1 = vdupq_n_s32(src1_qinfo.offset);
    const int32x4_t voffset_2 = vdupq_n_s32(src2_qinfo.offset);

    const float32x4_t vscale_out = vdupq_n_f32(1.0f / dst_qinfo.scale);
    const float32x4_t voffset_out = vdupq_n_f32((float) dst_qinfo.offset);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto input0_ptr = reinterpret_cast<const int8_t *>(input0.ptr());
        const auto input1_ptr = reinterpret_cast<const int8_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const int8_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

        int x = window_start_x;
        for (; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const int8x16_t a = vld1q_s8(input0_ptr + x);
            const int8x16_t b = vld1q_s8(input1_ptr + x);
            const int8x16_t c = vld1q_s8(input2_ptr + x);
            
            // const int16x8_t low_mac = vmlal_s8(vmovl_s8(c), vget_low_s8(a), vget_low_s8(b));
            // const int16x8_t high_mac = vmlal_high_s8(vmovl_high_s8(c), a, b);
            const auto af_0 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(a)))), voffset_0)), vscale_0);
            const auto af_1 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(a)))), voffset_0)), vscale_0);
            const auto af_2 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(a)))), voffset_0)), vscale_0);
            const auto af_3 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(a)))), voffset_0)), vscale_0);

            const auto bf_0 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(b)))), voffset_1)), vscale_1);
            const auto bf_1 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(b)))), voffset_1)), vscale_1);
            const auto bf_2 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(b)))), voffset_1)), vscale_1);
            const auto bf_3 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(b)))), voffset_1)), vscale_1);
            
            const auto cf_0 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(c)))), voffset_2)), vscale_2);
            const auto cf_1 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(c)))), voffset_2)), vscale_2);
            const auto cf_2 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(c)))), voffset_2)), vscale_2);
            const auto cf_3 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(c)))), voffset_2)), vscale_2);

            const auto rf_0 = vcvtnq_s32_f32(vmlaq_f32(voffset_out, vmlaq_f32(cf_0, af_0, bf_0), vscale_out));
            const auto rf_1 = vcvtnq_s32_f32(vmlaq_f32(voffset_out, vmlaq_f32(cf_1, af_1, bf_1), vscale_out));
            const auto rf_2 = vcvtnq_s32_f32(vmlaq_f32(voffset_out, vmlaq_f32(cf_2, af_2, bf_2), vscale_out));
            const auto rf_3 = vcvtnq_s32_f32(vmlaq_f32(voffset_out, vmlaq_f32(cf_3, af_3, bf_3), vscale_out));

            const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(rf_0), vqmovn_s32(rf_1)));
            const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(rf_2), vqmovn_s32(rf_3)));
            vst1q_s8(output_ptr + x, vcombine_s8(pa, pb));
            // vst1q_s8(output_ptr + x, a);
        }

        for (; x < window_end_x; ++x)
        {
            // *(output_ptr + x) = 0;
            const float af = (static_cast<int32_t>(*(input0_ptr + x)) - src0_qinfo.offset) * src0_qinfo.scale;
            const float bf = (static_cast<int32_t>(*(input1_ptr + x)) - src1_qinfo.offset) * src1_qinfo.scale;
            const float cf = (static_cast<int32_t>(*(input2_ptr + x)) - src2_qinfo.offset) * src2_qinfo.scale;
            *(output_ptr + x) = quantize_qasymm8_signed((af * bf + cf), dst_qinfo);
        }

    }, input0, input1, input2, output);
}
}
}