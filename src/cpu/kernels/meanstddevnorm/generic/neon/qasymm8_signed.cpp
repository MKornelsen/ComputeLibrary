/*
 * Copyright (c) 2022 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
// #include "src/cpu/kernels/meanstddevnorm/generic/neon/impl.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
void neon_qasymm8_signed_meanstddevnorm(ITensor *input, ITensor *output, float epsilon, const Window &window)
{
    // return mean_stddev_normalization<qasymm8_t, 4>(input, output, epsilon, window);
    
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int window_step_x = 16;
    const int window_start_x = static_cast<int>(window.x().start());
    const int window_end_x = static_cast<int>(window.x().end());

    // const float input_scale = input->info()->quantization_info().scale()[0];
    // const int input_offset = input->info()->quantization_info().offset()[0];

    const float output_scale = output->info()->quantization_info().scale()[0];
    const int output_offset = output->info()->quantization_info().offset()[0];

    // const uint32_t output_multiplier = static_cast<uint32_t>(1.0f / output_scale);

    Iterator input_itr(input, win);
    Iterator output_itr(output, win);

    const float output_inv_scale = 1.0f / output_scale;
    const float32x4_t output_inv_scale_vec = vdupq_n_f32(output_inv_scale);
    const float32x4_t output_offset_vec = vdupq_n_f32(output_offset);

    // const int32x4_t max_vec = vdupq_n_s32(0);
    // const int32x4_t min_vec = vdupq_n_s32(255);

    const float32x4_t quant_max_vec = vdupq_n_f32(127.0f);
    const float32x4_t quant_min_vec = vdupq_n_f32(-128.0f);

    execute_window_loop(win, [&](const Coordinates &)
    {
        int x = window_start_x;
        auto in_ptr = reinterpret_cast<const int8_t *>(input_itr.ptr());
        auto out_ptr = reinterpret_cast<int8_t *>(output_itr.ptr());

        uint32x4_t sum_vec = vdupq_n_s32(0);
        uint32x4_t sum_sq_vec = vdupq_n_s32(0);

        for (; x <= (window_end_x - window_step_x); x += window_step_x) {
            const int8x16_t data = vld1q_s8(in_ptr + x);
            
            sum_vec = vaddq_u32(sum_vec, vpaddlq_s16(vpaddlq_s8(data)));

            const int16x8_t squares_low = vmull_s8(vget_low_s8(data), vget_low_s8(data));
            const int16x8_t squares_high = vmull_high_s8(data, data);
            sum_sq_vec = vaddq_s32(sum_sq_vec, vaddq_s32(vpaddlq_s16(squares_low), vpaddlq_s16(squares_high)));
        }

        int32_t sum = vaddvq_s32(sum_vec);
        int32_t sum_sq = vaddvq_s32(sum_sq_vec);
        
        // std::cout << "vector only: " << "sum = " << sum << ", sum_sq = " << sum_sq << std::endl;

        for (; x < window_end_x; ++x) {
            int32_t data = static_cast<int32_t>(*(in_ptr + x));
            sum += data;
            sum_sq += (data * data);
        }

        // std::cout << "sum = " << sum << ", sum_sq = " << sum_sq << std::endl;

        float mean = (float) sum / (float) input->info()->dimension(0);
        float var = ((float) sum_sq / (float) input->info()->dimension(0)) - (mean * mean);
        float stdev_inv = 1.0f / sqrtf(var + epsilon);
        
        // std::cout << "mean = " << mean << std::endl;
        // std::cout << "var = " << var << std::endl;

        x = window_start_x;
        
        float32x4_t mean_vec = vdupq_n_f32(mean);
        float32x4_t stdev_inv_vec = vdupq_n_f32(stdev_inv);
        for (x = window_start_x; x <= (window_end_x - window_step_x); x+=window_step_x) {
            const int8x16_t data = vld1q_s8(in_ptr + x);

            float32x4_t db1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(data)))));
            float32x4_t db2 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(data)))));
            float32x4_t db3 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(data)))));
            float32x4_t db4 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(data)))));

            #define NORMALIZE(block) vmulq_f32(stdev_inv_vec, vsubq_f32(block, mean_vec))
            #define QUANTIZE(block) vaddq_f32(vmulq_f32(block, output_inv_scale_vec), output_offset_vec)
            #define CLAMP(block) vminq_f32(vmaxq_f32(block, quant_min_vec), quant_max_vec);

            db1 = CLAMP(QUANTIZE(NORMALIZE(db1)));
            db2 = CLAMP(QUANTIZE(NORMALIZE(db2)));
            db3 = CLAMP(QUANTIZE(NORMALIZE(db3)));
            db4 = CLAMP(QUANTIZE(NORMALIZE(db4)));

            #define FUSEWORDS(fb1, fb2) vmovn_high_s32(vmovn_s32(vcvtq_s32_f32(fb1)), vcvtq_s32_f32(fb2))
            #define FUSESHORTS(sb1, sb2) vmovn_high_s16(vmovn_s16(sb1), sb2)

            int8x16_t out = FUSESHORTS(FUSEWORDS(db1, db2), FUSEWORDS(db3, db4));
            vst1q_s8(out_ptr + x, out);
        }

        for (; x < window_end_x; ++x) {

            float32_t data = static_cast<float32_t>(*(in_ptr + x));
            float32_t normalized = (data - mean) * stdev_inv;
            float32_t quantized =  normalized * output_inv_scale + output_offset;
            // std::cout << data << " " << normalized << " " << quantized << std::endl;
            *(out_ptr + x) = static_cast<int8_t>(arm_compute::utility::clamp<int8_t>(quantized));
        }
        // std::cout << std::endl;

    }, input_itr, output_itr);
}
} // namespace cpu
} // namespace arm_compute
