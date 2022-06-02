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
void neon_qasymm8_meanstddevnorm(ITensor *input, ITensor *output, float epsilon, const Window &window)
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

    const int32x4_t max_vec = vdupq_n_s32(0);
    const int32x4_t min_vec = vdupq_n_s32(255);

    execute_window_loop(win, [&](const Coordinates &)
    {
        int x = window_start_x;
        auto in_ptr = reinterpret_cast<const uint8_t *>(input_itr.ptr());
        auto out_ptr = reinterpret_cast<uint8_t *>(output_itr.ptr());
        /*
        uint32x4_t sum_vec = vdupq_n_u32(0);
        uint32x4_t sum_sq_vec = vdupq_n_u32(0);

        for (; x <= (window_end_x - window_step_x); x += window_step_x) {
            const uint8x16_t data = vld1q_u8(in_ptr + x);
            
            const uint16x8_t datalow = vmovl_u8(vget_low_u8(data));
            const uint16x8_t datahigh = vmovl_high_u8(data);

            const uint32x4_t db1 = vmovl_u16(vget_low_u16(datalow));
            const uint32x4_t db2 = vmovl_high_u16(datalow);

            const uint32x4_t db3 = vmovl_u16(vget_low_u16(datahigh));
            const uint32x4_t db4 = vmovl_high_u16(datahigh);

            sum_vec = vaddq_u32(sum_vec, db1);
            sum_vec = vaddq_u32(sum_vec, db2);
            sum_vec = vaddq_u32(sum_vec, db3);
            sum_vec = vaddq_u32(sum_vec, db4);
            
            sum_sq_vec = vaddq_u32(sum_sq_vec, vmulq_u32(db1, db1));
            sum_sq_vec = vaddq_u32(sum_sq_vec, vmulq_u32(db2, db2));
            sum_sq_vec = vaddq_u32(sum_sq_vec, vmulq_u32(db3, db3));
            sum_sq_vec = vaddq_u32(sum_sq_vec, vmulq_u32(db4, db4));
        }

        uint32_t sum = vaddvq_u32(sum_vec);
        uint32_t sum_sq = vaddvq_u32(sum_sq_vec);
        */
        uint32_t sum = 0;
        uint32_t sum_sq = 0;
        // std::cout << "vector only: " << "sum = " << sum << ", sum_sq = " << sum_sq << std::endl;

        for (; x < window_end_x; ++x) {
            uint32_t data = static_cast<uint32_t>(*(in_ptr + x));
            sum += data;
            sum_sq += (data * data);
        }

        // std::cout << "sum = " << sum << ", sum_sq = " << sum_sq << std::endl;

        float mean = (float) sum / (float) input->info()->dimension(0);
        float var = ((float) sum_sq / (float) input->info()->dimension(0)) - (mean * mean);
        float stdev_inv = 1.0f / sqrtf(var + epsilon);
        
        x = window_start_x;
        // std::cout << "mean = " << mean << ", var = " << var << std::endl;
        /*
        uint32_t stdev = 256;
        stdev = (stdev + var / stdev) / 2;
        stdev = (stdev + var / stdev) / 2;
        stdev = (stdev + var / stdev) / 2;
        stdev = (stdev + var / stdev) / 2;

        int32x4_t mean_vec = vdupq_n_s32(mean);
        int32x4_t stdev_vec = vdupq_n_s32(stdev);
        */
        /*
        float32x4_t mean_vec = vdupq_n_f32(mean);
        float32x4_t stdev_inv_vec = vdupq_n_f32(stdev_inv);

        for (x = window_start_x; x <= (window_end_x - window_step_x); x++) {
            const uint8x16_t data = vld1q_u8(in_ptr + x);

            const uint16x8_t datalow = vmovl_u8(vget_low_u8(data));
            const uint16x8_t datahigh = vmovl_high_u8(data);

            const float32x4_t db1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(datalow)));
            const float32x4_t db2 = vcvtq_f32_u32(vmovl_high_u16(datalow));

            const float32x4_t db3 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(datahigh)));
            const float32x4_t db4 = vcvtq_f32_u32(vmovl_high_u16(datahigh));

            // const uint32x4_t normb1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vaddq_s32(vcvtq_s32_f32(vmulq_f32(vmulq_f32(vsubq_f32(db1, mean_vec), stdev_inv_vec), output_inv_scale_vec)), output_offset_vec), max_vec), min_vec));
            // const uint32x4_t normb2 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vaddq_s32(vcvtq_s32_f32(vmulq_f32(vmulq_f32(vsubq_f32(db2, mean_vec), stdev_inv_vec), output_inv_scale_vec)), output_offset_vec), max_vec), min_vec));
            // const uint32x4_t normb3 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vaddq_s32(vcvtq_s32_f32(vmulq_f32(vmulq_f32(vsubq_f32(db3, mean_vec), stdev_inv_vec), output_inv_scale_vec)), output_offset_vec), max_vec), min_vec));
            // const uint32x4_t normb4 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vaddq_s32(vcvtq_s32_f32(vmulq_f32(vmulq_f32(vsubq_f32(db4, mean_vec), stdev_inv_vec), output_inv_scale_vec)), output_offset_vec), max_vec), min_vec));
            const float32x4_t normb1 = vmulq_f32(vsubq_f32(db1, mean_vec), stdev_inv_vec);
            const float32x4_t quantb1 = vaddq_f32(vmulq_f32(normb1, output_inv_scale_vec), output_offset_vec);
            const uint32x4_t outb1 = vreinterpretq_u32_s32(vmaxq_s32(vminq_s32(vcvtq_s32_f32(quantb1), min_vec), max_vec));

            const float32x4_t normb2 = vmulq_f32(vsubq_f32(db2, mean_vec), stdev_inv_vec);
            const float32x4_t quantb2 = vaddq_f32(vmulq_f32(normb2, output_inv_scale_vec), output_offset_vec);
            const uint32x4_t outb2 = vreinterpretq_u32_s32(vmaxq_s32(vminq_s32(vcvtq_s32_f32(quantb2), min_vec), max_vec));

            const float32x4_t normb3 = vmulq_f32(vsubq_f32(db3, mean_vec), stdev_inv_vec);
            const float32x4_t quantb3 = vaddq_f32(vmulq_f32(normb3, output_inv_scale_vec), output_offset_vec);
            const uint32x4_t outb3 = vreinterpretq_u32_s32(vmaxq_s32(vminq_s32(vcvtq_s32_f32(quantb3), min_vec), max_vec));

            const float32x4_t normb4 = vmulq_f32(vsubq_f32(db4, mean_vec), stdev_inv_vec);
            const float32x4_t quantb4 = vaddq_f32(vmulq_f32(normb4, output_inv_scale_vec), output_offset_vec);
            const uint32x4_t outb4 = vreinterpretq_u32_s32(vmaxq_s32(vminq_s32(vcvtq_s32_f32(quantb4), min_vec), max_vec));

            const uint16x8_t outdatalow = vmovn_high_u32(vmovn_u32(outb1), outb2);
            const uint16x8_t outdatahigh = vmovn_high_u32(vmovn_u32(outb3), outb4);

            const uint8x16_t outdata = vmovn_high_u16(vmovn_u16(outdatalow), outdatahigh);

            vst1q_u8(out_ptr + x, outdata);
        }
        */
        for (; x < window_end_x; ++x) {
            // normed = (data - mean) / stdev
            // quantized = (normed / scale) + offset

            float32_t data = static_cast<float32_t>(*(in_ptr + x));
            float32_t normalized = (data - mean) * stdev_inv;
            float32_t quantized =  normalized * output_inv_scale + output_offset;
            // std::cout << data << " " << normalized << " " << quantized << std::endl;
            *(out_ptr + x) = static_cast<uint8_t>(arm_compute::utility::clamp<uint8_t>(quantized));
        }
        // std::cout << std::endl;

    }, input_itr, output_itr);
}
} // namespace cpu
} // namespace arm_compute
