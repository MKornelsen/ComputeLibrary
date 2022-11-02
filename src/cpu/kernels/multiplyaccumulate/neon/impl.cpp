#include "src/cpu/kernels/add/generic/neon/impl.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace cpu
{

inline float32x4_t float_vmla(const float32x4_t &a, const float32x4_t &b, const float32x4_t &c)
{
    return vfmaq_f32(a, b, c);
}

inline float16x8_t float_vmla(const float16x8_t &a, const float16x8_t &b, const float16x8_t &c)
{
    return vfmaq_f16(a, b, c);
}

template <typename ScalarType>
void multiply_accumulate_float(const ITensor *src0, const ITensor *src1, const ITensor *src2, ITensor *dst, const Window &window)
{

    Window src0_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
    Window src1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
    Window src2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

    Window win = window;

    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    constexpr int window_step_x = 16 / sizeof(ScalarType);
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x = static_cast<int>(window.x().end());
    
    src0_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    src1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    src2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input0(src0, src0_win);
    Iterator input1(src1, src1_win);
    Iterator input2(src2, src2_win);
    Iterator output(dst, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto input0_ptr = reinterpret_cast<const ScalarType *>(input0.ptr());
        const auto input1_ptr = reinterpret_cast<const ScalarType *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const ScalarType *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<ScalarType *>(output.ptr());

        int x = window_start_x;
        for (; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto val0 = wrapper::vloadq(input0_ptr + x);
            const auto val1 = wrapper::vloadq(input1_ptr + x);
            const auto val2 = wrapper::vloadq(input2_ptr + x);

            // const auto res = wrapper::vmla(val0, val1, val2);
            const auto res = float_vmla(val2, val0, val1);

            wrapper::vstore(output_ptr + x, res);
        }

        for (; x < window_end_x; ++x)
        {
            const auto val0 = *(input0_ptr + x);
            const auto val1 = *(input1_ptr + x);
            const auto val2 = *(input2_ptr + x);

            *(output_ptr + x) = val0 * val1 + val2;
        }
    },
    input0, input1, input2, output);
}

template void multiply_accumulate_float<float>(const ITensor *src0, const ITensor *src1, const ITensor *src2, ITensor *dst, const Window &window);

template void multiply_accumulate_float<float16_t>(const ITensor *src0, const ITensor *src1, const ITensor *src2, ITensor *dst, const Window &window);

}
}
