#include "asm_kernels.h"

#include <cstring>

#include <cstdint>

#include <arm_neon.h>


extern "C" {


void rgb_to_chw_fp32_asm(

    const uint8_t* src,

    float*         dst_r,

    float*         dst_g,

    float*         dst_b,

    int width,

    int height,

    int src_stride)

{

    const float scale = 1.0f / 255.0f;

    for (int y = 0; y < height; y++) {

        const uint8_t* row = src + y * src_stride;

        float* r = dst_r + y * width;

        float* g = dst_g + y * width;

        float* b = dst_b + y * width;

        int x = 0;

        for (; x <= width - 8; x += 8) {

            uint8x8x3_t rgb = vld3_u8(row + x * 3);

            float32x4_t sc  = vdupq_n_f32(scale);

            vst1q_f32(r + x,     vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(rgb.val[0])))), sc));

            vst1q_f32(r + x + 4, vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(rgb.val[0])))), sc));

            vst1q_f32(g + x,     vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(rgb.val[1])))), sc));

            vst1q_f32(g + x + 4, vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(rgb.val[1])))), sc));

            vst1q_f32(b + x,     vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(rgb.val[2])))), sc));

            vst1q_f32(b + x + 4, vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(rgb.val[2])))), sc));

        }

        for (; x < width; x++) {

            r[x] = row[x*3 + 0] * scale;

            g[x] = row[x*3 + 1] * scale;

            b[x] = row[x*3 + 2] * scale;

        }

    }

}


void memcpy_neon_asm(void* dst, const void* src, size_t bytes)

{

    memcpy(dst, src, bytes);

}

void transpose_84x8400_asm(
    const float* src,
    float* dst,
    int cols,    
    int rows)    
{
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            // dst[col][row] = src[row][col]
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}
void transpose_general_asm(
    const float* src,
    float* dst,
    int cols,    // x2: 8400
    int rows)    // x3: 14 (PPE)
{
    // Giống transpose_84x8400_asm nhưng dùng label khác để khớp với code gọi
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            // dst[c][r] = src[r][c]
            // Trong đó src[r][c] có chỉ số là r * cols + c
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}
} // extern "C" 
