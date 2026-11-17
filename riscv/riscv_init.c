/* riscv_init.c - RISC-V Vector optimized filter functions
 *
 * Copyright (c) 2023 Google LLC
 * Written by Dragoș Tiselice <dtiselice@google.com>, May 2023.
 *            Filip Wasil     <f.wasil@samsung.com>, March 2025.
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

#include "../pngpriv.h"

#ifdef PNG_READ_SUPPORTED

#if PNG_RISCV_RVV_IMPLEMENTATION == 1

#include <riscv_vector.h>

#ifndef PNG_ALIGNED_MEMORY_SUPPORTED
#  error "ALIGNED_MEMORY is required; set: -DPNG_ALIGNED_MEMORY_SUPPORTED"
#endif

void
png_init_filter_functions_rvv(png_structp pp, unsigned int bpp)
{
   png_debug(1, "in png_init_filter_functions_rvv");

   pp->read_filter[PNG_FILTER_VALUE_UP-1] = png_read_filter_row_up_rvv;

   if (bpp == 3)
   {
      /* 守卫（2026-11-11 X100 实测，16KB 行内核）：
       *   avg3   0.97x（串行链 vaaddu 延迟已到界，VL=3 发射开销吃掉收益）
       *   paeth3 0.78x（u16 域逐像素预测器，3 lane 喂不饱指令开销）
       * 两者 < 1x，按内核级守卫回退 C（bit-exact 由构造保证）。
       * 保留：sub3 1.83x（m4 Hillis-Steele 扫描）、up 13x。 */
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub3_rvv;
   }
   else if (bpp == 4)
   {
      /* it2-guard: paeth4 X100 measured 0.89x (23.5 vs 26.4 MP/s, stable 6x)
       * -> keep C. avg4 +5% and sub4 (m4 scan) stay RVV. */
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg4_rvv;
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub4_rvv;

   }
}

#endif /* PNG_RISCV_RVV_IMPLEMENTATION == 1 */
#endif /* PNG_READ_SUPPORTED */
