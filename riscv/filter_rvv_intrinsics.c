/* filter_rvv_intrinsics.c - RISC-V Vector optimized filter functions
 *
 * Copyright (c) 2023 Google LLC
 * Written by Manfred SCHLAEGL, 2022
 * Revised by:
 *  - Dragoș Tiselice <dtiselice@google.com>, May 2023
 *  - Filip Wasil <f.wasil@samsung.com>, March 2025
 *  - Liang Junzhao <junzhao.liang@spacemit.com>, November 2025
 *  - Alexander Smorkalov <alexander.smorkalov@opencv.ai>, December 2025
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

#include "../pngpriv.h"

#ifdef PNG_READ_SUPPORTED

#if PNG_RISCV_RVV_IMPLEMENTATION == 1 /* intrinsics code from pngpriv.h */

#include <riscv_vector.h>

void
png_read_filter_row_up_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   for (size_t vl; len > 0; len -= vl, row += vl, prev_row += vl)
   {
      vl = __riscv_vsetvl_e8m8(len);

      vuint8m8_t prev_vals = __riscv_vle8_v_u8m8(prev_row, vl);
      vuint8m8_t row_vals = __riscv_vle8_v_u8m8(row, vl);

      row_vals = __riscv_vadd_vv_u8m8(row_vals, prev_vals, vl);

      __riscv_vse8_v_u8m8(row, row_vals, vl);
   }
}

static inline void
png_read_filter_row_sub_rvv(size_t len, size_t bpp, unsigned char *row)
{
   /* it1b: m4 block (128B) Hillis-Steele prefix-sum scan.
    * v[i] += vslideup(v, s), s = bpp*2^k; u8 wrap == mod-256 (bit-exact vs C).
    * Cross-block carry broadcast via vrgather(idx = lane % bpp).
    * m8 makes GCC spill the reg group (vs8r.v) -> slower; m4 is optimum. */
   const size_t VB = __riscv_vsetvlmax_e8m4();
   vuint8m4_t idx = __riscv_vremu_vx_u8m4(
       __riscv_vid_v_u8m4(VB), (size_t)bpp, VB);
   vuint8m4_t carry_at = __riscv_vmv_v_x_u8m4(0, VB);
   const vuint8m4_t vzero = __riscv_vmv_v_x_u8m4(0, VB);

   while (len > 0)
   {
      size_t vl = __riscv_vsetvl_e8m4(len);
      vuint8m4_t v = __riscv_vle8_v_u8m4(row, vl);

      for (size_t s = bpp; s < vl; s *= 2)
         v = __riscv_vadd_vv_u8m4(
             v, __riscv_vslideup_vx_u8m4(vzero, v, s, vl), vl);

      v = __riscv_vadd_vv_u8m4(
          v, __riscv_vrgather_vv_u8m4(carry_at, idx, vl), vl);

      __riscv_vse8_v_u8m4(row, v, vl);

      if (len > vl)
         carry_at = __riscv_vslidedown_vx_u8m4(v, vl - bpp, vl);

      row += vl; len -= vl;
   }
}

/* 标量 sub 兜底：pngrutil.c 的 C 实现是 static 调不到，本地自写（数学等价）。 */
static inline void
sub_scalar(unsigned char *row, size_t len, size_t bpp)
{
   for (size_t i = bpp; i < len; i++)
      row[i] = (unsigned char)((int)row[i] + (int)row[i - bpp]);
}

/* 短行守卫：m4 前缀和固定开销 ~160-230 cycles，X100 实测 len<96B 时
 * RVV 0.04-0.68x（负优化），>=96B 转正 1.24x+。窄图走标量兜底。 */
#define SUB_GUARD_THRESHOLD 96

void
png_read_filter_row_sub3_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   if (len < SUB_GUARD_THRESHOLD)
   {
      sub_scalar(row, len, 3);
      PNG_UNUSED(prev_row)
      return;
   }

   png_read_filter_row_sub_rvv(len, 3, row);

   PNG_UNUSED(prev_row)
}

void
png_read_filter_row_sub4_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   if (len < SUB_GUARD_THRESHOLD)
   {
      sub_scalar(row, len, 4);
      PNG_UNUSED(prev_row)
      return;
   }

   png_read_filter_row_sub_rvv(len, 4, row);

   PNG_UNUSED(prev_row)
}

static inline void
png_read_filter_row_avg_rvv(size_t len, size_t bpp, unsigned char *row,
    const unsigned char *prev_row)
{
   /* it2: 4x 展开 + 预取。串行链（每像素 vaaddu+vadd ≈4 cycle）无法
    * 消除，但把每像素 9 条指令的发射开销摊薄；b/x 预取与链计算重叠。 */
   unsigned char* rp_end = row + len;

   size_t vl = __riscv_vsetvl_e8m1(bpp);
   vuint8m1_t a;
   vuint8m1_t b, x;

   /* first pixel */
   b = __riscv_vle8_v_u8m1(prev_row, vl);
   x = __riscv_vle8_v_u8m1(row, vl);
   b = __riscv_vsrl_vx_u8m1(b, 1, vl);
   a = __riscv_vadd_vv_u8m1(b, x, vl);
   __riscv_vse8_v_u8m1(row, a, vl);
   row += bpp; prev_row += bpp;

   /* 4x 展开主体 */
   while (row + 4 * bpp <= rp_end)
   {
      vuint8m1_t b0 = __riscv_vle8_v_u8m1(prev_row, vl);
      vuint8m1_t x0 = __riscv_vle8_v_u8m1(row, vl);
      vuint8m1_t b1 = __riscv_vle8_v_u8m1(prev_row + bpp, vl);
      vuint8m1_t x1 = __riscv_vle8_v_u8m1(row + bpp, vl);
      vuint8m1_t b2 = __riscv_vle8_v_u8m1(prev_row + 2 * bpp, vl);
      vuint8m1_t x2 = __riscv_vle8_v_u8m1(row + 2 * bpp, vl);
      vuint8m1_t b3 = __riscv_vle8_v_u8m1(prev_row + 3 * bpp, vl);
      vuint8m1_t x3 = __riscv_vle8_v_u8m1(row + 3 * bpp, vl);

      vuint8m1_t a0 = __riscv_vaaddu_vv_u8m1(a, b0, __RISCV_VXRM_RDN, vl);
      a0 = __riscv_vadd_vv_u8m1(a0, x0, vl);
      __riscv_vse8_v_u8m1(row, a0, vl);
      vuint8m1_t a1 = __riscv_vaaddu_vv_u8m1(a0, b1, __RISCV_VXRM_RDN, vl);
      a1 = __riscv_vadd_vv_u8m1(a1, x1, vl);
      __riscv_vse8_v_u8m1(row + bpp, a1, vl);
      vuint8m1_t a2 = __riscv_vaaddu_vv_u8m1(a1, b2, __RISCV_VXRM_RDN, vl);
      a2 = __riscv_vadd_vv_u8m1(a2, x2, vl);
      __riscv_vse8_v_u8m1(row + 2 * bpp, a2, vl);
      vuint8m1_t a3 = __riscv_vaaddu_vv_u8m1(a2, b3, __RISCV_VXRM_RDN, vl);
      a3 = __riscv_vadd_vv_u8m1(a3, x3, vl);
      __riscv_vse8_v_u8m1(row + 3 * bpp, a3, vl);

      a = a3;
      row += 4 * bpp; prev_row += 4 * bpp;
   }

   while (row < rp_end)
   {
      b = __riscv_vle8_v_u8m1(prev_row, vl);
      x = __riscv_vle8_v_u8m1(row, vl);
      a = __riscv_vaaddu_vv_u8m1(a, b, __RISCV_VXRM_RDN, vl);
      a = __riscv_vadd_vv_u8m1(a, x, vl);
      __riscv_vse8_v_u8m1(row, a, vl);
      row += bpp; prev_row += bpp;
   }
}

void
png_read_filter_row_avg3_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   png_read_filter_row_avg_rvv(len, 3, row, prev_row);

   PNG_UNUSED(prev_row)
}

void
png_read_filter_row_avg4_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   png_read_filter_row_avg_rvv(len, 4, row, prev_row);

   PNG_UNUSED(prev_row)
}

/* paeth: u16 域逐像素预测器（|x| = vminu(x, -x)，三级选择），
 * 4x 展开摊薄发射开销。串行链 a=merge(...)+x 无法并行。 */


/* paeth3: 4x 展开体沿用（bpp=3 派发已守卫回退 C，仅保留实现作对照）。 */
static inline void
png_read_filter_row_paeth_rvv(size_t len, size_t bpp, unsigned char *row,
    const unsigned char *prev)
{
   unsigned char* rp_end = row + len;

   size_t vl = __riscv_vsetvl_e8m1(bpp);

   /* first pixel */
   vuint8m1_t a = __riscv_vle8_v_u8m1(row, vl);
   vuint8m1_t c = __riscv_vle8_v_u8m1(prev, vl);
   a = __riscv_vadd_vv_u8m1(a, c, vl);
   __riscv_vse8_v_u8m1(row, a, vl);
   row += bpp; prev += bpp;

   while (row + 4 * bpp <= rp_end)
   {
      for (int k = 0; k < 4; k++)
      {
         vuint8m1_t b = __riscv_vle8_v_u8m1(prev, vl);
         vuint8m1_t x = __riscv_vle8_v_u8m1(row, vl);

         vuint16m2_t p = __riscv_vwsubu_vv_u16m2(b, c, vl);
         vuint16m2_t pc = __riscv_vwsubu_vv_u16m2(a, c, vl);

         vuint16m2_t tmp = __riscv_vrsub_vx_u16m2(p, 0, vl);
         vuint16m2_t pa = __riscv_vminu_vv_u16m2(p, tmp, vl);

         tmp = __riscv_vrsub_vx_u16m2(pc, 0, vl);
         vuint16m2_t pb = __riscv_vminu_vv_u16m2(pc, tmp, vl);

         pc = __riscv_vadd_vv_u16m2(p, pc, vl);
         tmp = __riscv_vrsub_vx_u16m2(pc, 0, vl);
         pc = __riscv_vminu_vv_u16m2(pc, tmp, vl);

         vbool8_t m1 = __riscv_vmsltu_vv_u16m2_b8(pb, pa, vl);
         pa = __riscv_vmerge_vvm_u16m2(pa, pb, m1, vl);
         a = __riscv_vmerge_vvm_u8m1(a, b, m1, vl);

         vbool8_t m2 = __riscv_vmsltu_vv_u16m2_b8(pc, pa, vl);
         a = __riscv_vmerge_vvm_u8m1(a, c, m2, vl);

         a = __riscv_vadd_vv_u8m1(a, x, vl);
         __riscv_vse8_v_u8m1(row, a, vl);

         c = b;
         row += bpp; prev += bpp;
      }
   }

   while (row < rp_end)
   {
      vuint8m1_t b = __riscv_vle8_v_u8m1(prev, vl);
      vuint8m1_t x = __riscv_vle8_v_u8m1(row, vl);

      vuint16m2_t p = __riscv_vwsubu_vv_u16m2(b, c, vl);
      vuint16m2_t pc = __riscv_vwsubu_vv_u16m2(a, c, vl);

      vuint16m2_t tmp = __riscv_vrsub_vx_u16m2(p, 0, vl);
      vuint16m2_t pa = __riscv_vminu_vv_u16m2(p, tmp, vl);

      tmp = __riscv_vrsub_vx_u16m2(pc, 0, vl);
      vuint16m2_t pb = __riscv_vminu_vv_u16m2(pc, tmp, vl);

      pc = __riscv_vadd_vv_u16m2(p, pc, vl);
      tmp = __riscv_vrsub_vx_u16m2(pc, 0, vl);
      pc = __riscv_vminu_vv_u16m2(pc, tmp, vl);

      vbool8_t m1 = __riscv_vmsltu_vv_u16m2_b8(pb, pa, vl);
      pa = __riscv_vmerge_vvm_u16m2(pa, pb, m1, vl);
      a = __riscv_vmerge_vvm_u8m1(a, b, m1, vl);

      vbool8_t m2 = __riscv_vmsltu_vv_u16m2_b8(pc, pa, vl);
      a = __riscv_vmerge_vvm_u8m1(a, c, m2, vl);

      a = __riscv_vadd_vv_u8m1(a, x, vl);
      __riscv_vse8_v_u8m1(row, a, vl);

      c = b;
      row += bpp; prev += bpp;
   }
}

void
png_read_filter_row_paeth3_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   png_read_filter_row_paeth_rvv(len, 3, row, prev_row);
}

void
png_read_filter_row_paeth4_rvv(png_row_infop row_info, png_bytep row,
    png_const_bytep prev_row)
{
   size_t len = row_info->rowbytes;

   png_read_filter_row_paeth_rvv(len, 4, row, prev_row);
}

#endif /* PNG_RISCV_RVV_IMPLEMENTATION == 1 */
#endif /* PNG_READ_SUPPORTED */
