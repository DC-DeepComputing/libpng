/* palette_rvv_intrinsics.c - RISC-V Vector optimized palette expansion
 *
 * Ported from arm/palette_neon_intrinsics.c (Richard Townsend, Arm).
 * RVV implementation: gather-based table lookup using indexed loads.
 *
 * This code is released under the libpng license.
 */

#include "../pngpriv.h"

#ifdef PNG_READ_SUPPORTED

#if PNG_RISCV_RVV_IMPLEMENTATION == 1

#include <riscv_vector.h>

/* Build an RGBA8 palette from the separate RGB and alpha palettes.
 * Called once per image, scalar is fine and bit-exact. */
void
__attribute__((noinline)) png_riffle_palette_rvv(png_structrp png_ptr)
{
   png_const_colorp palette = png_ptr->palette;
   png_bytep riffled = png_ptr->riffled_palette;
   png_const_bytep trans_alpha = png_ptr->trans_alpha;
   int num_trans = png_ptr->num_trans;
   int i;

   png_debug(1, "in png_riffle_palette_rvv");

   for (i = 0; i < 256; i++)
   {
      riffled[i * 4 + 0] = palette[i].red;
      riffled[i * 4 + 1] = palette[i].green;
      riffled[i * 4 + 2] = palette[i].blue;
      riffled[i * 4 + 3] = (i < num_trans) ? trans_alpha[i] : 0xff;
   }
}

/* Expand a palettized row into RGBA8 via gather over the 256-entry
 * pre-riffled u32 table.  ssp/ddp point at last source/output byte;
 * we consume vl indices from the tail, gather forward, hand pointers back.
 * Returns number of pixels consumed. */
int
__attribute__((noinline)) png_do_expand_palette_rgba8_rvv(png_structrp png_ptr, png_row_infop row_info,
    png_const_bytep row, png_bytepp ssp, png_bytepp ddp)
{
   png_uint_32 row_width = row_info->width;
   png_bytep riffle = png_ptr->riffled_palette;
   png_uint_32 i = 0;
   size_t vl;

   PNG_UNUSED(row);

   vl = __riscv_vsetvlmax_e32m4();  /* 64 u32 lanes @ VLEN=256 */
   if (vl > 64) vl = 64;

   while (i + vl <= row_width)
   {
      png_const_bytep sp = *ssp - (vl - 1);
      png_bytep dp = *ddp - (vl * 4 - 1);

      vuint8m1_t idx8 = __riscv_vle8_v_u8m1(sp, vl);
      vuint32m4_t idx = __riscv_vzext_vf4_u32m4(idx8, vl);
      vuint32m4_t off = __riscv_vmul_vx_u32m4(idx, 4, vl);
      vuint32m4_t g = __riscv_vloxei32_v_u32m4((const void *)riffle, off, vl);
      __riscv_vse32_v_u32m4((void *)dp, g, vl);

      *ssp = sp - 1;
      *ddp = dp - 1;
      i += (png_uint_32)vl;
   }

   return (int)i;
}

/* Expand a palettized row into RGB8.  Uses e8m2 indexed loads (EEW=32 index
 * with e8m2 result is the legal LMUL pairing on this toolchain), gathering
 * r/g/b at idx*3+{0,1,2} from the 3-byte-per-entry palette, then strided
 * store interleaves them into the 3-byte-per-pixel output.
 * Returns number of pixels consumed. */
int
__attribute__((noinline)) png_do_expand_palette_rgb8_rvv(png_structrp png_ptr, png_row_infop row_info,
    png_const_bytep row, png_bytepp ssp, png_bytepp ddp)
{
   png_uint_32 row_width = row_info->width;
   png_bytep palette = (png_bytep)png_ptr->palette;
   png_uint_32 i = 0;
   size_t vl;

   PNG_UNUSED(row);

   vl = __riscv_vsetvlmax_e8m2();  /* 64 lanes @ VLEN=256 */

   while (i + vl <= row_width)
   {
      png_const_bytep sp = *ssp - (vl - 1);
      png_bytep dp = *ddp - (vl * 3 - 1);

      vuint8m2_t idx8 = __riscv_vle8_v_u8m2(sp, vl);
      /* index for vloxei32 must be EEW=32; e8m2 result pairs with e32m8 idx */
      vuint32m8_t idx = __riscv_vzext_vf4_u32m8(idx8, vl);
      vuint32m8_t base3 = __riscv_vmul_vx_u32m8(idx, 3, vl);

      vuint8m2_t r = __riscv_vloxei32_v_u8m2((const void *)palette, base3, vl);
      vuint32m8_t o1 = __riscv_vadd_vx_u32m8(base3, 1, vl);
      vuint8m2_t g = __riscv_vloxei32_v_u8m2((const void *)palette, o1, vl);
      vuint32m8_t o2 = __riscv_vadd_vx_u32m8(base3, 2, vl);
      vuint8m2_t b = __riscv_vloxei32_v_u8m2((const void *)palette, o2, vl);

      __riscv_vsse8_v_u8m2((void *)(dp),     3, r, vl);
      __riscv_vsse8_v_u8m2((void *)(dp + 1), 3, g, vl);
      __riscv_vsse8_v_u8m2((void *)(dp + 2), 3, b, vl);

      *ssp = sp - 1;
      *ddp = dp - 1;
      i += (png_uint_32)vl;
   }

   return (int)i;
}


/* Full replacement for the RGBA (num_trans > 0) branch of
 * png_do_expand_palette: RVV gather chunks + the original scalar tail
 * (copied verbatim for bit-exactness) + the row_info update.  Returns 1 if
 * the row was handled, 0 to let the caller fall back to the C path. */
int
png_do_expand_palette_rgba8_rvv_full(png_structrp png_ptr,
    png_row_infop row_info, png_bytep row)
{
   png_const_colorp palette = png_ptr->palette;
   png_const_bytep trans_alpha = png_ptr->trans_alpha;
   int num_trans = png_ptr->num_trans;
   png_uint_32 row_width = row_info->width;
   png_bytep sp, dp;
   png_uint_32 i;

   if (row_info->color_type != PNG_COLOR_TYPE_PALETTE ||
       row_info->bit_depth != 8 ||
       num_trans <= 0)
      return 0; /* not our case: caller falls back to png_do_expand_palette */

   /* Lazy riffle-table init (was previously hooked in pngrtran.c; keeping
    * ALL RVV-specific code in this file minimizes compiler restructuring of
    * the shared hot function png_do_read_transformations). */
   if (png_ptr->riffled_palette == NULL)
   {
      png_ptr->riffled_palette = (png_bytep)png_malloc(png_ptr, 256 * 4);
      png_riffle_palette_rvv(png_ptr);
   }

   /* Short-row guard: the gather kernel consumes 64-pixel chunks; below
    * that the whole row would run the scalar tail through this extra call.
    * Fall back to png_do_expand_palette entirely (X100 call overhead). */
   if (row_width < 64)
      return 0;

   sp = row + (size_t)row_width - 1;
   dp = row + ((size_t)row_width << 2) - 1;

   i = png_do_expand_palette_rgba8_rvv(png_ptr, row_info, row, &sp, &dp);

   for (; i < row_width; i++)
   {
      if ((int)(*sp) >= num_trans)
         *dp-- = 0xff;
      else
         *dp-- = trans_alpha[*sp];
      *dp-- = palette[*sp].blue;
      *dp-- = palette[*sp].green;
      *dp-- = palette[*sp].red;
      sp--;
   }

   row_info->bit_depth = 8;
   row_info->pixel_depth = 32;
   row_info->rowbytes = (size_t)row_width * 4;
   row_info->color_type = 6;
   row_info->channels = 4;
   return 1;
}

#endif /* PNG_RISCV_RVV_IMPLEMENTATION == 1 */
#endif /* PNG_READ_SUPPORTED */
