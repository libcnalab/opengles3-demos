// es30-pixel-buffer-objects
//
// Pixel buffer objects (OpenGL ES 3.0 sections 3.7.1, 3.7.2.1, 3.8.3, 4.3.2): a buffer object bound to
// PIXEL_UNPACK_BUFFER is the SOURCE of glTexImage2D / glTexSubImage2D, a buffer object bound to
// PIXEL_PACK_BUFFER is the DESTINATION of glReadPixels, and in both cases the "pointer" argument is
// a BYTE OFFSET into the buffer. The pixel storage modes (ROW_LENGTH, SKIP_ROWS, SKIP_PIXELS, ALIGNMENT) work
// on buffer data exactly as on client memory, so a sub-rectangle of a larger image can be selected inside the
// buffer, and the data can be read back with glMapBufferRange.
//
// The 256 x 256 default framebuffer shows seven textures, each uploaded from a buffer (cells of 32 x 32
// pixels, row 0 at the bottom; small textures are magnified so that every texel is several pixels):
//
//   cell 0  the 32 x 32 chart, glTexImage2D with data = offset 0 (a NULL pointer with a buffer bound is NOT "no data")
//   cell 1  a 20 x 13 sub-rectangle of the chart: UNPACK_ROW_LENGTH 32, SKIP_PIXELS 5, SKIP_ROWS 3
//   cell 2  a 20 x 13 sub-rectangle of a second, inverted chart that starts at byte offset 4160 of the same buffer
//   cell 3  glTexSubImage2D of a 14 x 10 patch (taken from the chart with ROW_LENGTH/SKIP) into a flat texture
//   cell 4  RGB8 5 x 4 (x6) from a 7 x 6 image with UNPACK_ROW_LENGTH 7 and the default alignment 4 (rows of 24 bytes), data offset 5
//   cell 5  RGBA4 5 x 4 (x6), type UNSIGNED_SHORT_4_4_4_4 (rows of 12 bytes), data offset 6
//   cell 6  a GPU round trip: glReadPixels of cell 0 into a PIXEL_PACK_BUFFER, then the SAME buffer as PIXEL_UNPACK_BUFFER of glTexImage2D
//
// Validation: textures are compared with CPU models of the memory layout (eq. 3.12 / 3.13), exactly where the texture can be
// attached to a framebuffer and read back, by sampling (+-1) otherwise; glReadPixels into buffers is read back with
// glMapBufferRange and compared with a client-memory read of the same pixels and a CPU placement model; the asynchronous
// pattern (read into one of two buffers per frame, fence, map the PREVIOUS frame's buffer) is run for six frames; the defined
// errors and the state queries are checked.

#include <demo/gl_util.hpp>
#include <demo/runtime.hpp>

#include <metagl/metagl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace mgl = metagl;
using demo::Rgba8;

namespace
{
    constexpr int kChart = 32;      // the chart is 32 x 32 RGBA8
    constexpr int kCellPx = 32;     // edge of a cell of the sheet
    constexpr int kCells = 8;
    constexpr std::size_t kChartBytes = static_cast<std::size_t>(kChart) * kChart * 4;
    constexpr std::size_t kHeader = 64;                                   // bytes between the two charts in the buffer
    constexpr std::size_t kSecondChart = kChartBytes + kHeader;           // offset of the second (inverted) chart: 4160
    constexpr std::size_t kImagesSize = kSecondChart + kChartBytes;       // 8256
    constexpr std::uint8_t kFill = 0xEE;                                  // everything in a buffer that is not image data
    constexpr auto kColor0 = mgl::to_framebuffer_attachment(mgl::ColorAttachment::Color0);

    constexpr std::uint8_t u8(int v) { return static_cast<std::uint8_t>(v); }

    // The chart: every texel (x, y) is different, so a displaced, flipped or channel-swapped upload cannot pass.
    constexpr Rgba8 chart_texel(int x, int y)
    {
        const bool odd = (((x >> 2) ^ (y >> 2)) & 1) != 0;
        return {u8(8 * x), u8(8 * y), u8(odd ? 200 : 60), u8(128 + x + y)};
    }

    // The second chart, in the same buffer at another offset: a different image, so ignoring the offset is visible.
    constexpr Rgba8 chart2_texel(int x, int y)
    {
        const Rgba8 c = chart_texel(x, y);
        return {u8(255 - c[0]), c[1], u8(255 - c[2]), c[3]};
    }

    constexpr Rgba8 kBase = {64, 96, 160, 255};  // the flat texture that glTexSubImage2D patches

    // RGB8 test image (7 x 6) and the RGBA4 test image (5 x 4): small, every texel distinct.
    constexpr std::array<std::uint8_t, 3> rgb_texel(int x, int y) { return {u8(20 + 30 * x), u8(30 + 35 * y), u8(200 - 12 * x - 9 * y)}; }
    constexpr std::array<int, 4> nibbles(int x, int y) { return {(x * 3 + 1) % 16, (y * 5 + 2) % 16, (x + y * 4 + 3) % 16, (x * 2 + y * 3 + 5) % 16}; }
    constexpr int kRgbW = 7, kRgbH = 6, kRgbSubW = 5, kRgbSubH = 4;
    constexpr int k4444W = 5, k4444H = 4;

    // ---------------------------------------------------------------------------------------------
    // Memory layout of pixel groups (ES 3.0 section 3.7.2.1, equations 3.12 and 3.13; ReadPixels packs the same way).
    // ---------------------------------------------------------------------------------------------
    struct Layout
    {
        int n;  // elements per group
        int s;  // bytes per element
        [[nodiscard]] int group_bytes() const { return n * s; }
    };

    /// k of equation 3.13, in bytes: nl when s >= a, otherwise a/s * ceil(s*n*l / a) elements.
    std::size_t row_stride(const Layout& lay, int groups_per_row, int alignment)
    {
        const int nl = lay.n * groups_per_row;
        const int k = lay.s >= alignment ? nl : alignment / lay.s * ((lay.s * nl + alignment - 1) / alignment);
        return static_cast<std::size_t>(k) * static_cast<std::size_t>(lay.s);
    }

    using Group = std::function<std::array<std::uint8_t, 16>(int, int)>;  // bytes of the group (i, j); the first group_bytes() are used

    /// Writes the w x h groups of an image into `buf` as the GL would read or write them: group (i, j) at
    /// offset + (skip_rows + j) * k + (skip_pixels + i) * group_bytes. Returns the end of the last group. `mask`
    /// (optional) is set for every byte that belongs to a group.
    std::size_t place_groups(std::vector<std::uint8_t>& buf, std::size_t offset, const Layout& lay, int row_length, int skip_rows, int skip_pixels, int alignment, int w, int h,
                             const Group& group, std::vector<bool>* mask = nullptr)
    {
        const std::size_t k = row_stride(lay, row_length > 0 ? row_length : w, alignment);
        const std::size_t gb = static_cast<std::size_t>(lay.group_bytes());
        std::size_t end = offset;
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i)
            {
                const std::size_t pos = offset + static_cast<std::size_t>(skip_rows + j) * k + static_cast<std::size_t>(skip_pixels + i) * gb;
                const std::array<std::uint8_t, 16> g = group(i, j);
                for (std::size_t b = 0; b < gb; ++b)
                {
                    buf.at(pos + b) = g[b];
                    if (mask != nullptr) (*mask)[pos + b] = true;
                }
                end = std::max(end, pos + gb);
            }
        return end;
    }

    std::array<std::uint8_t, 16> rgba_group(Rgba8 c) { return {c[0], c[1], c[2], c[3]}; }

    // ---------------------------------------------------------------------------------------------
    // Small comparison helpers
    // ---------------------------------------------------------------------------------------------
    struct Diff
    {
        int bad = 0;
        int max_dev = 0;
        std::string first;
    };

    /// Compares two RGBA8 images (row-major, bottom-up); `tolerance` per channel.
    Diff compare_rgba(const std::vector<std::uint8_t>& got, const std::function<Rgba8(int, int)>& want, int w, int h, int tolerance)
    {
        Diff d;
        if (got.size() != static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4)
        {
            d.bad = 1 << 20;
            d.first = std::format("{} bytes read, {} expected", got.size(), w * h * 4);
            return d;
        }
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const Rgba8 e = want(x, y);
                int dev = 0;
                for (std::size_t k = 0; k < 4; ++k) dev = std::max(dev, std::abs(int{got[static_cast<std::size_t>((y * w + x) * 4) + k]} - int{e[k]}));
                d.max_dev = std::max(d.max_dev, dev);
                if (dev > tolerance && d.bad++ == 0)
                {
                    const std::size_t o = static_cast<std::size_t>((y * w + x) * 4);
                    d.first = std::format("first at ({},{}): got ({},{},{},{}) want {}", x, y, got[o], got[o + 1], got[o + 2], got[o + 3], demo::format_rgba(e));
                }
            }
        return d;
    }

    /// Sentinel-aware comparison of a packed buffer with the model. Bytes of placed groups must be equal; bytes outside
    /// [offset, end) must still hold the fill value; bytes inside the footprint that no group covers (row padding, skipped
    /// pixels) are not specified and are only counted.
    struct PackDiff
    {
        int group_bad = 0;
        int outside_bad = 0;
        int padding_changed = 0;
        std::string first;
    };

    PackDiff compare_pack(const std::vector<std::uint8_t>& got, const std::vector<std::uint8_t>& model, const std::vector<bool>& mask, std::size_t offset, std::size_t end)
    {
        PackDiff d;
        if (got.size() != model.size())
        {
            d.group_bad = 1 << 20;
            d.first = std::format("{} bytes mapped, {} expected", got.size(), model.size());
            return d;
        }
        for (std::size_t b = 0; b < got.size(); ++b)
        {
            if (mask[b])
            {
                if (got[b] != model[b] && d.group_bad++ == 0) d.first = std::format("first group byte {} is {:#04x}, expected {:#04x}", b, got[b], model[b]);
            }
            else if (b < offset || b >= end)
            {
                if (got[b] != kFill && d.outside_bad++ == 0) d.first = std::format("byte {} outside the footprint [{}, {}) was changed to {:#04x}", b, offset, end, got[b]);
            }
            else if (got[b] != kFill)
            {
                ++d.padding_changed;
            }
        }
        return d;
    }

    template <typename F>
    void expect_error(demo::Check& check, mgl::ErrorCode code, std::string_view what, F&& call)
    {
        (void)demo::drain_gl_errors();  // start from an empty queue so the error belongs to this call
        call();
        check.expect_gl_error(code, what);
    }

    // glReadPixels takes a void* that is a byte offset while a pixel pack buffer is bound.
    void* pack_offset(std::size_t bytes) { return const_cast<void*>(demo::buffer_offset(bytes)); }

    struct ShowCell
    {
        std::string what;
        mgl::TextureId tex{};
        int index = 0;
        int w = 0, h = 0, scale = 1;
        std::function<Rgba8(int, int)> texel;  // the expected texel (i, j)
        bool exact = false;                     // RGBA8 texture: read it back exactly through a framebuffer
    };

    class PixelBufferObjects final : public demo::Demo
    {
    public:
        void init(demo::Env& env) override
        {
            if (env.width < kCells * kCellPx || env.height < kCellPx)
                throw demo::Skip(std::format("needs a default framebuffer of at least {}x{}", kCells * kCellPx, kCellPx));

            show_ = demo::build_program(env, "shaders/quad.vert", "shaders/show.frag");
            show_tex_ = mgl::glGetUniformLocation(show_, "u_tex");

            mgl::glGenBuffers(static_cast<GLsizei>(buf_.size()), buf_.data());
            mgl::glGenTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glGenFramebuffers(1, &attach_fbo_);

            upload_images();
            upload_rgb();
            upload_4444();

            // The buffer of the GPU round trip (cell 6) only needs a data store of the right size: glReadPixels fills it.
            // Without a store its size is 0 and any read into it is "beyond the end of the buffer" (INVALID_OPERATION).
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, buf_[kBufRoundTrip]);
            mgl::glBufferData(mgl::BufferTarget::PixelPack, static_cast<GLsizeiptr>(kChartBytes), nullptr, mgl::BufferUsage::StreamCopy);
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
            // The texture that render() re-specifies every frame: its sampling state must make a one-level texture complete
            // (the default MIN_FILTER NEAREST_MIPMAP_LINEAR would not).
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kTexRoundTrip]);
            nearest_clamp();
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            build_cells();
        }

        void render(demo::Env& /*env*/, int /*frame*/) override
        {
            mgl::glDisable(mgl::Capability::Dither);  // dithering could change a stored 8-bit value by one step
            mgl::glViewport(0, 0, kCells * kCellPx, kCellPx);
            mgl::glClearColor(0.08f, 0.08f, 0.1f, 1.0f);
            mgl::glClear(mgl::ClearBufferBit::Color);
            mgl::glUseProgram(show_);
            mgl::glUniform1i(show_tex_, 0);

            // Cells 0 .. 5 show the textures that init() uploaded from pixel unpack buffers.
            for (std::size_t c = 0; c < cells_.size() - 1; ++c) draw_cell(cells_[c]);

            // Cell 6: a GPU-side round trip. The pixels of cell 0 are read into a pixel PACK buffer ...
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, buf_[kBufRoundTrip]);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, 4);
            mgl::glReadPixels(0, 0, kChart, kChart, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, pack_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
            // ... and the very same buffer object becomes the PIXEL_UNPACK_BUFFER of the texture upload: no pixel crosses to the CPU.
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, buf_[kBufRoundTrip]);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kTexRoundTrip]);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, kChart, kChart, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, demo::buffer_offset(0));
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, mgl::BufferId{0});
            draw_cell(cells_.back());

            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glViewport(0, 0, kCells * kCellPx, kCellPx);
        }

        void validate(demo::Env& /*env*/, demo::Check& check) override
        {
            cell0_ = demo::read_rgba8(0, 0, kChart, kChart);  // the pixels of the chart cell as they are in the framebuffer now
            check_pixel_store_state(check);
            check_buffer_bindings(check);
            check_cells(check);
            check_unpack_volume(check);
            check_pack(check);
            check_async_pattern(check);
            check_unpack_errors(check);
            check_pack_errors(check);
            check_mapping_rules(check);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            check.expect_no_gl_error("no stray GL error left by the validation sequence");
        }

        void shutdown(demo::Env& /*env*/) override
        {
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, mgl::BufferId{0});
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glUseProgram(mgl::ProgramId{0});
            mgl::glDeleteFramebuffers(1, &attach_fbo_);
            mgl::glDeleteTextures(static_cast<GLsizei>(tex_.size()), tex_.data());
            mgl::glDeleteBuffers(static_cast<GLsizei>(buf_.size()), buf_.data());
            mgl::glDeleteProgram(show_);
        }

    private:
        enum : std::size_t { kBufImages, kBufRgb, kBuf4444, kBufRoundTrip, kBufCount };
        enum : std::size_t { kTexChart, kTexSub1, kTexSub2, kTexPatch, kTexRgb, kTex4444, kTexRoundTrip, kTexCount };

        // ------------------------------------------------------------------------------------------
        // Uploads from pixel unpack buffers
        // ------------------------------------------------------------------------------------------
        static void nearest_clamp()
        {
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMinFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureMagFilter::Nearest);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapS, mgl::TextureWrapMode::ClampToEdge);
            mgl::glTexParameter(mgl::TextureTarget::Texture2D, mgl::TextureWrapParameter::WrapT, mgl::TextureWrapMode::ClampToEdge);
        }

        static void set_unpack(int row_length, int skip_pixels, int skip_rows)
        {
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackRowLength, row_length);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipPixels, skip_pixels);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipRows, skip_rows);
        }

        static void reset_unpack()
        {
            set_unpack(0, 0, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 4);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackImageHeight, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipImages, 0);
        }

        void upload_images()
        {
            // One buffer holds two charts: the chart at offset 0, 64 bytes of fill, the inverted chart at offset 4160.
            std::vector<std::uint8_t> images(kImagesSize, kFill);
            for (int y = 0; y < kChart; ++y)
                for (int x = 0; x < kChart; ++x)
                {
                    const Rgba8 a = chart_texel(x, y), b = chart2_texel(x, y);
                    std::copy(a.begin(), a.end(), images.begin() + static_cast<std::ptrdiff_t>((y * kChart + x) * 4));
                    std::copy(b.begin(), b.end(), images.begin() + static_cast<std::ptrdiff_t>(kSecondChart + static_cast<std::size_t>((y * kChart + x) * 4)));
                }
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, buf_[kBufImages]);
            mgl::glBufferData(mgl::BufferTarget::PixelUnpack, static_cast<GLsizeiptr>(images.size()), images.data(), mgl::BufferUsage::StaticDraw);

            const mgl::TextureTarget t2d = mgl::TextureTarget::Texture2D;
            using IF = mgl::InternalFormat;
            using PF = mgl::PixelFormat;
            const mgl::PixelType ub = mgl::PixelType::UnsignedByte;

            // Cell 0: the whole chart. The last argument is "0": with a buffer bound that is the byte offset 0, not "no data".
            mgl::glBindTexture(t2d, tex_[kTexChart]);
            mgl::glTexImage2D(t2d, 0, IF::Rgba8, kChart, kChart, 0, PF::Rgba, ub, demo::buffer_offset(0));
            nearest_clamp();

            // Cell 1: a 20 x 13 sub-rectangle at column 5, row 3 of the 32-texel-wide chart.
            set_unpack(kChart, 5, 3);
            mgl::glBindTexture(t2d, tex_[kTexSub1]);
            mgl::glTexImage2D(t2d, 0, IF::Rgba8, 20, 13, 0, PF::Rgba, ub, demo::buffer_offset(0));
            nearest_clamp();

            // Cell 2: the same kind of selection in the second chart, which starts at byte offset 4160.
            set_unpack(kChart, 2, 8);
            mgl::glBindTexture(t2d, tex_[kTexSub2]);
            mgl::glTexImage2D(t2d, 0, IF::Rgba8, 20, 13, 0, PF::Rgba, ub, demo::buffer_offset(kSecondChart));
            nearest_clamp();
            reset_unpack();

            // Cell 3: a flat texture from client memory (no unpack buffer), then a patch from the buffer with glTexSubImage2D.
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, mgl::BufferId{0});
            const std::vector<Rgba8> flat(static_cast<std::size_t>(kChart) * kChart, kBase);
            mgl::glBindTexture(t2d, tex_[kTexPatch]);
            mgl::glTexImage2D(t2d, 0, IF::Rgba8, kChart, kChart, 0, PF::Rgba, ub, flat.data());
            nearest_clamp();
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, buf_[kBufImages]);
            set_unpack(kChart, 11, 2);
            mgl::glTexSubImage2D(t2d, 0, 6, 9, 14, 10, PF::Rgba, ub, demo::buffer_offset(0));
            reset_unpack();
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, mgl::BufferId{0});
        }

        void upload_rgb()
        {
            // A 7 x 6 RGB image with 3-byte groups. With the default UNPACK_ALIGNMENT 4 a row of 7 groups (21 bytes) is
            // padded to 24 bytes (equation 3.13). The image starts 5 bytes into the buffer. Rows 1..4 and columns 1..5 are uploaded.
            std::vector<std::uint8_t> buf(5 + 6 * 24, kFill);
            place_groups(buf, 5, {3, 1}, kRgbW, 0, 0, 4, kRgbW, kRgbH, [](int i, int j) {
                const auto c = rgb_texel(i, j);
                return std::array<std::uint8_t, 16>{c[0], c[1], c[2]};
            });
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, buf_[kBufRgb]);
            mgl::glBufferData(mgl::BufferTarget::PixelUnpack, static_cast<GLsizeiptr>(buf.size()), buf.data(), mgl::BufferUsage::StaticDraw);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kTexRgb]);
            set_unpack(kRgbW, 1, 1);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgb8, kRgbSubW, kRgbSubH, 0, mgl::PixelFormat::Rgb, mgl::PixelType::UnsignedByte, demo::buffer_offset(5));
            reset_unpack();
            nearest_clamp();
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, mgl::BufferId{0});
        }

        void upload_4444()
        {
            // A 5 x 4 RGBA image in UNSIGNED_SHORT_4_4_4_4: one 16-bit element per group, so rows of 10 bytes are padded to 12
            // for alignment 4. The data starts 6 bytes into the buffer (a multiple of the element size 2, as required).
            std::vector<std::uint8_t> buf(6 + 4 * 12, kFill);
            place_groups(buf, 6, {1, 2}, 0, 0, 0, 4, k4444W, k4444H, [](int i, int j) {
                const std::array<int, 4> n = nibbles(i, j);
                const std::uint16_t v = static_cast<std::uint16_t>((n[0] << 12) | (n[1] << 8) | (n[2] << 4) | n[3]);
                std::array<std::uint8_t, 16> g{};
                std::memcpy(g.data(), &v, 2);
                return g;
            });
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, buf_[kBuf4444]);
            mgl::glBufferData(mgl::BufferTarget::PixelUnpack, static_cast<GLsizeiptr>(buf.size()), buf.data(), mgl::BufferUsage::StaticDraw);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, tex_[kTex4444]);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba4, k4444W, k4444H, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedShort4444, demo::buffer_offset(6));
            nearest_clamp();
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, mgl::BufferId{0});
        }

        // The expected content of every cell (the CPU model of what the unpack should have produced).
        void build_cells()
        {
            cells_.push_back({"cell 0: glTexImage2D from a PIXEL_UNPACK_BUFFER at offset 0 (32x32 chart)", tex_[kTexChart], 0, kChart, kChart, 1,
                              [](int i, int j) { return chart_texel(i, j); }, true});
            cells_.push_back({"cell 1: ROW_LENGTH 32, SKIP_PIXELS 5, SKIP_ROWS 3 select a 20x13 sub-rectangle of the chart in the buffer", tex_[kTexSub1], 1, 20, 13, 1,
                              [](int i, int j) { return chart_texel(5 + i, 3 + j); }, true});
            cells_.push_back({"cell 2: the same selection (SKIP 2, 8) in the inverted chart at byte offset 4160 of the buffer", tex_[kTexSub2], 2, 20, 13, 1,
                              [](int i, int j) { return chart2_texel(2 + i, 8 + j); }, true});
            cells_.push_back({"cell 3: glTexSubImage2D 14x10 patch (SKIP 11, 2) at texel (6, 9) of a flat texture; the rest keeps its color", tex_[kTexPatch], 3, kChart, kChart, 1,
                              [](int i, int j) {
                                  const bool inside = i >= 6 && i < 6 + 14 && j >= 9 && j < 9 + 10;
                                  return inside ? chart_texel(11 + (i - 6), 2 + (j - 9)) : kBase;
                              },
                              true});
            cells_.push_back({"cell 4: RGB8 5x4 from a 7x6 image (ROW_LENGTH 7, SKIP 1, 1, alignment 4: 24-byte rows), data offset 5", tex_[kTexRgb], 4, kRgbSubW, kRgbSubH, 6,
                              [](int i, int j) {
                                  const auto c = rgb_texel(1 + i, 1 + j);
                                  return Rgba8{c[0], c[1], c[2], 255};
                              },
                              false});
            cells_.push_back({"cell 5: RGBA4 5x4, UNSIGNED_SHORT_4_4_4_4 from data offset 6 (12-byte rows)", tex_[kTex4444], 5, k4444W, k4444H, 6,
                              [](int i, int j) {
                                  const std::array<int, 4> n = nibbles(i, j);
                                  return Rgba8{u8(n[0] * 17), u8(n[1] * 17), u8(n[2] * 17), u8(n[3] * 17)};  // c / 15 * 255 = 17 c
                              },
                              false});
            cells_.push_back({"cell 6: glReadPixels into a PIXEL_PACK_BUFFER, then the same buffer as PIXEL_UNPACK_BUFFER of glTexImage2D (cell 0 again)", tex_[kTexRoundTrip], 6,
                              kChart, kChart, 1, [this](int i, int j) { return cell0_texel(i, j); }, true});
        }

        [[nodiscard]] Rgba8 cell0_texel(int i, int j) const
        {
            const std::size_t o = static_cast<std::size_t>((j * kChart + i) * 4);
            return {cell0_[o], cell0_[o + 1], cell0_[o + 2], cell0_[o + 3]};
        }

        void draw_cell(const ShowCell& c) const
        {
            mgl::glViewport(c.index * kCellPx, 0, c.w * c.scale, c.h * c.scale);
            mgl::glActiveTexture(mgl::TextureUnit::Texture0);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, c.tex);
            mgl::glDrawArrays(mgl::PrimitiveType::TriangleStrip, 0, 4);
        }

        // ------------------------------------------------------------------------------------------
        // Helpers for the checks
        // ------------------------------------------------------------------------------------------
        /// Reads level 0 of an RGBA8 texture exactly, through a framebuffer it is attached to.
        std::vector<std::uint8_t> read_attached(mgl::TextureId tex, int w, int h)
        {
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, attach_fbo_);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, tex, 0);
            std::vector<std::uint8_t> out;
            if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) == mgl::FramebufferStatus::Complete) out = demo::read_rgba8(0, 0, w, h);
            mgl::glFramebufferTexture2D(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureTarget::Texture2D, mgl::TextureId{0}, 0);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            return out;
        }

        /// Maps the whole buffer bound to `target` for reading, copies it, unmaps. Empty on failure.
        static std::vector<std::uint8_t> map_and_copy(mgl::BufferTarget target, std::size_t size, bool* unmap_ok = nullptr)
        {
            void* p = mgl::glMapBufferRange(target, 0, static_cast<GLsizeiptr>(size), mgl::MapBufferAccessMask::Read);
            if (p == nullptr) return {};
            std::vector<std::uint8_t> out(size);
            std::memcpy(out.data(), p, size);
            const bool ok = mgl::glUnmapBuffer(target) != GL_FALSE;
            if (unmap_ok != nullptr) *unmap_ok = ok;
            return out;
        }

        /// A buffer object bound to `target` with `size` bytes of `fill`.
        static void make_buffer(mgl::BufferTarget target, mgl::BufferId id, std::size_t size, std::uint8_t fill, mgl::BufferUsage usage)
        {
            const std::vector<std::uint8_t> bytes(size, fill);
            mgl::glBindBuffer(target, id);
            mgl::glBufferData(target, static_cast<GLsizeiptr>(size), bytes.data(), usage);
        }

        // ------------------------------------------------------------------------------------------
        // State: pixel storage modes and buffer bindings
        // ------------------------------------------------------------------------------------------
        void check_pixel_store_state(demo::Check& check) const
        {
            using GP = mgl::GetParameter;
            // Initial values (ES 3.0 Tables 3.1 and 4.4). Nothing has changed them since the demo reset them to 0 / 4 after each use.
            check.expect(demo::get_integer(GP::UnpackRowLength) == 0 && demo::get_integer(GP::UnpackSkipRows) == 0 && demo::get_integer(GP::UnpackSkipPixels) == 0 &&
                             demo::get_integer(GP::UnpackAlignment) == 4 && demo::get_integer(GP::UnpackImageHeight) == 0 && demo::get_integer(GP::UnpackSkipImages) == 0,
                         "UNPACK_ROW_LENGTH / SKIP_ROWS / SKIP_PIXELS / IMAGE_HEIGHT / SKIP_IMAGES are 0 and UNPACK_ALIGNMENT is 4 (Table 3.1; as left by the demo's own resets)");
            check.expect(demo::get_integer(GP::PackRowLength) == 0 && demo::get_integer(GP::PackSkipRows) == 0 && demo::get_integer(GP::PackSkipPixels) == 0 &&
                             demo::get_integer(GP::PackAlignment) == 4,
                         "PACK_ROW_LENGTH / SKIP_ROWS / SKIP_PIXELS are 0 and PACK_ALIGNMENT is 4 (Table 4.4)");

            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackRowLength, 37);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipRows, 5);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipPixels, 7);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 8);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackImageHeight, 9);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipImages, 2);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackRowLength, 41);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipRows, 3);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipPixels, 6);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, 2);
            check.expect(demo::get_integer(GP::UnpackRowLength) == 37 && demo::get_integer(GP::UnpackSkipRows) == 5 && demo::get_integer(GP::UnpackSkipPixels) == 7 &&
                             demo::get_integer(GP::UnpackAlignment) == 8 && demo::get_integer(GP::UnpackImageHeight) == 9 && demo::get_integer(GP::UnpackSkipImages) == 2,
                         "glPixelStorei UNPACK_* values (37, 5, 7, 8, 9, 2) are read back with glGetIntegerv");
            check.expect(demo::get_integer(GP::PackRowLength) == 41 && demo::get_integer(GP::PackSkipRows) == 3 && demo::get_integer(GP::PackSkipPixels) == 6 &&
                             demo::get_integer(GP::PackAlignment) == 2,
                         "glPixelStorei PACK_* values (41, 3, 6, 2) are read back with glGetIntegerv");

            // 3.7.1: a value outside the allowed range is INVALID_VALUE (and leaves the state as it was).
            expect_error(check, mgl::ErrorCode::InvalidValue, "UNPACK_ROW_LENGTH = -1 -> INVALID_VALUE (3.7.1)", [] { mgl::glPixelStorei(mgl::PixelStoreParam::UnpackRowLength, -1); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "PACK_SKIP_PIXELS = -1 -> INVALID_VALUE", [] { mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipPixels, -1); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "UNPACK_ALIGNMENT = 3 (not 1, 2, 4, 8) -> INVALID_VALUE", [] { mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 3); });
            expect_error(check, mgl::ErrorCode::InvalidValue, "PACK_ALIGNMENT = 16 -> INVALID_VALUE", [] { mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, 16); });
            check.expect(demo::get_integer(GP::UnpackRowLength) == 37 && demo::get_integer(GP::PackSkipPixels) == 6 && demo::get_integer(GP::UnpackAlignment) == 8 &&
                             demo::get_integer(GP::PackAlignment) == 2,
                         "the rejected glPixelStorei calls changed nothing");

            // Restore the defaults (the later checks rely on them).
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackRowLength, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipRows, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipPixels, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackAlignment, 4);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackImageHeight, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipImages, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackRowLength, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipRows, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipPixels, 0);
            mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, 4);
        }

        void check_buffer_bindings(demo::Check& check) const
        {
            using GP = mgl::GetParameter;
            check.expect(demo::get_integer(GP::PixelPackBufferBinding) == 0 && demo::get_integer(GP::PixelUnpackBufferBinding) == 0,
                         "PIXEL_PACK_BUFFER_BINDING and PIXEL_UNPACK_BUFFER_BINDING are 0 initially / after the uploads (Table 6.16)");

            mgl::BufferId a{}, b{};
            mgl::glGenBuffers(1, &a);
            mgl::glGenBuffers(1, &b);
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, a);
            check.expect(demo::get_integer(GP::PixelPackBufferBinding) == static_cast<GLint>(a.value) && demo::get_integer(GP::PixelUnpackBufferBinding) == 0 &&
                             demo::get_integer(GP::ArrayBufferBinding) == 0,
                         "binding a buffer to PIXEL_PACK_BUFFER changes only PIXEL_PACK_BUFFER_BINDING", std::format("pack = {}, buffer name = {}", demo::get_integer(GP::PixelPackBufferBinding), a.value));
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, b);
            check.expect(demo::get_integer(GP::PixelPackBufferBinding) == static_cast<GLint>(a.value) && demo::get_integer(GP::PixelUnpackBufferBinding) == static_cast<GLint>(b.value),
                         "PIXEL_PACK and PIXEL_UNPACK bindings are independent: two different buffers at once");
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, a);
            check.expect(demo::get_integer(GP::PixelPackBufferBinding) == static_cast<GLint>(a.value) && demo::get_integer(GP::PixelUnpackBufferBinding) == static_cast<GLint>(a.value),
                         "one buffer object can be bound to PIXEL_PACK_BUFFER and PIXEL_UNPACK_BUFFER at the same time (the round trip of cell 6)");
            mgl::glBufferData(mgl::BufferTarget::PixelPack, 128, nullptr, mgl::BufferUsage::StreamRead);
            GLint size = 0, usage = 0;
            mgl::glGetBufferParameteriv(mgl::BufferTarget::PixelUnpack, mgl::BufferParameter::Size, &size);  // the same object through the other target
            mgl::glGetBufferParameteriv(mgl::BufferTarget::PixelPack, mgl::BufferParameter::Usage, &usage);
            check.expect(size == 128 && usage == static_cast<GLint>(GL_STREAM_READ), "a pixel buffer is an ordinary buffer object: BUFFER_SIZE 128 and BUFFER_USAGE STREAM_READ queried through either target",
                         std::format("size {}, usage {:#x}", size, usage));

            // 2.10.1: deleting a bound buffer resets the bindings to zero in the current context.
            mgl::glDeleteBuffers(1, &a);
            check.expect(demo::get_integer(GP::PixelPackBufferBinding) == 0 && demo::get_integer(GP::PixelUnpackBufferBinding) == 0,
                         "deleting a buffer that is bound to both targets resets both bindings to 0 (2.10.1)");
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, mgl::BufferId{0});
            mgl::glDeleteBuffers(1, &b);
            check.expect_no_gl_error("binding, querying and deleting pixel buffers raised no error");
        }

        // ------------------------------------------------------------------------------------------
        // Textures uploaded from buffers
        // ------------------------------------------------------------------------------------------
        void check_cells(demo::Check& check)
        {
            // The chart cell itself: CPU chart vs framebuffer (the shader's float -> 8-bit conversion allows +-1, 2.1.6.2).
            {
                const Diff d = compare_rgba(cell0_, [](int x, int y) { return chart_texel(x, y); }, kChart, kChart, 1);
                check.expect(d.bad == 0, std::format("cell 0 as drawn matches the CPU chart (+-1 for the shader's conversion; max deviation {})", d.max_dev), d.first);
            }
            for (const ShowCell& c : cells_)
            {
                // What the texture contains: exact where it can be attached to a framebuffer. For the round trip the reference is the framebuffer
                // content that was read into the buffer; for the others the CPU model.
                if (c.exact)
                {
                    const std::vector<std::uint8_t> got = read_attached(c.tex, c.w, c.h);
                    const Diff d = compare_rgba(got, c.texel, c.w, c.h, 0);
                    check.expect(d.bad == 0 && !got.empty(), std::format("{}: the texture is exactly the model (glReadPixels through a framebuffer, {}x{} texels)", c.what, c.w, c.h), d.first);
                }
                // What the picture shows: the model through a NEAREST lookup (+-1 for the conversions).
                const std::vector<std::uint8_t> px = demo::read_rgba8(c.index * kCellPx, 0, c.w * c.scale, c.h * c.scale);
                const int scale = c.scale;
                const auto texel = c.texel;
                const Diff d = compare_rgba(px, [scale, texel](int x, int y) { return texel(x / scale, y / scale); }, c.w * c.scale, c.h * c.scale, 1);
                check.expect(d.bad == 0, std::format("{}: the framebuffer shows the model ({}x{} pixels, +-1; max deviation {})", c.what, c.w * c.scale, c.h * c.scale, d.max_dev), d.first);
            }

            // Negative controls: the comparisons are able to fail.
            {
                // (1) cell 1 against a model that ignores ROW_LENGTH / SKIP: the top-left 20x13 texels of the chart.
                const ShowCell& c1 = cells_[1];
                const Diff naive = compare_rgba(read_attached(c1.tex, c1.w, c1.h), [](int x, int y) { return chart_texel(x, y); }, c1.w, c1.h, 0);
                check.expect(naive.bad > c1.w * c1.h * 9 / 10, "negative control: the sub-rectangle texture is not the top-left corner of the chart (ROW_LENGTH and SKIP had an effect)",
                             std::format("{} of {} texels differ", naive.bad, c1.w * c1.h));
                // (2) cell 2 against the first chart: the buffer offset had an effect.
                const ShowCell& c2 = cells_[2];
                const Diff other = compare_rgba(read_attached(c2.tex, c2.w, c2.h), [](int x, int y) { return chart_texel(2 + x, 8 + y); }, c2.w, c2.h, 0);
                check.expect(other.bad > c2.w * c2.h * 9 / 10, "negative control: cell 2 is not the same rectangle of the FIRST chart (the byte offset 4160 had an effect)",
                             std::format("{} of {} texels differ", other.bad, c2.w * c2.h));
            }
        }

        // UNPACK_IMAGE_HEIGHT and UNPACK_SKIP_IMAGES only exist for 3D uploads (3.8.3): glTexImage3D walks "images" of IMAGE_HEIGHT rows
        // and skips SKIP_IMAGES of them; glTexImage2D is "TexImage3D with a depth of 1, except that UNPACK_SKIP_IMAGES is ignored".
        void check_unpack_volume(demo::Check& check)
        {
            constexpr int big_w = 6, big_h = 5, big_d = 3;  // the volume in the buffer
            constexpr int w = 4, h = 3, d = 2;              // the sub-volume that is uploaded: columns 1.., rows 1.., images 1..
            constexpr std::size_t offset = 8;
            constexpr std::size_t image_bytes = static_cast<std::size_t>(big_h) * big_w * 4;  // IMAGE_HEIGHT 5 rows of 24 bytes
            const auto big = [](int x, int y, int z) { return Rgba8{u8(15 * x + 10 * z), u8(20 * y + 3 * z + 5), u8(60 + 40 * z + x), u8(200 - 7 * y - 3 * x)}; };

            std::vector<std::uint8_t> bytes(offset + big_d * image_bytes, kFill);
            for (int z = 0; z < big_d; ++z)
                place_groups(bytes, offset + static_cast<std::size_t>(z) * image_bytes, {4, 1}, big_w, 0, 0, 4, big_w, big_h, [&big, z](int i, int j) { return rgba_group(big(i, j, z)); });

            mgl::BufferId pbo{};
            mgl::TextureId volume{}, flat{};
            mgl::glGenBuffers(1, &pbo);
            mgl::glGenTextures(1, &volume);
            mgl::glGenTextures(1, &flat);
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, pbo);
            mgl::glBufferData(mgl::BufferTarget::PixelUnpack, static_cast<GLsizeiptr>(bytes.size()), bytes.data(), mgl::BufferUsage::StaticDraw);
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, volume);
            set_unpack(big_w, 1, 1);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackImageHeight, big_h);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipImages, 1);
            mgl::glTexImage3D(mgl::TextureTarget::Texture3D, 0, mgl::InternalFormat::Rgba8, w, h, d, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, demo::buffer_offset(offset));
            check.expect_no_gl_error("glTexImage3D from an unpack buffer with ROW_LENGTH 6, IMAGE_HEIGHT 5, SKIP_PIXELS 1, SKIP_ROWS 1, SKIP_IMAGES 1 raised no error");
            reset_unpack();

            int bad_layers = 0;
            std::string first;
            for (int k = 0; k < d; ++k)
            {
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, attach_fbo_);
                mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, volume, 0, k);
                std::vector<std::uint8_t> got;
                if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) == mgl::FramebufferStatus::Complete) got = demo::read_rgba8(0, 0, w, h);
                mgl::glFramebufferTextureLayer(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::TextureId{0}, 0, 0);
                mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
                const Diff dd = compare_rgba(got, [&big, k](int x, int y) { return big(1 + x, 1 + y, 1 + k); }, w, h, 0);
                if (dd.bad != 0 && bad_layers++ == 0) first = std::format("layer {}: {}", k, dd.first);
            }
            check.expect(bad_layers == 0, "the 4x3x2 texture is exactly the sub-volume (1, 1, 1) of the 6x5x3 volume in the buffer: every texel of both layers (ROW_LENGTH, IMAGE_HEIGHT, SKIP_*)", first);

            // glTexImage2D ignores UNPACK_SKIP_IMAGES (and has only one image, so IMAGE_HEIGHT cannot matter either).
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, buf_[kBufImages]);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, flat);
            nearest_clamp();
            set_unpack(kChart, 5, 3);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackImageHeight, 7);
            mgl::glPixelStorei(mgl::PixelStoreParam::UnpackSkipImages, 2);
            mgl::glTexImage2D(mgl::TextureTarget::Texture2D, 0, mgl::InternalFormat::Rgba8, 20, 13, 0, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, demo::buffer_offset(0));
            reset_unpack();
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, mgl::BufferId{0});
            const Diff d2 = compare_rgba(read_attached(flat, 20, 13), [](int x, int y) { return chart_texel(5 + x, 3 + y); }, 20, 13, 0);
            check.expect(d2.bad == 0, "glTexImage2D with UNPACK_SKIP_IMAGES 2 and UNPACK_IMAGE_HEIGHT 7 uploads the same 20x13 rectangle as without them (SKIP_IMAGES is ignored for 2D, 3.8.3)", d2.first);

            mgl::glBindTexture(mgl::TextureTarget::Texture2D, mgl::TextureId{0});
            mgl::glBindTexture(mgl::TextureTarget::Texture3D, mgl::TextureId{0});
            mgl::glDeleteTextures(1, &flat);
            mgl::glDeleteTextures(1, &volume);
            mgl::glDeleteBuffers(1, &pbo);
        }

        // ------------------------------------------------------------------------------------------
        // glReadPixels into pixel pack buffers
        // ------------------------------------------------------------------------------------------
        struct PackCase
        {
            std::string what;
            int x, y, w, h;
            int row_length, skip_rows, skip_pixels, alignment;
            std::size_t offset;
            std::size_t buffer_size;
        };

        void check_pack(demo::Check& check)
        {
            const std::vector<PackCase> cases = {
                {"the whole chart cell into a buffer of exactly 4096 bytes at offset 0", 0, 0, 32, 32, 0, 0, 0, 4, 0, 4096},
                {"a 10x8 rectangle at byte offset 100 of a 4096-byte buffer", 4, 6, 10, 8, 0, 0, 0, 4, 100, 4096},
                {"PACK_ROW_LENGTH 24, SKIP_PIXELS 3, SKIP_ROWS 2 for a 10x7 rectangle at offset 36", 8, 4, 10, 7, 24, 2, 3, 4, 36, 1024},
                {"PACK_ALIGNMENT 8: a 5x3 rectangle (20-byte rows) with 24-byte rows in the buffer", 3, 5, 5, 3, 0, 0, 0, 8, 0, 128},
            };
            int number = 1;
            for (const PackCase& c : cases)
            {
                // Ground truth: the same read into CLIENT memory (tightly packed).
                mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, 4);
                const std::vector<std::uint8_t> client = demo::read_rgba8(c.x, c.y, c.w, c.h);

                mgl::BufferId pbo{};
                mgl::glGenBuffers(1, &pbo);
                make_buffer(mgl::BufferTarget::PixelPack, pbo, c.buffer_size, kFill, mgl::BufferUsage::StreamRead);
                mgl::glPixelStorei(mgl::PixelStoreParam::PackRowLength, c.row_length);
                mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipRows, c.skip_rows);
                mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipPixels, c.skip_pixels);
                mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, c.alignment);
                mgl::glReadPixels(c.x, c.y, c.w, c.h, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, pack_offset(c.offset));
                mgl::glPixelStorei(mgl::PixelStoreParam::PackRowLength, 0);
                mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipRows, 0);
                mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipPixels, 0);
                mgl::glPixelStorei(mgl::PixelStoreParam::PackAlignment, 4);

                // glMapBufferRange returns the pointer "once all pending operations on that buffer have completed" (2.10.3): no explicit wait needed.
                bool unmap_ok = false;
                const std::vector<std::uint8_t> got = map_and_copy(mgl::BufferTarget::PixelPack, c.buffer_size, &unmap_ok);

                // The model: the client rows placed where equations 3.12 / 3.13 (with the PACK_* modes) put them.
                std::vector<std::uint8_t> model(c.buffer_size, kFill);
                std::vector<bool> mask(c.buffer_size, false);
                const std::size_t end = place_groups(model, c.offset, {4, 1}, c.row_length, c.skip_rows, c.skip_pixels, c.alignment, c.w, c.h,
                                                     [&client, &c](int i, int j) {
                                                         const std::size_t o = static_cast<std::size_t>((j * c.w + i) * 4);
                                                         return rgba_group({client[o], client[o + 1], client[o + 2], client[o + 3]});
                                                     },
                                                     &mask);
                const PackDiff d = compare_pack(got, model, mask, c.offset, end);
                check.expect(!got.empty() && unmap_ok && d.group_bad == 0,
                             std::format("pack {}: {}: every pixel is where the model puts it ({} bytes mapped, footprint [{}, {}))", number, c.what, got.size(), c.offset, end), d.first);
                check.expect(d.outside_bad == 0, std::format("pack {}: nothing outside the footprint was written (bytes before the offset, after the last pixel group)", number), d.first);
                if (d.padding_changed != 0)
                    std::cout << std::format("   note: pack {}: {} padding bytes inside the footprint were changed (their value is not specified)\n", number, d.padding_changed);

                if (number == 3)
                {
                    // Negative control: the model WITHOUT PACK_ROW_LENGTH / SKIP (tightly packed at the offset) must not match this buffer.
                    std::vector<std::uint8_t> flat(c.buffer_size, kFill);
                    std::vector<bool> flat_mask(c.buffer_size, false);
                    const std::size_t flat_end = place_groups(flat, c.offset, {4, 1}, 0, 0, 0, 4, c.w, c.h,
                                                              [&client, &c](int i, int j) {
                                                                  const std::size_t o = static_cast<std::size_t>((j * c.w + i) * 4);
                                                                  return rgba_group({client[o], client[o + 1], client[o + 2], client[o + 3]});
                                                              },
                                                              &flat_mask);
                    const PackDiff naive = compare_pack(got, flat, flat_mask, c.offset, flat_end);
                    check.expect(naive.group_bad + naive.outside_bad > 100, "negative control: the PACK_ROW_LENGTH/SKIP result does not match a tightly packed model (the comparison can fail)",
                                 std::format("{} group bytes and {} outside bytes differ", naive.group_bad, naive.outside_bad));
                }
                mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
                mgl::glDeleteBuffers(1, &pbo);
                ++number;
            }

            check_pack_integer(check);
            check_map_pointer_state(check);
        }

        // A normalized RGBA8 framebuffer always allows RGBA/UNSIGNED_BYTE (1-byte elements); an unsigned-integer framebuffer
        // requires RGBA_INTEGER/UNSIGNED_INT, whose 4-byte elements make the offset alignment rule observable (4.3.2.4).
        void check_pack_integer(demo::Check& check) const
        {
            mgl::RenderbufferId rb{};
            mgl::FramebufferId fbo{};
            mgl::glGenRenderbuffers(1, &rb);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8UI, 4, 4);
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::RenderbufferTarget::Renderbuffer, rb);
            const bool complete = mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) == mgl::FramebufferStatus::Complete;
            check.expect(complete, "an RGBA8UI renderbuffer framebuffer is complete (color-renderable in ES 3.0)");
            if (complete)
            {
                const std::array<GLuint, 4> value = {7, 150, 255, 33};
                mgl::glClearBuffer(mgl::UnsignedIntegerClearBuffer::Color, 0, value.data());

                mgl::BufferId pbo{};
                mgl::glGenBuffers(1, &pbo);
                const std::size_t size = 16 + 4 * 4 * 16 + 16;  // 16 bytes in front, 16 pixels of 16 bytes, 16 bytes behind
                make_buffer(mgl::BufferTarget::PixelPack, pbo, size, kFill, mgl::BufferUsage::StreamRead);
                mgl::glReadPixels(0, 0, 4, 4, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, pack_offset(16));
                const std::vector<std::uint8_t> got = map_and_copy(mgl::BufferTarget::PixelPack, size);
                int bad = 0;
                std::string first;
                for (std::size_t px = 0; px < 16 && !got.empty(); ++px)
                    for (std::size_t c = 0; c < 4; ++c)
                    {
                        GLuint v = 0;
                        std::memcpy(&v, &got[16 + px * 16 + c * 4], 4);
                        if (v != value[c] && bad++ == 0) first = std::format("pixel {} component {} is {}, expected {}", px, c, v, value[c]);
                    }
                bool outside_ok = !got.empty();
                for (std::size_t b = 0; b < 16 && outside_ok; ++b) outside_ok = got[b] == kFill && got[size - 1 - b] == kFill;
                check.expect(!got.empty() && bad == 0, "glReadPixels(RGBA_INTEGER, UNSIGNED_INT) of an RGBA8UI framebuffer into a pack buffer at offset 16: all 16 pixels are exactly (7, 150, 255, 33)", first);
                check.expect(outside_ok, "the 16 bytes before and after the 16 pixel groups still hold the fill value");

                // A 4-byte type needs an offset that is a multiple of 4 (3.7.2.1, 4.3.2.4): 2 is rejected, 4 is fine.
                make_buffer(mgl::BufferTarget::PixelPack, pbo, size, kFill, mgl::BufferUsage::StreamRead);
                // ES 3.0 4.3.2.4: "If a pixel pack buffer object is bound and data is not evenly divisible by the number of
                // basic machine units needed to store in memory the corresponding GL data type from table 3.4 for the type
                // parameter, an INVALID_OPERATION error results." The ANGLE build in the matrix applies this rule to
                // unpack buffers (checked below) but not to glReadPixels into a pack buffer.
                {
                    (void)demo::drain_gl_errors();
                    mgl::glReadPixels(0, 0, 4, 4, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, pack_offset(2));
                    const mgl::ErrorCode e = demo::drain_gl_errors();
                    check.expect_or_known_deviation(e == mgl::ErrorCode::InvalidOperation,
                                                    "glReadPixels(RGBA_INTEGER, UNSIGNED_INT) into a pack buffer at byte offset 2 (not a multiple of 4) -> INVALID_OPERATION (4.3.2.4)",
                                                    std::format("glGetError() = {}, expected GL_INVALID_OPERATION", demo::error_name(e)), "ANGLE",
                                                    "ANGLE accepts a pack-buffer offset that is not a multiple of the size of the type's GL data type "
                                                    "(ES 3.0 4.3.2.4 requires INVALID_OPERATION; the unpack rule of 3.7.2.1 is enforced)");
                }
                expect_error(check, mgl::ErrorCode::NoError, "... and at byte offset 4 it succeeds",
                             [] { mgl::glReadPixels(0, 0, 2, 2, mgl::PixelFormat::RgbaInteger, mgl::PixelType::UnsignedInt, pack_offset(4)); });
                mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
                mgl::glDeleteBuffers(1, &pbo);
            }
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteFramebuffers(1, &fbo);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glDeleteRenderbuffers(1, &rb);
        }

        void check_map_pointer_state(demo::Check& check) const
        {
            mgl::BufferId pbo{};
            mgl::glGenBuffers(1, &pbo);
            make_buffer(mgl::BufferTarget::PixelPack, pbo, 256, 0x11, mgl::BufferUsage::StreamRead);
            GLint mapped = -1;
            mgl::glGetBufferParameteriv(mgl::BufferTarget::PixelPack, mgl::BufferParameter::Mapped, &mapped);
            void* p = mgl::glMapBufferRange(mgl::BufferTarget::PixelPack, 16, 64, mgl::MapBufferAccessMask::Read);
            void* q = nullptr;
            mgl::glGetBufferPointerv(mgl::BufferTarget::PixelPack, mgl::BufferPointerParameter::MapPointer, &q);
            GLint mapped_now = -1, offset = -1, length = -1;
            mgl::glGetBufferParameteriv(mgl::BufferTarget::PixelPack, mgl::BufferParameter::Mapped, &mapped_now);
            mgl::glGetBufferParameteriv(mgl::BufferTarget::PixelPack, mgl::BufferParameter::MapOffset, &offset);
            mgl::glGetBufferParameteriv(mgl::BufferTarget::PixelPack, mgl::BufferParameter::MapLength, &length);
            const bool first_byte = p != nullptr && static_cast<const std::uint8_t*>(p)[0] == 0x11 && static_cast<const std::uint8_t*>(p)[63] == 0x11;
            const bool unmapped = mgl::glUnmapBuffer(mgl::BufferTarget::PixelPack) != GL_FALSE;
            int sentinel = 0;
            void* after = &sentinel;  // overwritten by glGetBufferPointerv; stays non-null if the call does nothing
            mgl::glGetBufferPointerv(mgl::BufferTarget::PixelPack, mgl::BufferPointerParameter::MapPointer, &after);
            check.expect(mapped == GL_FALSE && p != nullptr && p == q && mapped_now == GL_TRUE && offset == 16 && length == 64 && first_byte && unmapped && after == nullptr,
                         "a pixel pack buffer maps like any buffer: BUFFER_MAPPED false -> true -> (unmap TRUE) with MAP_POINTER = the returned pointer, MAP_OFFSET 16, MAP_LENGTH 64, then NULL",
                         std::format("mapped {} -> {}, pointer {} / {}, offset {}, length {}", mapped, mapped_now, p, q, offset, length));
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
            mgl::glDeleteBuffers(1, &pbo);
        }

        // ------------------------------------------------------------------------------------------
        // The asynchronous pattern
        // ------------------------------------------------------------------------------------------
        static constexpr int kFrames = 6;
        static constexpr int kFrameSize = 16;

        static std::array<int, 4> frame_color(int k) { return {30 + 40 * k, 220 - 30 * k, 10 + 20 * k, 255 - 12 * k}; }

        // Frame k is rendered, glReadPixels queues its read-back into buffer k % 2 and returns at once; a fence marks the end of the read; one
        // frame later (when the GPU has had time to finish) the previous frame's buffer is mapped. Without the fence the map itself would block.
        void check_async_pattern(demo::Check& check)
        {
            mgl::RenderbufferId rb{};
            mgl::FramebufferId fbo{};
            mgl::glGenRenderbuffers(1, &rb);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, rb);
            mgl::glRenderbufferStorage(mgl::RenderbufferTarget::Renderbuffer, mgl::InternalFormat::Rgba8, kFrameSize, kFrameSize);
            mgl::glGenFramebuffers(1, &fbo);
            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, fbo);
            mgl::glFramebufferRenderbuffer(mgl::FramebufferTarget::Framebuffer, kColor0, mgl::RenderbufferTarget::Renderbuffer, rb);
            if (mgl::glCheckFramebufferStatus(mgl::FramebufferTarget::Framebuffer) != mgl::FramebufferStatus::Complete)
                throw demo::Failure("the RGBA8 renderbuffer framebuffer of the asynchronous test is not complete");

            const std::size_t bytes = static_cast<std::size_t>(kFrameSize) * kFrameSize * 4;
            std::array<mgl::BufferId, 2> pbo{};
            mgl::glGenBuffers(2, pbo.data());
            for (const mgl::BufferId b : pbo) make_buffer(mgl::BufferTarget::PixelPack, b, bytes, kFill, mgl::BufferUsage::StreamRead);
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});

            mgl::glViewport(0, 0, kFrameSize, kFrameSize);
            mgl::glDisable(mgl::Capability::Dither);
            std::array<GLsync, kFrames> fence{};
            int ok_frames = 0;
            int waits_signaled = 0;
            std::string first_bad;
            auto verify = [&](int k) {
                // Wait for the fence of frame k (a real application would poll it with timeout 0), then map ITS buffer.
                const mgl::SyncWaitResult r = mgl::glClientWaitSync(fence[static_cast<std::size_t>(k)], mgl::SyncFlushMask::Commands, 1'000'000'000ULL);
                if (r == mgl::SyncWaitResult::AlreadySignaled || r == mgl::SyncWaitResult::ConditionSatisfied) ++waits_signaled;
                mgl::glDeleteSync(fence[static_cast<std::size_t>(k)]);
                mgl::glBindBuffer(mgl::BufferTarget::PixelPack, pbo[static_cast<std::size_t>(k % 2)]);
                const std::vector<std::uint8_t> got = map_and_copy(mgl::BufferTarget::PixelPack, bytes);
                mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
                const std::array<int, 4> c = frame_color(k);
                const Diff d = compare_rgba(got, [&c](int, int) { return Rgba8{u8(c[0]), u8(c[1]), u8(c[2]), u8(c[3])}; }, kFrameSize, kFrameSize, 1);
                if (d.bad == 0)
                    ++ok_frames;
                else if (first_bad.empty())
                    first_bad = std::format("frame {}: {}", k, d.first);
            };
            for (int k = 0; k < kFrames; ++k)
            {
                const std::array<int, 4> c = frame_color(k);
                mgl::glClearColor(static_cast<float>(c[0]) / 255.0f, static_cast<float>(c[1]) / 255.0f, static_cast<float>(c[2]) / 255.0f, static_cast<float>(c[3]) / 255.0f);
                mgl::glClear(mgl::ClearBufferBit::Color);  // "render frame k"
                mgl::glBindBuffer(mgl::BufferTarget::PixelPack, pbo[static_cast<std::size_t>(k % 2)]);
                mgl::glReadPixels(0, 0, kFrameSize, kFrameSize, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, pack_offset(0));  // queued, returns at once
                mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
                fence[static_cast<std::size_t>(k)] = mgl::glFenceSync(mgl::SyncCondition::GpuCommandsComplete, mgl::SyncFlag::None);
                if (k >= 1) verify(k - 1);  // the previous frame: its buffer is not the one frame k is being read into
            }
            verify(kFrames - 1);

            check.expect(ok_frames == kFrames, std::format("asynchronous pattern: all {} frames read back exactly (+-1 for the clear color) from alternating buffers, each mapped one frame after its glReadPixels", kFrames), first_bad);
            check.expect(waits_signaled == kFrames, "every glClientWaitSync on the read-back fence returned ALREADY_SIGNALED or CONDITION_SATISFIED within one second", std::format("{} of {}", waits_signaled, kFrames));

            // "Just map it": no fence at all. glMapBufferRange waits for the pending read, so the data is right (2.10.3).
            {
                const std::array<int, 4> c = {201, 17, 99, 250};
                mgl::glClearColor(static_cast<float>(c[0]) / 255.0f, static_cast<float>(c[1]) / 255.0f, static_cast<float>(c[2]) / 255.0f, static_cast<float>(c[3]) / 255.0f);
                mgl::glClear(mgl::ClearBufferBit::Color);
                mgl::glBindBuffer(mgl::BufferTarget::PixelPack, pbo[0]);
                mgl::glReadPixels(0, 0, kFrameSize, kFrameSize, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, pack_offset(0));
                const std::vector<std::uint8_t> got = map_and_copy(mgl::BufferTarget::PixelPack, bytes);  // blocks until the read is complete
                mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
                const Diff d = compare_rgba(got, [&c](int, int) { return Rgba8{u8(c[0]), u8(c[1]), u8(c[2]), u8(c[3])}; }, kFrameSize, kFrameSize, 1);
                check.expect(d.bad == 0, "mapping the pack buffer immediately after glReadPixels (no fence) also returns the finished read-back", d.first);
            }

            mgl::glBindFramebuffer(mgl::FramebufferTarget::Framebuffer, mgl::FramebufferId{0});
            mgl::glDeleteBuffers(2, pbo.data());
            mgl::glDeleteFramebuffers(1, &fbo);
            mgl::glBindRenderbuffer(mgl::RenderbufferTarget::Renderbuffer, mgl::RenderbufferId{0});
            mgl::glDeleteRenderbuffers(1, &rb);
            mgl::glViewport(0, 0, kCells * kCellPx, kCellPx);
        }

        // ------------------------------------------------------------------------------------------
        // Defined errors
        // ------------------------------------------------------------------------------------------
        void check_unpack_errors(demo::Check& check)
        {
            mgl::TextureId scratch{};
            mgl::glGenTextures(1, &scratch);
            mgl::glBindTexture(mgl::TextureTarget::Texture2D, scratch);
            nearest_clamp();
            const mgl::TextureTarget t2d = mgl::TextureTarget::Texture2D;
            using IF = mgl::InternalFormat;
            using PF = mgl::PixelFormat;
            const mgl::PixelType ub = mgl::PixelType::UnsignedByte;

            mgl::BufferId pbo{};
            mgl::glGenBuffers(1, &pbo);

            // 3.7.2.1 / 3.8.3: unpacking beyond the end of the buffer is INVALID_OPERATION; the exact size is fine.
            make_buffer(mgl::BufferTarget::PixelUnpack, pbo, kChartBytes - 1, 0x55, mgl::BufferUsage::StaticDraw);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexImage2D 32x32 RGBA from a 4095-byte unpack buffer (4096 bytes are needed) -> INVALID_OPERATION",
                         [&] { mgl::glTexImage2D(t2d, 0, IF::Rgba8, kChart, kChart, 0, PF::Rgba, ub, demo::buffer_offset(0)); });
            make_buffer(mgl::BufferTarget::PixelUnpack, pbo, kChartBytes, 0x55, mgl::BufferUsage::StaticDraw);
            expect_error(check, mgl::ErrorCode::NoError, "... the same call from a buffer of exactly 4096 bytes succeeds",
                         [&] { mgl::glTexImage2D(t2d, 0, IF::Rgba8, kChart, kChart, 0, PF::Rgba, ub, demo::buffer_offset(0)); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "... and from that buffer at byte offset 1 it is one byte too much -> INVALID_OPERATION",
                         [&] { mgl::glTexImage2D(t2d, 0, IF::Rgba8, kChart, kChart, 0, PF::Rgba, ub, demo::buffer_offset(1)); });

            // The skip and row-length modes enlarge the footprint: 4160 + (SKIP_ROWS + 12) * 128 + 20 * 4 bytes.
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, buf_[kBufImages]);
            set_unpack(kChart, 0, 20);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexImage2D 20x13 with ROW_LENGTH 32 and SKIP_ROWS 20 from offset 4160 reaches byte 8336 of an 8256-byte buffer -> INVALID_OPERATION",
                         [&] { mgl::glTexImage2D(t2d, 0, IF::Rgba8, 20, 13, 0, PF::Rgba, ub, demo::buffer_offset(kSecondChart)); });
            set_unpack(kChart, 0, 19);
            expect_error(check, mgl::ErrorCode::NoError, "... with SKIP_ROWS 19 the last byte is 8208 and the upload succeeds",
                         [&] { mgl::glTexImage2D(t2d, 0, IF::Rgba8, 20, 13, 0, PF::Rgba, ub, demo::buffer_offset(kSecondChart)); });
            reset_unpack();

            // glTexSubImage2D obeys the same rule, and a failed call leaves the texture unchanged (an error has no effect, 2.5).
            const ShowCell& patch = cells_[3];
            mgl::glBindTexture(t2d, patch.tex);
            const std::vector<std::uint8_t> before = read_attached(patch.tex, kChart, kChart);
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, buf_[kBufImages]);
            set_unpack(kChart, 0, 25);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexSubImage2D 14x10 with SKIP_ROWS 25 from offset 4160 runs past the end of the buffer -> INVALID_OPERATION",
                         [&] { mgl::glTexSubImage2D(t2d, 0, 6, 9, 14, 10, PF::Rgba, ub, demo::buffer_offset(kSecondChart)); });
            reset_unpack();
            const std::vector<std::uint8_t> after = read_attached(patch.tex, kChart, kChart);
            check.expect(!before.empty() && before == after, "the rejected glTexSubImage2D did not modify the texture (all 1024 texels equal)");

            // The offset must be a multiple of the size of the data type: UNSIGNED_SHORT_4_4_4_4 elements are 2 bytes.
            mgl::glBindTexture(t2d, scratch);
            make_buffer(mgl::BufferTarget::PixelUnpack, pbo, 64, 0x55, mgl::BufferUsage::StaticDraw);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexImage2D RGBA / UNSIGNED_SHORT_4_4_4_4 from byte offset 3 (not a multiple of 2) -> INVALID_OPERATION (3.7.2.1)",
                         [&] { mgl::glTexImage2D(t2d, 0, IF::Rgba4, 4, 4, 0, PF::Rgba, mgl::PixelType::UnsignedShort4444, demo::buffer_offset(3)); });
            expect_error(check, mgl::ErrorCode::NoError, "... from byte offset 4 it succeeds",
                         [&] { mgl::glTexImage2D(t2d, 0, IF::Rgba4, 4, 4, 0, PF::Rgba, mgl::PixelType::UnsignedShort4444, demo::buffer_offset(4)); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glTexSubImage2D with the same type from byte offset 5 -> INVALID_OPERATION",
                         [&] { mgl::glTexSubImage2D(t2d, 0, 0, 0, 2, 2, PF::Rgba, mgl::PixelType::UnsignedShort4444, demo::buffer_offset(5)); });
            expect_error(check, mgl::ErrorCode::NoError, "a UNSIGNED_BYTE upload from odd byte offset 5 is fine (1-byte elements)",
                         [&] { mgl::glTexImage2D(t2d, 0, IF::Rgba8, 2, 2, 0, PF::Rgba, ub, demo::buffer_offset(5)); });

            // With no buffer object bound, targets cannot be used (2.10.1).
            mgl::glBindBuffer(mgl::BufferTarget::PixelUnpack, mgl::BufferId{0});
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glBufferData on PIXEL_UNPACK_BUFFER with buffer 0 bound -> INVALID_OPERATION (2.10.1)",
                         [] { mgl::glBufferData(mgl::BufferTarget::PixelUnpack, 16, nullptr, mgl::BufferUsage::StaticDraw); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glGetBufferParameteriv on PIXEL_PACK_BUFFER with buffer 0 bound -> INVALID_OPERATION",
                         [] { GLint v = 0; mgl::glGetBufferParameteriv(mgl::BufferTarget::PixelPack, mgl::BufferParameter::Size, &v); });

            mgl::glDeleteBuffers(1, &pbo);
            mgl::glBindTexture(t2d, mgl::TextureId{0});
            mgl::glDeleteTextures(1, &scratch);
        }

        void check_pack_errors(demo::Check& check)
        {
            mgl::BufferId pbo{};
            mgl::glGenBuffers(1, &pbo);
            // 4.3.2.4: packing beyond the end of the buffer is INVALID_OPERATION. An 8x8 RGBA/UNSIGNED_BYTE read is 256 bytes.
            make_buffer(mgl::BufferTarget::PixelPack, pbo, 255, kFill, mgl::BufferUsage::StreamRead);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glReadPixels 8x8 RGBA into a 255-byte pack buffer (256 bytes are needed) -> INVALID_OPERATION",
                         [] { mgl::glReadPixels(0, 0, 8, 8, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, pack_offset(0)); });
            const std::vector<std::uint8_t> untouched = map_and_copy(mgl::BufferTarget::PixelPack, 255);
            check.expect(!untouched.empty() && std::all_of(untouched.begin(), untouched.end(), [](std::uint8_t b) { return b == kFill; }), "the rejected glReadPixels wrote nothing into the buffer");
            make_buffer(mgl::BufferTarget::PixelPack, pbo, 256, kFill, mgl::BufferUsage::StreamRead);
            expect_error(check, mgl::ErrorCode::NoError, "... into a buffer of exactly 256 bytes it succeeds",
                         [] { mgl::glReadPixels(0, 0, 8, 8, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, pack_offset(0)); });
            expect_error(check, mgl::ErrorCode::InvalidOperation, "... at byte offset 1 it is one byte too much -> INVALID_OPERATION",
                         [] { mgl::glReadPixels(0, 0, 8, 8, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, pack_offset(1)); });
            mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipRows, 1);
            expect_error(check, mgl::ErrorCode::InvalidOperation, "... and with PACK_SKIP_ROWS 1 (one more row of 32 bytes in front) it is too much as well",
                         [] { mgl::glReadPixels(0, 0, 8, 8, mgl::PixelFormat::Rgba, mgl::PixelType::UnsignedByte, pack_offset(0)); });
            mgl::glPixelStorei(mgl::PixelStoreParam::PackSkipRows, 0);
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
            mgl::glDeleteBuffers(1, &pbo);
            // The misaligned offset of the 4-byte type is checked together with the integer read in check_pack_integer().
        }

        // 2.10.3: a mapped buffer cannot be mapped again or have its data replaced; the explicitly described errors.
        void check_mapping_rules(demo::Check& check)
        {
            mgl::BufferId pbo{};
            mgl::glGenBuffers(1, &pbo);
            make_buffer(mgl::BufferTarget::PixelPack, pbo, 256, 0x22, mgl::BufferUsage::StreamRead);
            void* p = mgl::glMapBufferRange(mgl::BufferTarget::PixelPack, 0, 256, mgl::MapBufferAccessMask::Read);
            check.expect(p != nullptr, "glMapBufferRange(READ) of a 256-byte pixel pack buffer returns a pointer");
            expect_error(check, mgl::ErrorCode::InvalidOperation, "mapping a pixel pack buffer that is already mapped -> INVALID_OPERATION (2.10.3)", [] {
                void* again = mgl::glMapBufferRange(mgl::BufferTarget::PixelPack, 0, 16, mgl::MapBufferAccessMask::Read);
                if (again != nullptr) (void)mgl::glUnmapBuffer(mgl::BufferTarget::PixelPack);
            });
            const std::array<std::uint8_t, 8> bytes = {1, 2, 3, 4, 5, 6, 7, 8};
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glBufferSubData into a range of a mapped pixel buffer -> INVALID_OPERATION (2.10.2)",
                         [&] { mgl::glBufferSubData(mgl::BufferTarget::PixelPack, 0, 8, bytes.data()); });
            check.expect(mgl::glUnmapBuffer(mgl::BufferTarget::PixelPack) != GL_FALSE, "glUnmapBuffer returns TRUE for an intact mapping");
            expect_error(check, mgl::ErrorCode::InvalidOperation, "glUnmapBuffer of a buffer that is not mapped -> INVALID_OPERATION (2.10.3.1)", [] { (void)mgl::glUnmapBuffer(mgl::BufferTarget::PixelPack); });
            mgl::glBindBuffer(mgl::BufferTarget::PixelPack, mgl::BufferId{0});
            mgl::glDeleteBuffers(1, &pbo);
        }

        mgl::ProgramId show_{};
        mgl::UniformLocation show_tex_{};
        std::array<mgl::BufferId, kBufCount> buf_{};
        std::array<mgl::TextureId, kTexCount> tex_{};
        mgl::FramebufferId attach_fbo_{};
        std::vector<ShowCell> cells_;
        std::vector<std::uint8_t> cell0_;
    };
}

int main(int argc, char** argv)
{
    const demo::Info info{
        .name = "es30-pixel-buffer-objects",
        .title = "Pixel buffer objects: texture uploads and read-backs through buffer objects",
        .min_es = {3, 0},
        .summary = "PIXEL_UNPACK_BUFFER as the source of glTexImage2D/glTexSubImage2D (byte-offset pointers, ROW_LENGTH/SKIP selecting sub-rectangles) and PIXEL_PACK_BUFFER "
                   "as the destination of glReadPixels read back with glMapBufferRange, the asynchronous ping-pong pattern, bindings, pixel-store queries and errors.",
        .source_dir = DEMO_SOURCE_DIR,
    };
    return demo::run<PixelBufferObjects>(argc, argv, info);
}
