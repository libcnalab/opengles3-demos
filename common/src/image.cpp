#include "demo/image.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>

namespace demo
{
    namespace
    {
        std::array<std::uint32_t, 256> make_crc_table()
        {
            std::array<std::uint32_t, 256> t{};
            for (std::uint32_t n = 0; n < 256; ++n)
            {
                std::uint32_t c = n;
                for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                t[n] = c;
            }
            return t;
        }

        std::uint32_t crc32(const std::uint8_t* data, std::size_t len, std::uint32_t crc = 0xFFFFFFFFu)
        {
            static const auto table = make_crc_table();
            for (std::size_t i = 0; i < len; ++i) crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
            return crc;
        }

        void put_u32(std::vector<std::uint8_t>& v, std::uint32_t x)
        {
            v.push_back(static_cast<std::uint8_t>(x >> 24));
            v.push_back(static_cast<std::uint8_t>(x >> 16));
            v.push_back(static_cast<std::uint8_t>(x >> 8));
            v.push_back(static_cast<std::uint8_t>(x));
        }

        void chunk(std::vector<std::uint8_t>& out, const char type[4], const std::vector<std::uint8_t>& payload)
        {
            put_u32(out, static_cast<std::uint32_t>(payload.size()));
            const std::size_t start = out.size();
            out.insert(out.end(), type, type + 4);
            out.insert(out.end(), payload.begin(), payload.end());
            put_u32(out, crc32(out.data() + start, out.size() - start) ^ 0xFFFFFFFFu);
        }
    }

    bool write_png_rgba8(const std::string& path, int width, int height, std::span<const std::uint8_t> rgba,
                         bool bottom_up)
    {
        if (width <= 0 || height <= 0 ||
            rgba.size() < static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u)
            return false;

        // Raw scanlines with filter byte 0.
        const std::size_t row_bytes = static_cast<std::size_t>(width) * 4u;
        std::vector<std::uint8_t> raw;
        raw.reserve((row_bytes + 1) * static_cast<std::size_t>(height));
        for (int y = 0; y < height; ++y)
        {
            const int src_row = bottom_up ? height - 1 - y : y;
            raw.push_back(0);
            const auto* row = rgba.data() + static_cast<std::size_t>(src_row) * row_bytes;
            raw.insert(raw.end(), row, row + row_bytes);
        }

        // zlib stream made of stored (uncompressed) deflate blocks + Adler-32.
        std::vector<std::uint8_t> z{0x78, 0x01};
        std::size_t pos = 0;
        do
        {
            const std::size_t n = std::min<std::size_t>(65535, raw.size() - pos);
            const bool final_block = pos + n == raw.size();
            z.push_back(final_block ? 1 : 0);
            z.push_back(static_cast<std::uint8_t>(n & 0xFFu));
            z.push_back(static_cast<std::uint8_t>(n >> 8));
            z.push_back(static_cast<std::uint8_t>(~n & 0xFFu));
            z.push_back(static_cast<std::uint8_t>((~n >> 8) & 0xFFu));
            z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos),
                     raw.begin() + static_cast<std::ptrdiff_t>(pos + n));
            pos += n;
        } while (pos < raw.size());
        std::uint32_t a = 1, b = 0;
        for (const std::uint8_t byte : raw)
        {
            a = (a + byte) % 65521u;
            b = (b + a) % 65521u;
        }
        put_u32(z, (b << 16) | a);

        std::vector<std::uint8_t> png{0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
        std::vector<std::uint8_t> ihdr;
        put_u32(ihdr, static_cast<std::uint32_t>(width));
        put_u32(ihdr, static_cast<std::uint32_t>(height));
        ihdr.insert(ihdr.end(), {8, 6, 0, 0, 0});  // 8-bit RGBA, no interlace
        chunk(png, "IHDR", ihdr);
        chunk(png, "IDAT", z);
        chunk(png, "IEND", {});

        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) return false;
        const bool ok = std::fwrite(png.data(), 1, png.size(), f) == png.size();
        return std::fclose(f) == 0 && ok;
    }
}
