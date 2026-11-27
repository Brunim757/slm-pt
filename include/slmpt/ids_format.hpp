#pragma once
#include <cstdint>

namespace slmpt {
namespace idsfmt {

// On-disk format for a corpus ids file (little-endian):
//   bytes 0..7   magic "SLMPTIDS"
//   bytes 8..11  dtype (2 = uint16 ids)
//   bytes 12..15 element count (uint32)
//   bytes 16..   count * sizeof(id) payload
inline constexpr char kMagic[8] = {'S', 'L', 'M', 'P', 'T', 'I', 'D', 'S'};
inline constexpr uint32_t kDtypeU16 = 2;
inline constexpr int kHeaderBytes = 16;

}  // namespace idsfmt
}  // namespace slmpt
