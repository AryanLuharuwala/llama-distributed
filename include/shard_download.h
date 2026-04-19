// shard_download.h
//
// Blocking HTTP fetch for GGUF shards.  Supports http:// URLs with optional
// query strings (used for signed shard URLs: /models/N/shards/S?exp=...&sig=...).
// Writes to `dest_path`; returns false on error.  No https support yet (M5).

#pragma once

#include <string>

namespace dist {

bool fetch_shard(const std::string & url, const std::string & dest_path,
                 std::string & err);

} // namespace dist
