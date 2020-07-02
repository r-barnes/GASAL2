#pragma once

#include <string>
#include <vector>

struct FastaInput {
  std::vector<std::string> sequences;
  std::vector<std::string> headers;
  std::vector<uint8_t>     modifiers;
  size_t maximum_sequence_length = 0;
  size_t total_sequence_bytes    = 0;
};

FastaInput ReadFasta(const std::string &filename);

std::pair<FastaInput,FastaInput> ReadFastaQueryTargetPair(const std::string &query, const std::string &target);