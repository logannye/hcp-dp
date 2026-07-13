#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
#include "wavefront/wfa.h"
}

namespace {

std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (true) {
    const std::size_t tab = line.find('\t', start);
    fields.push_back(line.substr(start, tab - start));
    if (tab == std::string::npos) break;
    start = tab + 1;
  }
  return fields;
}

bool parse_int(const std::string& value, int& output) {
  try {
    std::size_t consumed = 0;
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size() || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
      return false;
    }
    output = static_cast<int>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

std::string rle_cigar(const cigar_t* cigar, int begin, int end,
                      int& query_consumed, int& target_consumed) {
  std::ostringstream output;
  char current = '\0';
  int count = 0;
  query_consumed = 0;
  target_consumed = 0;
  for (int i = begin; i < end; ++i) {
    const char operation = cigar->operations[i];
    if (operation == 'M' || operation == 'X') {
      ++query_consumed;
      ++target_consumed;
    } else if (operation == 'D') {
      ++query_consumed;  // WFA pattern consumption; normalized to SAM I in Rust.
    } else if (operation == 'I') {
      ++target_consumed;  // WFA text consumption; normalized to SAM D in Rust.
    } else {
      return {};
    }
    if (operation == current) {
      ++count;
    } else {
      if (count > 0) output << count << current;
      current = operation;
      count = 1;
    }
  }
  if (count > 0) output << count << current;
  return output.str();
}

void emit_error(const std::string& message) {
  std::string clean = message;
  for (char& value : clean) {
    if (value == '\t' || value == '\n' || value == '\r') value = ' ';
  }
  std::cout << "error\t" << clean << '\n' << std::flush;
}

void align_request(const std::vector<std::string>& fields) {
  if (fields.size() != 6) {
    emit_error("request must have six tab-delimited fields");
    return;
  }
  const std::string& mode = fields[0];
  int mismatch = 0;
  int opening = 0;
  int extension = 0;
  if (!parse_int(fields[1], mismatch) || !parse_int(fields[2], opening) ||
      !parse_int(fields[3], extension) || mismatch <= 0 || extension <= 0) {
    emit_error("invalid WFA penalties");
    return;
  }
  const std::string& pattern = fields[4];
  const std::string& text = fields[5];
  if (pattern.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    emit_error("sequence length exceeds WFA integer coordinates");
    return;
  }

  wavefront_aligner_attr_t attributes = wavefront_aligner_attr_default;
  attributes.alignment_scope = compute_alignment;
  attributes.heuristic.strategy = wf_heuristic_none;
  if (mode == "edit-distance") {
    attributes.distance_metric = edit;
  } else if (mode == "global-linear" || mode == "semiglobal-linear") {
    attributes.distance_metric = gap_linear;
    attributes.linear_penalties.match = 0;
    attributes.linear_penalties.mismatch = mismatch;
    attributes.linear_penalties.indel = extension;
  } else if (mode == "global-affine") {
    attributes.distance_metric = gap_affine;
    attributes.affine_penalties.match = 0;
    attributes.affine_penalties.mismatch = mismatch;
    attributes.affine_penalties.gap_opening = opening;
    attributes.affine_penalties.gap_extension = extension;
  } else {
    std::cout << "unsupported\tmode " << mode << '\n' << std::flush;
    return;
  }

  wavefront_aligner_t* aligner = wavefront_aligner_new(&attributes);
  if (aligner == nullptr) {
    emit_error("wavefront_aligner_new returned null");
    return;
  }
  wavefront_aligner_set_heuristic_none(aligner);
  if (mode == "semiglobal-linear") {
    wavefront_aligner_set_alignment_free_ends(
        aligner, 0, 0, static_cast<int>(text.size()),
        static_cast<int>(text.size()));
  } else {
    wavefront_aligner_set_alignment_end_to_end(aligner);
  }
  const int status = wavefront_align(
      aligner, pattern.data(), static_cast<int>(pattern.size()), text.data(),
      static_cast<int>(text.size()));
  if (status != WF_STATUS_ALG_COMPLETED) {
    const std::string message = wavefront_align_strerror(status);
    wavefront_aligner_delete(aligner);
    emit_error(message);
    return;
  }

  int begin = aligner->cigar->begin_offset;
  int end = aligner->cigar->end_offset;
  int free_target_prefix = 0;
  int free_target_suffix = 0;
  if (mode == "semiglobal-linear") {
    while (begin < end && aligner->cigar->operations[begin] == 'I') {
      ++begin;
      ++free_target_prefix;
    }
    while (end > begin && aligner->cigar->operations[end - 1] == 'I') {
      --end;
      ++free_target_suffix;
    }
  }
  int query_consumed = 0;
  int target_consumed = 0;
  const std::string cigar = rle_cigar(aligner->cigar, begin, end,
                                      query_consumed, target_consumed);
  int query_end = aligner->cigar->end_v;
  int target_end = aligner->cigar->end_h;
  int query_start = query_end - query_consumed;
  int target_start = target_end - target_consumed;
  if (mode == "semiglobal-linear") {
    query_start = 0;
    query_end = static_cast<int>(pattern.size());
    if (begin == end && pattern.empty()) {
      target_start = 0;
      target_end = 0;
    } else {
      target_start = free_target_prefix;
      target_end = static_cast<int>(text.size()) - free_target_suffix;
    }
  }
  const int cost = mode == "edit-distance" ? aligner->cigar->score
                                             : -aligner->cigar->score;
  std::cout << "ok\t" << cost << '\t' << query_start << '\t' << query_end
            << '\t' << target_start << '\t' << target_end << '\t' << cigar
            << '\n' << std::flush;
  wavefront_aligner_delete(aligner);
}

}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::string line;
  while (std::getline(std::cin, line)) {
    align_request(split_tabs(line));
  }
  return 0;
}
