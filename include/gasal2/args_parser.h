#pragma once

#include "gasal.h"

#include <string>

enum class FailType {
    NOT_ENOUGH_ARGS,
    TOO_MANY_ARGS,
    WRONG_ARG,
    WRONG_FILES,
    WRONG_ALGO
};

struct Parameters{
    void print();

    int32_t match_score    = 1;
    int32_t mismatch_score = 4;
    int32_t gap_open_score = 6;
    int32_t gap_ext_score  = 1;
    CompStart start_pos = CompStart::WITHOUT_START;
    int print_out  = 0;
    int n_threads  = 1;
    int32_t k_band = 0;

    Bool secondBest = Bool::FALSE;

    bool isPacked = false;
    bool isReverseComplement = false;

    DataSource semiglobal_skipping_head = DataSource::TARGET;
    DataSource semiglobal_skipping_tail = DataSource::TARGET;

    algo_type algo = algo_type::UNKNOWN;

    std::string query_batch_fasta_filename = "";
    std::string target_batch_fasta_filename = "";
};
