#pragma once

#include "gasal.h"

#include <string>

enum fail_type {
    NOT_ENOUGH_ARGS,
    TOO_MANY_ARGS,
    WRONG_ARG,
    WRONG_FILES,
    WRONG_ALGO
};

class Parameters{
    public:
        Parameters(int argc, char** argv);
        void print();
        void failure(fail_type f);
        void help();
        void parse();

        int32_t match_score   = 1;
        int32_t mismatch_score   = 4;
        int32_t gap_open_score = 6;
        int32_t gap_ext_score = 1;
        comp_start start_pos = WITHOUT_START;
        int print_out  = 0;
        int n_threads  = 1;
        int32_t k_band = 0;

        Bool secondBest = FALSE;

        bool isPacked = false;
        bool isReverseComplement = false;

        data_source semiglobal_skipping_head = TARGET;
        data_source semiglobal_skipping_tail = TARGET;

        algo_type algo = UNKNOWN;

        std::string query_batch_fasta_filename = "";
        std::string target_batch_fasta_filename = "";

    private:
        int argc;
        char** argv;
};
