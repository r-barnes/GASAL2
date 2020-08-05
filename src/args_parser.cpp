#include <fstream>
#include <iostream>

#include <gasal2/args_parser.h>

std::ostream& operator<<(std::ostream &out, const algo_type value){
    out<<static_cast<typename std::underlying_type<algo_type>::type>(value);
    return out;
}

std::ostream& operator<<(std::ostream &out, const DataSource value){
    out<<static_cast<typename std::underlying_type<DataSource>::type>(value);
    return out;
}

std::ostream& operator<<(std::ostream &out, const Bool value){
    out<<static_cast<typename std::underlying_type<Bool>::type>(value);
    return out;
}

std::ostream& operator<<(std::ostream &out, const CompStart value){
    out<<static_cast<typename std::underlying_type<CompStart>::type>(value);
    return out;
}

void Parameters::print() {
    std::cerr <<  "sa=" << match_score <<" , sb=" << mismatch_score <<" , gapo=" <<  gap_open_score << " , gape="<<gap_ext_score << std::endl;
    std::cerr <<  "start_pos=" << start_pos <<" , print_out=" << print_out <<" , n_threads=" <<  n_threads << std::endl;
    std::cerr <<  "semiglobal_skipping_head=" << semiglobal_skipping_head <<" , semiglobal_skipping_tail=" << semiglobal_skipping_tail <<" , algo=" <<  algo << std::endl;
    std::cerr <<  std::boolalpha << "isPacked = " << isPacked  << " , secondBest = " << secondBest << std::endl;
    std::cerr <<  "query_batch_fasta_filename=" << query_batch_fasta_filename <<" , target_batch_fasta_filename=" << target_batch_fasta_filename << std::endl;
}
