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


Parameters::Parameters(int argc_, char **argv_) {
    argc = argc_;
    argv = argv_;
}

void Parameters::print() {
    std::cerr <<  "sa=" << match_score <<" , sb=" << mismatch_score <<" , gapo=" <<  gap_open_score << " , gape="<<gap_ext_score << std::endl;
    std::cerr <<  "start_pos=" << start_pos <<" , print_out=" << print_out <<" , n_threads=" <<  n_threads << std::endl;
    std::cerr <<  "semiglobal_skipping_head=" << semiglobal_skipping_head <<" , semiglobal_skipping_tail=" << semiglobal_skipping_tail <<" , algo=" <<  algo << std::endl;
    std::cerr <<  std::boolalpha << "isPacked = " << isPacked  << " , secondBest = " << secondBest << std::endl;
    std::cerr <<  "query_batch_fasta_filename=" << query_batch_fasta_filename <<" , target_batch_fasta_filename=" << target_batch_fasta_filename << std::endl;
}

void Parameters::failure(fail_type f) {
    switch(f)
    {
            case NOT_ENOUGH_ARGS:
                std::cerr << "Not enough Parameters. Required: -y AL_TYPE file1.fasta file2.fasta. See help (--help, -h) for usage. " << std::endl;
            break;
            case WRONG_ARG:
                std::cerr << "Wrong argument. See help (--help, -h) for usage. " << std::endl;
            break;
            case WRONG_FILES:
                std::cerr << "File error: either a file doesn't exist, or cannot be opened." << std::endl;
            break;

            default:
            break;
    }
    exit(1);
}

void Parameters::help() {
            std::cerr << "Usage: ./test_prog.out [-a] [-b] [-q] [-r] [-s] [-t] [-p] [-n] [-y] <query_batch.fasta> <target_batch.fasta>" << std::endl;
            std::cerr << "Options: -a INT    match score ["<< match_score <<"]" << std::endl;
            std::cerr << "         -b INT    mismatch penalty [" << mismatch_score << "]"<< std::endl;
            std::cerr << "         -q INT    gap open penalty [" << gap_open_score << "]" << std::endl;
            std::cerr << "         -r INT    gap extension penalty ["<< gap_ext_score <<"]" << std::endl;
            std::cerr << "         -s        find the start position" << std::endl;
            std::cerr << "         -t        compute traceback. With this option enabled, \"-s\" has no effect as start position will always be computed with traceback" << std::endl;
            std::cerr << "         -p        print the alignment results" << std::endl;
            std::cerr << "         -n INT    Number of threads ["<< n_threads<<"]" << std::endl;
            std::cerr << "         -y AL_TYPE       Alignment type . Must be \"local\", \"semi_global\", \"global\", \"ksw\" "  << std::endl;
	    std::cerr << "         -x HEAD TAIL     specifies, for semi-global alignment, wha should be skipped for heads and tails of the sequences. (NONE, QUERY, TARGET, BOTH)" << std::endl;
            std::cerr << "         -k INT    Band width in case \"banded\" is selected."  << std::endl;
            std::cerr << "         --help, -h : displays this message." << std::endl;
            std::cerr << "         --second-best   displays second best score (WITHOUT_START only)." << std::endl;
            std::cerr << "Single-pack multi-Parameters (e.g. -sp) is not supported." << std::endl;
            std::cerr << "		  "  << std::endl;
}


void Parameters::parse() {
    // before testing anything, check if calling for help.
    int c;

    std::string arg_next = "";
    std::string arg_cur = "";

    for (c = 1; c < argc; c++)
    {
        arg_cur = std::string((const char*) (*(argv + c) ) );
        arg_next = "";
        if (!arg_cur.compare("--help") || !arg_cur.compare("-h"))
        {
            help();
            exit(0);
        }
    }

    if (argc < 4)
    {
        failure(NOT_ENOUGH_ARGS);
    }

    for (c = 1; c < argc - 2; c++)
    {
        arg_cur = std::string((const char*) (*(argv + c) ) );
        if (arg_cur.at(0) == '-' && arg_cur.at(1) == '-' )
        {
            if (!arg_cur.compare("--help"))
            {
                help();
                exit(0);
            }
            if (!arg_cur.compare("--second-best"))
            {
                secondBest = Bool::TRUE;
            }

        } else if (arg_cur.at(0) == '-' )
        {
            if (arg_cur.length() > 2)
                failure(WRONG_ARG);
            char param = arg_cur.at(1);
            switch(param)
            {
                case 'y':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    if (!arg_next.compare("local"))
                        algo = algo_type::LOCAL;
                    else if (!arg_next.compare("semi_global"))
                        algo = algo_type::SEMI_GLOBAL;
                    else if (!arg_next.compare("global"))
                        algo = algo_type::GLOBAL;
                    else if (!arg_next.compare("ksw"))
                    {
                        algo = algo_type::KSW;
                    }
                break;
                case 'a':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    match_score = std::stoi(arg_next);
                break;
                case 'b':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    mismatch_score = std::stoi(arg_next);
                break;
                case 'q':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    gap_open_score = std::stoi(arg_next);
                break;
                case 'r':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    gap_ext_score = std::stoi(arg_next);
                break;
                case 's':
                    start_pos = CompStart::WITH_START;
                break;
                case 't':
                	start_pos = CompStart::WITH_TB;
                	break;
                case 'p':
                    print_out = 1;
                break;
                case 'n':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    n_threads = std::stoi(arg_next);
                break;
                case 'k':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    k_band = std::stoi(arg_next);
                break;
                case 'x':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    if (!arg_next.compare("NONE"))
                        semiglobal_skipping_head = DataSource::NONE;
                    else if (!arg_next.compare("TARGET"))
                        semiglobal_skipping_head = DataSource::TARGET;
                    else if (!arg_next.compare("QUERY"))
                        semiglobal_skipping_head = DataSource::QUERY;
                    else if (!arg_next.compare("BOTH"))
                        semiglobal_skipping_head = DataSource::BOTH;
                    else
                    {
                        failure(WRONG_ARG);
                    }

                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    if (!arg_next.compare("NONE"))
                        semiglobal_skipping_tail = DataSource::NONE;
                    else if (!arg_next.compare("TARGET"))
                        semiglobal_skipping_tail = DataSource::TARGET;
                    else if (!arg_next.compare("QUERY"))
                        semiglobal_skipping_tail = DataSource::QUERY;
                    else if (!arg_next.compare("BOTH"))
                        semiglobal_skipping_tail = DataSource::BOTH;
                    else
                    {
                        failure(WRONG_ARG);
                    }
                break;

            }


        } else {
            failure(WRONG_ARG);
        }
    }


    // the last 2 Parameters are the 2 filenames.
    query_batch_fasta_filename = std::string( (const char*)  (*(argv + c) ) );
    c++;
    target_batch_fasta_filename = std::string( (const char*) (*(argv + c) ) );
}
