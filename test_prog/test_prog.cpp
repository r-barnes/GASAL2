#include <gasal2/gasal_header.h>

#include <albp/cli_options.hpp>
#include <albp/read_fasta.hpp>
#include <albp/timer.hpp>

#include <cmath>
#include <iostream>
#include <map>
#include <omp.h>
#include <vector>

using namespace albp;

const size_t NB_STREAMS = 2;

//#define STREAM_BATCH_SIZE (262144)
// this gives each stream HALF of the sequences.
//#define STREAM_BATCH_SIZE ceil((double)target_seqs.size() / (double)(2))

#define STREAM_BATCH_SIZE 5000//ceil((double)target_seqs.size() / (double)(2 * 2))


// #define DEBUG

//A struct to hold data structures of a stream
struct gpu_batch{
  gasal_gpu_storage_t *gpu_storage; //the struct that holds the GASAL2 data structures
  int n_seqs_batch;                 //number of sequences in the batch (<= (target_seqs.size() / NB_STREAMS))
  int batch_start;                  //starting index of batch
};



char op_to_letter(const int op){
  switch (op) {
    case 0: return 'M';
    case 1: return 'X';
    case 2: return 'D';
    case 3: return 'I';
    default: return 'E';
  }
}



void print_batch(
  const FastaPair &input_data,
  const gpu_batch &batch,
  const Parameters &args
){
  auto &batch_storage = *batch.gpu_storage;
  #pragma omp critical
  for (int j=0, i = batch.batch_start; j < batch.n_seqs_batch; i++, j++) {
    std::cout << "query_name="    << input_data.a.headers.at(i);
    std::cout << "\ttarget_name=" << input_data.b.headers.at(i);
    std::cout << "\tscore=" << batch_storage.host_res->aln_score[j];

    if ((args.start_pos == CompStart::WITH_START || args.start_pos == CompStart::WITH_TB)
      && ((args.algo == algo_type::SEMI_GLOBAL && (args.semiglobal_skipping_head != DataSource::NONE || args.semiglobal_skipping_head != DataSource::NONE))
        || args.algo > algo_type::SEMI_GLOBAL))
    {
      std::cout << "\tquery_batch_start=" << batch_storage.host_res->query_batch_start[j];
      std::cout << "\ttarget_batch_start=" << batch_storage.host_res->target_batch_start[j];
    }

    if (args.algo != algo_type::GLOBAL){
      std::cout << "\tquery_batch_end="  << batch_storage.host_res->query_batch_end[j];
      std::cout << "\ttarget_batch_end=" << batch_storage.host_res->target_batch_end[j] ;
    }

    if (args.secondBest==Bool::TRUE){
      std::cout << "\t2nd_score=" << batch_storage.host_res_second->aln_score[j] ;
      std::cout << "\t2nd_query_batch_end="  << batch_storage.host_res_second->query_batch_end[j];
      std::cout << "\t2nd_target_batch_end=" << batch_storage.host_res_second->target_batch_end[j] ;
    }

    if (args.start_pos == CompStart::WITH_TB){
      std::cout << "\tCIGAR=";
      const int offset = batch_storage.host_query_batch_offsets[j];
      const int n_cigar_ops = batch_storage.host_res->n_cigar_ops[j];
      int last_op = (batch_storage.host_res->cigar[offset + n_cigar_ops - 1]) & 3;
      int count = (batch_storage.host_res->cigar[offset + n_cigar_ops - 1]) >> 2;
      for (int u = n_cigar_ops - 2; u >= 0 ; u--){
        int curr_op = (batch_storage.host_res->cigar[offset + u]) & 3;
        if (curr_op == last_op) {
          count += (batch_storage.host_res->cigar[offset + u]) >> 2;
        } else {
          std::cout << count << op_to_letter(last_op);
          count = (batch_storage.host_res->cigar[offset + u]) >> 2;
        }
        last_op = curr_op;
      }
      std::cout << count << op_to_letter(last_op);
    }
    std::cout << std::endl;
  }
}



void per_thread_processing(
  const FastaPair &input_data,
  const Parameters &args,
  const std::vector<int> &thread_seqs_idx,
  const std::vector<int> &thread_n_seqs,
  const std::vector<int> &thread_n_batchs,
  std::vector<gasal_gpu_storage_v> &gpu_storage_vecs
){
  const auto thread_id = omp_get_thread_num();
  auto curr_idx = thread_seqs_idx[thread_id];//number of sequences allocated to this thread
  int seqs_done = 0;

  if (thread_n_seqs.at(thread_id)<=0)
    return;

  #ifdef DEBUG
    std::cerr << "[TEST_PROG DEBUG]: ";
    std::cerr << "Number of gpu_batch in gpu_batch_arr : " << gpu_storage_vecs[thread_id].size() << std::endl;
    std::cerr << "[TEST_PROG DEBUG]: ";
    std::cerr << "Number of gpu_storage_vecs in a gpu_batch : " << thread_id+1 << std::endl;
  #endif

  std::vector<gpu_batch> gpu_batch_arr(gpu_storage_vecs[thread_id].size());

  for(size_t z = 0; z < gpu_batch_arr.size(); z++) {
    gpu_batch_arr[z].gpu_storage = &(gpu_storage_vecs[thread_id].at(z));
  }

  int n_batchs_done = 0;
  while (n_batchs_done < thread_n_batchs.at(thread_id)) { // Loop on streams
    int gpu_batch_arr_idx = 0;
    //------------checking the availability of a "free" stream"-----------------
    while(gpu_batch_arr_idx < gpu_storage_vecs[thread_id].size() && (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->is_free != 1) {
      gpu_batch_arr_idx++;
    }

    auto &this_storage = *gpu_batch_arr[gpu_batch_arr_idx].gpu_storage;

    if (seqs_done < thread_n_seqs.at(thread_id) && gpu_batch_arr_idx < gpu_storage_vecs[thread_id].size()) {
      uint32_t query_batch_idx = 0;
      uint32_t target_batch_idx = 0;
      unsigned int j = 0;
      //-----------Create a batch of sequences to be aligned on the GPU. The batch contains (target_seqs.size() / NB_STREAMS) number of sequences-----------------------

      for (int i = curr_idx; seqs_done < thread_n_seqs.at(thread_id) && j < (STREAM_BATCH_SIZE); i++, j++, seqs_done++){
        this_storage.current_n_alns++ ;

        if(this_storage.current_n_alns > this_storage.host_max_n_alns){
          gasal_host_alns_resize(this_storage, this_storage.host_max_n_alns * 2, args);
        }

        this_storage.host_query_batch_offsets[j] = query_batch_idx;
        this_storage.host_target_batch_offsets[j] = target_batch_idx;

        /*
          All the filling is moved on the library size, to take care of the memory size and expansions (when needed).
          The function gasal_host_batch_fill takes care of how to fill, how much to pad with 'N', and how to deal with memory.
          It's the same function for query and target, and you only need to set the final flag to either ; this avoides code duplication.
          The way the host memory is filled changes the current _idx (it's increased by size, and by the padding). That's why it's returned by the function.
        */

        query_batch_idx = gasal_host_batch_fill(this_storage,
                query_batch_idx,
                input_data.a.sequences.at(i).c_str(),
                input_data.a.sequences.at(i).size(),
                DataSource::QUERY);

        target_batch_idx = gasal_host_batch_fill(this_storage,
                target_batch_idx,
                input_data.b.sequences.at(i).c_str(),
                input_data.b.sequences.at(i).size(),
                DataSource::TARGET);

        this_storage.host_query_batch_lens[j] = input_data.a.sequences.at(i).size();
        this_storage.host_target_batch_lens[j] = input_data.b.sequences.at(i).size();
      }

      #ifdef DEBUG
        std::cerr << "[TEST_PROG DEBUG]: ";
        std::cerr << "Stream " << gpu_batch_arr_idx << ": j = " << j << ", seqs_done = " << seqs_done <<", query_batch_idx=" << query_batch_idx << " , target_batch_idx=" << target_batch_idx << std::endl;
      #endif

      // Here, we fill the operations arrays for the current batch to be processed by the stream
      gasal_op_fill(this_storage.host_query_op, input_data.a.modifiers.data() + seqs_done - j, j);
      gasal_op_fill(this_storage.host_target_op, input_data.b.modifiers.data() + seqs_done - j, j);

      gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch = j;
      uint32_t query_batch_bytes = query_batch_idx;
      uint32_t target_batch_bytes = target_batch_idx;
      gpu_batch_arr[gpu_batch_arr_idx].batch_start = curr_idx;
      curr_idx += (STREAM_BATCH_SIZE);

      gasal_aln_async(this_storage, query_batch_bytes, target_batch_bytes, gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch, args);
      this_storage.current_n_alns = 0;
    }

    //Find and print completed streams
    for (size_t i=0; i < gpu_storage_vecs[thread_id].size(); i++) {
      if (gasal_is_aln_async_done(*gpu_batch_arr[i].gpu_storage) != AlignmentStatus::Finished)
        continue;
      if(args.print_out){
        print_batch(input_data, gpu_batch_arr[i], args);
      }
      n_batchs_done++;
    }
  }
}



int main(int argc, char **argv) {
  //gasal_set_device(GPU_SELECT); //TODO

  Timer total_time;
  total_time.start();

  CLI::App app("GASAL2\n\n(Single-pack multi-Parameters (e.g. -sp) is not supported.)");
  Parameters args;
  std::string query_file;
  std::string reference_file;
  app.add_option("query_file", args.query_batch_fasta_filename,  "Query filename"       )->required();
  app.add_option("ref_file",   args.target_batch_fasta_filename, "Reference filename"   )->required();
  app.add_option("-a",         args.match_score,                 "Match score"          )->default_val(1);
  app.add_option("-b",         args.mismatch_score,              "Mismatch score"       )->default_val(4);
  app.add_option("-q",         args.gap_open_score,              "Gap open penalty"     )->default_val(6);
  app.add_option("-r",         args.gap_ext_score,               "Gap extension penalty")->default_val(1);
  app.add_option("-n",         args.n_threads,                   "Number of threads"    )->default_val(1);  app.add_flag_function("-s", [&](int){ args.start_pos = CompStart::WITH_START; }, "Find the start position");
  app.add_flag(  "-p",         args.print_out,                   "Print the alignment results");
  app.add_flag_function("-t", [&](int){ args.start_pos = CompStart::WITH_TB;    }, "Comptue traceback. With this option enabled, '-s' has no effect as start position will always be computed with traceback.");


  const std::map<std::string, algo_type> algo_map {
    {"local",       algo_type::LOCAL      },
    {"semi_global", algo_type::SEMI_GLOBAL},
    {"ksw",         algo_type::KSW        },
    {"banded",      algo_type::BANDED     }
  };
  app.add_option("-y", args.algo, "Alignment type. Must be local, semi_global, global, or ksw")->transform(CLI::CheckedTransformer(algo_map, CLI::ignore_case));

  const std::map<std::string, DataSource> ds_map {
    {"none",   DataSource::NONE  },
    {"query",  DataSource::QUERY },
    {"target", DataSource::TARGET},
    {"both",   DataSource::BOTH  }
  };
  app.add_option("-x,--head", args.semiglobal_skipping_head, "Specifies, for semi-global alignment, what should be skipped for heads and tails of the sequences. (NONE, QUERY, TARGET, BOTH)")->transform(CLI::CheckedTransformer(ds_map, CLI::ignore_case));
  app.add_option("-z,--tail", args.semiglobal_skipping_tail, "Specifies, for semi-global alignment, what should be skipped for heads and tails of the sequences. (NONE, QUERY, TARGET, BOTH)")->transform(CLI::CheckedTransformer(ds_map, CLI::ignore_case));

  app.add_option("-k",            args.k_band,     "Band width in case 'banded' is selected.");
  app.add_option("--second-best", args.secondBest, "Displays second best score (WITHOUT_START only)");


  CLI11_PARSE(app, argc, argv);

  args.print();

  //--------------copy substitution scores to GPU--------------------
  gasal_subst_scores sub_scores;
  sub_scores.match      = args.match_score;
  sub_scores.mismatch   = args.mismatch_score;
  sub_scores.gap_open   = args.gap_open_score;
  sub_scores.gap_extend = args.gap_ext_score;

  gasal_copy_subst_scores(sub_scores);

  //Read input data
  Timer timer_io;
  timer_io.start();
  const auto input_data = ReadFastaQueryTargetPair(
    args.query_batch_fasta_filename,
    args.target_batch_fasta_filename
  );
  timer_io.stop();
  std::cerr<<"IO time = "<<timer_io.getSeconds()<<" s"<<std::endl;

  const auto maximum_sequence_length = std::max(input_data.a.maximum_sequence_length, input_data.b.maximum_sequence_length);
  const auto total_seqs = input_data.a.headers.size();

  std::cerr<<"Loaded sequences = "<<input_data.sequence_count()<<std::endl;
  std::cerr<<"Max sequence length = "<<maximum_sequence_length<<std::endl;

  std::vector<int> thread_seqs_idx;
  std::vector<int> thread_n_seqs;
  std::vector<int> thread_n_batchs;

  const size_t thread_batch_size = (int)std::ceil((double)total_seqs/args.n_threads);
  size_t n_seqs_alloc = 0;
  for (int i = 0; i < args.n_threads; i++){//distribute the sequences among the threads equally
    thread_seqs_idx.push_back(n_seqs_alloc);
    if (n_seqs_alloc + thread_batch_size < total_seqs){
      thread_n_seqs.push_back(thread_batch_size);
    } else {
      thread_n_seqs.push_back(total_seqs - n_seqs_alloc);
    }
    thread_n_batchs.push_back( (int)std::ceil((double)thread_n_seqs[i]/(STREAM_BATCH_SIZE)) );
    n_seqs_alloc += thread_n_seqs[i];
  }

  std::cerr << "Processing..." << std::endl;

  omp_set_num_threads(args.n_threads);
  std::vector<gasal_gpu_storage_v> gpu_storage_vecs;
  for (int z = 0; z < args.n_threads; z++) {
    gpu_storage_vecs.emplace_back(NB_STREAMS);// creating NB_STREAMS streams per thread

    /*
      About memory sizes:
      The required memory is the total size of the batch + its padding, divided by the number of streams.
      The worst case would be that every sequence has to be padded with 7 'N', since they must have a length multiple of 8.
      Even though the memory can be dynamically expanded both for Host and Device, it is advised to start with a memory large enough so that these expansions rarely occur (for better performance.)
      Modifying the factor '1' in front of each size lets you see how GASAL2 expands the memory when needed.
    */
    /*
    // For exemple, this is exactly the memory needed to allocate to fit all sequences is a single GPU BATCH.
    gasal_init_streams(&(gpu_storage_vecs[z]),
              1 * ceil((double)(query_seqs_len +7*total_seqs) / (double)(NB_STREAMS)) ,
              1 * ceil((double)(query_seqs_len +7*total_seqs) / (double)(NB_STREAMS)) ,
              1 * ceil((double)(query_seqs_len +7*total_seqs) / (double)(NB_STREAMS)) ,
              1 * ceil((double)(query_seqs_len +7*total_seqs) / (double)(NB_STREAMS))  ,
              ceil((double)target_seqs.size() / (double)(NB_STREAMS)), // maximum number of alignments is bigger on target than on query side.
              ceil((double)target_seqs.size() / (double)(NB_STREAMS)),
              args);
    */
    //initializing the streams by allocating the required CPU and GPU memory
    // note: the calculations of the detailed sizes to allocate could be done on the library side (to hide it from the user's perspective)
    gasal_init_streams(gpu_storage_vecs[z], (input_data.a.maximum_sequence_length + 7),
            (maximum_sequence_length + 7) ,
             STREAM_BATCH_SIZE, //device
             args);
  }
  #ifdef DEBUG
    std::cerr << "[TEST_PROG DEBUG]: ";
    std::cerr << "size of host_unpack_query is " << (input_data.a.total_sequence_bytes +7*total_seqs) / (NB_STREAMS) << std::endl ;
  #endif


  #pragma omp parallel
  per_thread_processing(input_data, args, thread_seqs_idx, thread_n_seqs, thread_n_batchs, gpu_storage_vecs);

  for (int z = 0; z < args.n_threads; z++) {
    gasal_destroy_streams(gpu_storage_vecs[z], args);
  }

  total_time.stop();

  std::cerr << "Total Cells = "<<input_data.total_cells_1_to_1()<<std::endl;
  std::cerr << "Wall-time   = "<<total_time.getSeconds()<<std::endl;
  std::cerr << "GCUPS       = "<<(input_data.total_cells_1_to_1()/total_time.getSeconds()/(1e9))<<std::endl;

  return 0;
}
