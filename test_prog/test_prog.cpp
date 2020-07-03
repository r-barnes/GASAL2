#include <gasal2/gasal_header.h>
#include <gasal2/read_fasta.h>

#include <iostream>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <gasal2/Timer.h>

#define NB_STREAMS 2

//#define STREAM_BATCH_SIZE (262144)
// this gives each stream HALF of the sequences.
//#define STREAM_BATCH_SIZE ceil((double)target_seqs.size() / (double)(2))

#define STREAM_BATCH_SIZE 5000//ceil((double)target_seqs.size() / (double)(2 * 2))


#define DEBUG

#define MAX(a,b) (a>b ? a : b)

//#define GPU_SELECT 0


int main(int argc, char **argv) {
  //gasal_set_device(GPU_SELECT);

  Parameters args(argc, argv);
  args.parse();
  args.print();

  //--------------copy substitution scores to GPU--------------------
  gasal_subst_scores sub_scores;
  sub_scores.match      = args.match_score;
  sub_scores.mismatch   = args.mismatch_score;
  sub_scores.gap_open   = args.gap_open_score;
  sub_scores.gap_extend = args.gap_ext_score;

  gasal_copy_subst_scores(sub_scores);

  //Read input data
  const auto input_data = ReadFastaQueryTargetPair(
    args.query_batch_fasta_filename,
    args.target_batch_fasta_filename
  );

  const auto maximum_sequence_length = std::max(input_data.first.maximum_sequence_length, input_data.second.maximum_sequence_length);
  const auto total_seqs = input_data.first.headers.size();

  #ifdef DEBUG
    std::cerr << "[TEST_PROG DEBUG]: ";
    std::cerr << "Size of read batches are: query=" << input_data.first.total_sequence_bytes << ", target=" << input_data.second.total_sequence_bytes << ". maximum_sequence_length=" << maximum_sequence_length << std::endl;
  #endif


  // transforming the _mod into a char* array (to be passed to GASAL, which deals with C types)
  auto *const target_seq_mod = new uint8_t [total_seqs];
  auto *const query_seq_mod  = new uint8_t [total_seqs];
  auto *const target_seq_id  = new uint32_t[total_seqs];
  auto *const query_seq_id   = new uint32_t[total_seqs];

  for (size_t i = 0; i < total_seqs; i++){
    query_seq_mod[i] = input_data.first.modifiers.at(i);
    query_seq_id[i] = i;

    target_seq_mod[i] = input_data.second.modifiers.at(i);
    target_seq_id[i] = i;
  }

  #ifdef DEBUG
    std::cerr << "[TEST_PROG DEBUG]: query, mod@id=";
    for (size_t i = 0; i < total_seqs; i++){
      if (query_seq_mod[i] > 0)
        std::cerr << +(query_seq_mod[i]) << "@" << query_seq_id[i] << "| ";
    }
    std::cerr << std::endl;
  #endif

  auto *const thread_seqs_idx  = new int[args.n_threads];
  auto *const thread_n_seqs    = new int[args.n_threads];
  auto *const thread_n_batchs  = new int[args.n_threads];
  auto *const thread_misc_time = new double[args.n_threads];

  size_t thread_batch_size = (int)ceil((double)total_seqs/args.n_threads);
  size_t n_seqs_alloc = 0;
  for (int i = 0; i < args.n_threads; i++){//distribute the sequences among the threads equally
    thread_seqs_idx[i] = n_seqs_alloc;
    if (n_seqs_alloc + thread_batch_size < total_seqs) thread_n_seqs[i] = thread_batch_size;
    else thread_n_seqs[i] = total_seqs - n_seqs_alloc;
    thread_n_batchs[i] = (int)ceil((double)thread_n_seqs[i]/(STREAM_BATCH_SIZE));
    n_seqs_alloc += thread_n_seqs[i];
  }

  std::cerr << "Processing..." << std::endl;

  Timer total_time;
  total_time.start();
  omp_set_num_threads(args.n_threads);
  auto *const gpu_storage_vecs = new gasal_gpu_storage_v[args.n_threads];
  for (int z = 0; z < args.n_threads; z++) {
    gpu_storage_vecs[z] = gasal_init_gpu_storage_v(NB_STREAMS);// creating NB_STREAMS streams per thread

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
    gasal_init_streams(&(gpu_storage_vecs[z]), (input_data.first.maximum_sequence_length + 7),
            (maximum_sequence_length + 7) ,
             STREAM_BATCH_SIZE, //device
             args);
  }
  #ifdef DEBUG
    std::cerr << "[TEST_PROG DEBUG]: ";
    std::cerr << "size of host_unpack_query is " << (input_data.first.total_sequence_bytes +7*total_seqs) / (NB_STREAMS) << std::endl ;
  #endif

  #pragma omp parallel
  {
  int n_seqs = thread_n_seqs[omp_get_thread_num()];//number of sequences allocated to this thread
  int curr_idx = thread_seqs_idx[omp_get_thread_num()];//number of sequences allocated to this thread
  int seqs_done = 0;
  int n_batchs_done = 0;

  struct gpu_batch{ //a struct to hold data structures of a stream
      gasal_gpu_storage_t *gpu_storage; //the struct that holds the GASAL2 data structures
      int n_seqs_batch;//number of sequences in the batch (<= (target_seqs.size() / NB_STREAMS))
      int batch_start;//starting index of batch
  };

  #ifdef DEBUG
    std::cerr << "[TEST_PROG DEBUG]: ";
    std::cerr << "Number of gpu_batch in gpu_batch_arr : " << gpu_storage_vecs[omp_get_thread_num()].n << std::endl;
    std::cerr << "[TEST_PROG DEBUG]: ";
    std::cerr << "Number of gpu_storage_vecs in a gpu_batch : " << omp_get_thread_num()+1 << std::endl;
  #endif

  std::vector<gpu_batch> gpu_batch_arr(gpu_storage_vecs[omp_get_thread_num()].n);

  for(size_t z = 0; z < gpu_batch_arr.size(); z++) {
    gpu_batch_arr[z].gpu_storage = &(gpu_storage_vecs[omp_get_thread_num()].a[z]);
  }

  if (n_seqs > 0) {
    while (n_batchs_done < thread_n_batchs[omp_get_thread_num()]) { // Loop on streams
      int gpu_batch_arr_idx = 0;
      //------------checking the availability of a "free" stream"-----------------
      while(gpu_batch_arr_idx < gpu_storage_vecs[omp_get_thread_num()].n && (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->is_free != 1) {
        gpu_batch_arr_idx++;
      }

      if (seqs_done < n_seqs && gpu_batch_arr_idx < gpu_storage_vecs[omp_get_thread_num()].n) {
        uint32_t query_batch_idx = 0;
        uint32_t target_batch_idx = 0;
        unsigned int j = 0;
        //-----------Create a batch of sequences to be aligned on the GPU. The batch contains (target_seqs.size() / NB_STREAMS) number of sequences-----------------------

        for (int i = curr_idx; seqs_done < n_seqs && j < (STREAM_BATCH_SIZE); i++, j++, seqs_done++)
        {

          gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns++ ;

          if(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns > gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->host_max_n_alns)
          {
            gasal_host_alns_resize(*gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->host_max_n_alns * 2, args);
          }

          (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_offsets[j] = query_batch_idx;
          (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_offsets[j] = target_batch_idx;

          /*
            All the filling is moved on the library size, to take care of the memory size and expansions (when needed).
            The function gasal_host_batch_fill takes care of how to fill, how much to pad with 'N', and how to deal with memory.
            It's the same function for query and target, and you only need to set the final flag to either ; this avoides code duplication.
            The way the host memory is filled changes the current _idx (it's increased by size, and by the padding). That's why it's returned by the function.
          */

          query_batch_idx = gasal_host_batch_fill(*gpu_batch_arr[gpu_batch_arr_idx].gpu_storage,
                  query_batch_idx,
                  input_data.first.sequences.at(i).c_str(),
                  input_data.first.sequences.at(i).size(),
                  QUERY);

          target_batch_idx = gasal_host_batch_fill(*gpu_batch_arr[gpu_batch_arr_idx].gpu_storage,
                  target_batch_idx,
                  input_data.second.sequences.at(i).c_str(),
                  input_data.second.sequences.at(i).size(),
                  TARGET);


          (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_lens[j] = input_data.first.sequences.at(i).size();
          (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_lens[j] = input_data.second.sequences.at(i).size();

        }

        #ifdef DEBUG
          std::cerr << "[TEST_PROG DEBUG]: ";
          std::cerr << "Stream " << gpu_batch_arr_idx << ": j = " << j << ", seqs_done = " << seqs_done <<", query_batch_idx=" << query_batch_idx << " , target_batch_idx=" << target_batch_idx << std::endl;
        #endif

        // Here, we fill the operations arrays for the current batch to be processed by the stream
        gasal_op_fill(*gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_seq_mod + seqs_done - j, j, QUERY);
        gasal_op_fill(*gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, target_seq_mod + seqs_done - j, j, TARGET);


        gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch = j;
        uint32_t query_batch_bytes = query_batch_idx;
        uint32_t target_batch_bytes = target_batch_idx;
        gpu_batch_arr[gpu_batch_arr_idx].batch_start = curr_idx;
        curr_idx += (STREAM_BATCH_SIZE);

        //----------------------------------------------------------------------------------------------------
        //-----------------calling the GASAL2 non-blocking alignment function---------------------------------

        gasal_aln_async(*gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_bytes, target_batch_bytes, gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch, args);
        gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns = 0;
        //---------------------------------------------------------------------------------
      }


      //-------------------------------print alignment results----------------------------------------

      gpu_batch_arr_idx = 0;
      while (gpu_batch_arr_idx < gpu_storage_vecs[omp_get_thread_num()].n) {//loop through all the streams and print the results
                                          //of the finished streams.
        if (gasal_is_aln_async_done(*gpu_batch_arr[gpu_batch_arr_idx].gpu_storage) == 0) {
          int j = 0;
          if(args.print_out) {
          #pragma omp critical
          for (int i = gpu_batch_arr[gpu_batch_arr_idx].batch_start; j < gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch; i++, j++) {

            std::cout << "query_name=" << input_data.first.headers.at(i);
            std::cout << "\ttarget_name=" << input_data.second.headers.at(i);
            std::cout << "\tscore=" << (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->aln_score[j] ;


            /// WARNING : INEQUALITY ON ENUM: CAN BREAK IF ENUM ORDER IS CHANGED
            if ((args.start_pos == WITH_START || args.start_pos == WITH_TB)
              && ((args.algo == SEMI_GLOBAL && (args.semiglobal_skipping_head != NONE || args.semiglobal_skipping_head != NONE))
                || args.algo > SEMI_GLOBAL))
            {
              std::cout << "\tquery_batch_start=" << (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->query_batch_start[j];
              std::cout << "\ttarget_batch_start=" << (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->target_batch_start[j];
            }

            if (args.algo != GLOBAL)
            {
              std::cout << "\tquery_batch_end="  << (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->query_batch_end[j];
              std::cout << "\ttarget_batch_end=" << (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->target_batch_end[j] ;
            }

            if (args.secondBest)
            {
              std::cout << "\t2nd_score=" << (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res_second->aln_score[j] ;
              std::cout << "\t2nd_query_batch_end="  << (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res_second->query_batch_end[j];
              std::cout << "\t2nd_target_batch_end=" << (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res_second->target_batch_end[j] ;
            }

            if (args.start_pos == WITH_TB) {
              std::cout << "\tCIGAR=";
              int offset = (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_offsets[j];
              int n_cigar_ops = (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->n_cigar_ops[j];
              int last_op = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + n_cigar_ops - 1]) & 3;
              int count = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + n_cigar_ops - 1]) >> 2;
              for (int u = n_cigar_ops - 2; u >= 0 ; u--){
                int curr_op = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) & 3;
                if (curr_op == last_op) {
                  count +=  ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) >> 2;
                } else {
                  char op;
                  switch (last_op) {
                    case 0: op = 'M';  break;
                    case 1: op = 'X';  break;
                    case 2: op = 'D';  break;
                    case 3: op = 'I';  break;
                    default: op = 'E'; break;
                  }
                  std::cout << count << op;
                  count =  ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) >> 2;
                }
                last_op = curr_op;
              }
              char op;
              switch (last_op) {
                case 0: op = 'M'; break;
                case 1: op = 'X'; break;
                case 2: op = 'D'; break;
                case 3: op = 'I'; break;
              }
              std::cout << count << op;
            }
            std::cout << std::endl;
          }
          }
          n_batchs_done++;
        }
        gpu_batch_arr_idx++;
      }
    }
  }


  }

  for (int z = 0; z < args.n_threads; z++) {
    gasal_destroy_streams(&(gpu_storage_vecs[z]), args);
    gasal_destroy_gpu_storage_v(&(gpu_storage_vecs[z]));
  }
  free(gpu_storage_vecs);
  total_time.stop();

  /*
  string algorithm = al_type;
  string start_type[2] = {"without_start", "with_start"};
  al_type += "_";
  al_type += start_type[start_pos==WITH_START];
  */

  double av_misc_time = 0.0;
  for (int i = 0; i < args.n_threads; ++i){
    av_misc_time += (thread_misc_time[i]/args.n_threads);
  }

  std::cerr << std::endl << "Done" << std::endl;
  fprintf(stderr, "Total execution time (in milliseconds): %.3f\n", total_time.getTime());

  return 0;
}
