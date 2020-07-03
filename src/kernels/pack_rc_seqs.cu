#include <gasal2/gasal_kernels.h>

#include <cassert>
#include <cstdint>

//TODO: This is a really scary way of defining the numbers since we're only
//choosing the bottom four bits but the letter range is larger. It seems to work
//out by convenience.
#define A_PAK ('A'&0x0F) //1
#define C_PAK ('C'&0x0F) //3
#define G_PAK ('G'&0x0F) //7
#define T_PAK ('T'&0x0F) //4
//#define N_PAK ('N'&0x0F)


__global__ void pack_data(
	const uint32_t *const unpacked,
	uint32_t *const packed,
	const uint64_t N
){
  assert(N%8==0);

  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride    = gridDim.x * blockDim.x;

  //Grid stride loop in which each thread takes 2 items
  //See: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  for(int i=thread_id; i<N/2; i+=stride){
    const uint32_t reg1 = unpacked[2*i  ]; //Load 4 bases of the query sequence from global memory
    const uint32_t reg2 = unpacked[2*i+1]; //Load another 4 bases
    uint32_t packed_reg = 0;
    packed_reg |= ((reg1 >>  0) & 0xF) << 28; // ----
    packed_reg |= ((reg1 >>  8) & 0xF) << 24; //    |
    packed_reg |= ((reg1 >> 16) & 0xF) << 20; //    |
    packed_reg |= ((reg1 >> 24) & 0xF) << 16; //    |
    packed_reg |= ((reg2 >>  0) & 0xF) << 12; //     > pack sequence
    packed_reg |= ((reg2 >>  8) & 0xF) <<  8; //    |
    packed_reg |= ((reg2 >> 16) & 0xF) <<  4; //    |
    packed_reg |= ((reg2 >> 24) & 0xF) <<  0; //-----
    packed[i] = packed_reg;
	}
}



__host__ __device__ uint32_t complement_word(const uint32_t packed_word){
  uint32_t comp_word = 0;
  #pragma unroll 8
  for(int k = 28; k >= 0; k = k - 4){ // complement 32-bits word... is pragma-unrolled.
    auto nucleotide = (packed_word>>k) & 0xF;
    switch(nucleotide){
      case A_PAK: nucleotide = T_PAK; break;
      case C_PAK: nucleotide = G_PAK; break;
      case T_PAK: nucleotide = A_PAK; break;
      case G_PAK: nucleotide = C_PAK; break;
      default: break;
    }
    comp_word |= (nucleotide << k);
  }

  return comp_word;
}



__host__ __device__ uint32_t reverse_word(uint32_t word){
  //Our strategy for reversing a word is to use a logarithmic swap, like so
  //Initial: AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH
  //   ->    BBBBAAAADDDDCCCCFFFFEEEEHHHHGGGG
  //   ->    DDDDCCCCBBBBAAAAHHHHGGGGFFFFEEEE
  //   ->    HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
  //Note: this is more memory intensive than a simple loop because of the
  //constants, so if reversal is slow testing a loop might be a good idea.
  word = ((word&0xF0F0F0F0)>> 4) | ((word&0x0F0F0F0F)<< 4); //Swap positions of adjacent nibbles
  word = ((word&0xFF00FF00)>> 8) | ((word&0x00FF00FF)<< 8); //Swap positions of adjacent bytes
  word = ((word&0xFFFF0000)>>16) | ((word&0x0000FFFF)<<16); //Swap positions of shorts
  return word;
}



__host__ __device__ uint8_t count_word_trailing_n(uint32_t word){
  uint8_t number_of_n = 0;  //Number of Ns is initially 0
  for(int j=0;j<8;j++){     //Consider each nibble of the word
    if((word&0xF)==N_VALUE) //If this nibble is an N
      number_of_n++;        //count it
    else                    //otherwise
      break;                //stop counting, we've found all the trailing Ns
    word >>= 4;             //Shift one nibble to the right so we consider the next
  }
  return number_of_n;       //This many Ns
}



__global__ void	gasal_reversecomplement_kernel(
	uint32_t       *const packed_query_batch,
	uint32_t       *const packed_target_batch,
	const uint32_t *const query_batch_lens,
	const uint32_t *const target_batch_lens,
	const uint32_t *const query_batch_offsets,
	const uint32_t *const target_batch_offsets,
	const uint8_t  *const query_op,
	const uint8_t  *const target_op,
	const uint32_t        n_tasks
){
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid >= n_tasks) return;
	if (query_op[tid] == '>' && target_op[tid] == '>') return;		// if there's nothing to do (op=0, meaning sequence is Forward Natural), just exit the kernel ASAP.

	const uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3;//starting index of the target_batch sequence
	const uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	const uint32_t read_len = query_batch_lens[tid];
	const uint32_t ref_len = target_batch_lens[tid];
	const uint32_t query_batch_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding sequence of query_batch
	const uint32_t target_batch_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding sequence of target_batch

	const uint32_t query_batch_regs_to_swap = (query_batch_regs >> 1) + (query_batch_regs & 1); // that's (query_batch_regs / 2) + 1 if it's odd, + 0 otherwise. Used for reverse (we start a both ends, and finish at the center of the sequence)
	const uint32_t target_batch_regs_to_swap = (target_batch_regs >> 1) + (target_batch_regs & 1); // that's (target_batch_regs / 2) + 1 if it's odd, + 0 otherwise. Used for reverse (we start a both ends, and finish at the center of the sequence)

	// variables used dependent on target and query:

	const uint8_t  *op                 = nullptr;
	uint32_t       *packed_batch       = nullptr;
	const uint32_t *batch_regs         = nullptr;
	const uint32_t *batch_regs_to_swap = nullptr;
	const uint32_t *packed_batch_idx   = nullptr;

	// avoid useless code duplicate thanks to pointers to route the data flow where it should be, twice.
	// The kernel is already generic. Later on this can be used to split the kernel into two using templates...
	#pragma unroll 2
	for (int p = QUERY; p <= TARGET; p++)
	{
		switch(p)
		{
			case QUERY:
				op = query_op;
				packed_batch = packed_query_batch;
				batch_regs = &query_batch_regs;
				batch_regs_to_swap = &query_batch_regs_to_swap;
				packed_batch_idx = &packed_query_batch_idx;
				break;
			case TARGET:
				op = target_op;
				packed_batch = packed_target_batch;
				batch_regs = &target_batch_regs;
				batch_regs_to_swap = &target_batch_regs_to_swap;
				packed_batch_idx = &packed_target_batch_idx;
				break;
			default:
			break;
		}

		if (*(op + tid)=='<' || *(op + tid)=='+') // reverse
		{
			// deal with N's : read last word, find how many N's, store that number as offset, and pad with that many for the last
			uint8_t nbr_N = 0;
			for (int j = 0; j < 32; j = j + 4)
			{
				nbr_N += (((*(packed_batch + *(packed_batch_idx) + *(batch_regs)-1) & (0x0F << j)) >> j) == N_CODE);
			}

			//printf("KERNEL_DEBUG: nbr_N=%d\n", nbr_N);

			nbr_N = nbr_N << 2; // we operate on nibbles so we will need to do our shifts 4 bits by 4 bits, so 4*nbr_N

			for (uint32_t i = 0; i < *(batch_regs_to_swap); i++) // reverse all words. There's a catch with the last word (in the middle of the sequence), see final if.
			{
				/* This  is the current operation flow:\
					- Read the first 32-bits word on HEAD
					- Combine the reads of 2 last 32-bits words on tail to create the 32-bits word WITHOUT N's
					- Swap them
					- Write them at the correct places. Remember we're building 32-bits words across two 32-bits words on tail.
					So we have to take care of which bits are to be written on tail, too.

				You progress through both heads and tails that way, until you reach the center of the sequence.
				When you reach it, you actually don't write one of the words to avoid overwrite.
				*/
				const uint32_t rpac_1 = *(packed_batch + *(packed_batch_idx) + i); //load 8 packed bases from head
				const uint32_t rpac_2 = ((*(packed_batch + *(packed_batch_idx) + *(batch_regs)-2 - i)) << (32-nbr_N)) | ((*(packed_batch + *(packed_batch_idx) + *(batch_regs)-1 - i)) >> nbr_N);

				uint32_t reverse_rpac_1 = 0;
				uint32_t reverse_rpac_2 = 0;

				#pragma unroll 8
				for(int k = 28; k >= 0; k = k - 4)		// reverse 32-bits word... is pragma-unrolled.
				{
					reverse_rpac_1 |= ((rpac_1 & (0x0F << k)) >> (k)) << (28-k);
					reverse_rpac_2 |= ((rpac_2 & (0x0F << k)) >> (k)) << (28-k);
				}
				// last swap operated manually, because of its irregular size (32 - 4*nbr_N bits, hence 8 - nbr_N nibbles)

				const uint32_t to_queue_1 = (reverse_rpac_1 << nbr_N) | ((*(packed_batch + *(packed_batch_idx) + *(batch_regs)-1 - i)) & ((1<<nbr_N) - 1));
				const uint32_t to_queue_2 = ((*(packed_batch + *(packed_batch_idx) + *(batch_regs)-2 - i)) & (0xFFFFFFFF - ((1<<nbr_N) - 1))) | (reverse_rpac_1 >> (32-nbr_N));

				//printf("KERNEL DEBUG: rpac_1 Word before reverse: %x, after: %x, split into %x + %x \n", rpac_1, reverse_rpac_1, to_queue_2, to_queue_1 );
				//printf("KERNEL DEBUG: rpac_2 Word before reverse: %x, after: %x\n", rpac_2, reverse_rpac_2 );

				*(packed_batch + *(packed_batch_idx) + i) = reverse_rpac_2;
				(*(packed_batch + *(packed_batch_idx) + *(batch_regs)-1 - i)) = to_queue_1;
				if (i!=*(batch_regs_to_swap)-1)
					(*(packed_batch + *(packed_batch_idx) + *(batch_regs)-2 - i)) = to_queue_2;
			}
		}

    if (*(op+tid)=='/' || *(op+tid)=='+'){ // complement
      for (uint32_t i = 0; i < *(batch_regs); i++){ // reverse all words. There's a catch with the last word (in the middle of the sequence), see final if.
        *(packed_batch + *(packed_batch_idx) + i) = complement_word(*(packed_batch + *(packed_batch_idx) + i)); //load 8 packed bases from head
      }
    }
  }
}


















__global__ void	new_reversecomplement_kernel(
  uint32_t       *const packed_batch,
  const uint32_t *const batch_lengths,
  const uint32_t *const batch_offsets,
  const uint8_t  *const op,
  const uint32_t        n_tasks
){
  const auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid >= n_tasks) return;
  // If there's nothing to do (op=0, meaning sequence is Forward Natural), just exit the kernel ASAP.
  if (op[tid] == 0) return;

  const auto packed_batch_idx = batch_offsets[tid]/8;  //Starting index of the query_batch sequence
  const auto read_len         = batch_lengths[tid];    //Number of 32-bit words holding sequence of query_batch
  const auto batch_regs       = (read_len/8) + (read_len&7 ? 1 : 0);

  //That's (query_batch_regs / 2) + 1 if it's odd, + 0 otherwise. Used for
  //reverse (we start a both ends, and finish at the center of the sequence).
  const auto batch_regs_to_swap = (batch_regs/2) + (batch_regs & 1);

  if (op[tid] & 0x01){ // reverse
    // Deal with N's: read last word, find how many N's, store that number as offset,
    // and pad with that many for the last. Since we operate on nibbles, we multiply
    // by four.
    const uint8_t nbr_N = 4*count_word_trailing_n(packed_batch[packed_batch_idx + batch_regs-1]);

    for (uint32_t i = 0; i < batch_regs_to_swap; i++){ // reverse all words. There's a catch with the last word (in the middle of the sequence), see final if.
      /* This is the current operation flow:\
        - Read the first 32-bits word on HEAD
        - Combine the reads of 2 last 32-bits words on tail to create the 32-bits word WITHOUT N's
        - Swap them
        - Write them at the correct places. Remember we're building 32-bits words across two 32-bits words on tail.
        So we have to take care of which bits are to be written on tail, too.

      You progress through both heads and tails that way, until you reach the center of the sequence.
      When you reach it, you actually don't write one of the words to avoid overwrite.
      */
      const uint32_t rpac_1 = packed_batch[packed_batch_idx+i]; //load 8 packed bases from head
      const uint32_t rpac_2 = (packed_batch[packed_batch_idx + batch_regs-2 - i] << (32-nbr_N)) | (packed_batch[packed_batch_idx + batch_regs-1 - i] >> nbr_N);

      const uint32_t reverse_rpac_1 = reverse_word(rpac_1);
      const uint32_t reverse_rpac_2 = reverse_word(rpac_2);

      // last swap operated manually, because of its irregular size (32 - 4*nbr_N bits, hence 8 - nbr_N nibbles)

      const uint32_t to_queue_1 = (reverse_rpac_1 << nbr_N) | (packed_batch[packed_batch_idx + batch_regs-1 - i] & ((1<<nbr_N) - 1));
      const uint32_t to_queue_2 = (packed_batch[packed_batch_idx + batch_regs-2 - i] & (0xFFFFFFFF - ((1<<nbr_N) - 1))) | (reverse_rpac_1 >> (32-nbr_N));

      //printf("KERNEL DEBUG: rpac_1 Word before reverse: %x, after: %x, split into %x + %x \n", rpac_1, reverse_rpac_1, to_queue_2, to_queue_1 );
      //printf("KERNEL DEBUG: rpac_2 Word before reverse: %x, after: %x\n", rpac_2, reverse_rpac_2 );

      packed_batch[packed_batch_idx + i] = reverse_rpac_2;
      packed_batch[packed_batch_idx + batch_regs-1 - i] = to_queue_1;
      if (i!=batch_regs_to_swap-1)
        packed_batch[packed_batch_idx + batch_regs-2 - i] = to_queue_2;
    }
  }

  //TODO: Turn into a grid strided loop
  if (op[tid] & 0x02){ // Complement
    for (uint32_t i = 0; i < batch_regs; i++){ // Complement all words
      packed_batch[packed_batch_idx + i] = complement_word(packed_batch[packed_batch_idx + i]); //load 8 packed bases from head
    }
  }
}