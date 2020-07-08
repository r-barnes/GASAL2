#pragma once

#include <cstdint>

//TODO: This is a really scary way of defining the numbers since we're only
//choosing the bottom four bits but the letter range is larger. It seems to work
//out by convenience.
#define A_PAK ('A'&0x0F) //1
#define C_PAK ('C'&0x0F) //3
#define G_PAK ('G'&0x0F) //7
#define T_PAK ('T'&0x0F) //4
//#define N_PAK ('N'&0x0F)


__device__ uint32_t pack_2words(const uint32_t word1, const uint32_t word2){
  uint32_t packed = 0;
  packed |= ((word1 >>  0) & 0xF) << 28;
  packed |= ((word1 >>  8) & 0xF) << 24;
  packed |= ((word1 >> 16) & 0xF) << 20;
  packed |= ((word1 >> 24) & 0xF) << 16;
  packed |= ((word2 >>  0) & 0xF) << 12;
  packed |= ((word2 >>  8) & 0xF) <<  8;
  packed |= ((word2 >> 16) & 0xF) <<  4;
  packed |= ((word2 >> 24) & 0xF) <<  0;
  return packed;
}



__global__ void pack_data(
	const uint32_t *const unpacked,
	uint32_t *const packed,
	const uint64_t N
){
  // assert(N%8==0); //TODO

  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride    = gridDim.x * blockDim.x;

  //Grid stride loop in which each thread takes 2 items
  //See: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  for(int i=thread_id; i<N/2; i+=stride){
    const uint32_t word1 = unpacked[2*i  ]; //Load 4 bases of the query sequence from global memory
    const uint32_t word2 = unpacked[2*i+1]; //Load another 4 bases
    packed[i] = pack_2words(word1, word2);
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



//TODO: Clean up padding
// * - '>' translates to 0b00 (0) = Forward, natural
// * - '<' translates to 0b01 (1) = Reverse, natural
// * - '/' translates to 0b10 (2) = Forward, complemented
// * - '+' translates to 0b11 (3) = Reverse, complemented
//Kernel uses one thread per sequence to reverse and complement that sequence
//TODO: Figure out a way to use grid strided loops and memory coalescing to
//accelerate this
__global__ void	new_reversecomplement_kernel(
  uint32_t       *const packed_batch,
  const uint32_t *const batch_lengths,
  const uint32_t *const batch_offsets,
  const uint8_t  *const op,
  const uint32_t        batch_size
){
  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id >= batch_size) return; //More threads than sequences, so quit
  if (op[thread_id] == '>') return;    //If the sequence is forward natural, there's nothing to do

  //Get the starting index of the batch. `batch_offsets` is measured in ASCII
  //characters. We've packed bytes into nibbles and stored the data in
  //uint32_t types which hold 8 nibbles, so we divide by 8 to get the packed
  //address.
  const auto packed_batch_idx = batch_offsets[thread_id]/8;

  //Get the length of the sequence. As above, we divide by 8 to move from ASCII
  //to packed data. If the unpacked data wasn't a multiple of 8, then we'll
  //have added one additional uint32_t. TODO: Doesn't this mess up the offsets?
  auto seq_len = batch_lengths[thread_id];
  seq_len = (seq_len/8) + (seq_len%8!=0);

  //Alias `packed_batch` to make things easier below
  auto my_batch = &packed_batch[packed_batch_idx];

  //That's (query_batch_regs / 2) + 1 if it's odd, + 0 otherwise. Used for
  //reverse (we start a both ends, and finish at the center of the sequence).
  const auto batch_regs_to_swap = (seq_len/2) + (seq_len & 1);

  //TODO: This sequentializes threads with different operations
  if (op[thread_id]=='<' || op[thread_id]=='+'){ // reverse
    // Deal with N's: read last word, find how many N's, store that number as offset,
    // and pad with that many for the last. Since we operate on nibbles, we multiply
    // by four.
    const uint8_t nbr_N = 4*count_word_trailing_n(my_batch[seq_len-1]);

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
      const uint32_t rpac_1 = my_batch[i]; //load 8 packed bases from head
      const uint32_t rpac_2 = (my_batch[seq_len-2 - i] << (32-nbr_N)) | (my_batch[seq_len-1 - i] >> nbr_N);

      const uint32_t reverse_rpac_1 = reverse_word(rpac_1);
      const uint32_t reverse_rpac_2 = reverse_word(rpac_2);

      // last swap operated manually, because of its irregular size (32 - 4*nbr_N bits, hence 8 - nbr_N nibbles)

      const uint32_t to_queue_1 = (reverse_rpac_1 << nbr_N) | (my_batch[seq_len-1 - i] & ((1<<nbr_N) - 1));
      const uint32_t to_queue_2 = (my_batch[seq_len-2 - i] & (0xFFFFFFFF - ((1<<nbr_N) - 1))) | (reverse_rpac_1 >> (32-nbr_N));

      //printf("KERNEL DEBUG: rpac_1 Word before reverse: %x, after: %x, split into %x + %x \n", rpac_1, reverse_rpac_1, to_queue_2, to_queue_1 );
      //printf("KERNEL DEBUG: rpac_2 Word before reverse: %x, after: %x\n", rpac_2, reverse_rpac_2 );

      my_batch[i] = reverse_rpac_2;
      my_batch[seq_len-1 - i] = to_queue_1;
      if (i!=batch_regs_to_swap-1)
        my_batch[seq_len-2 - i] = to_queue_2;
    }
  }

  if (op[thread_id]=='/' || op[thread_id]=='+'){  // Complement
    for (uint32_t i = 0; i < seq_len; i++){    // Complement all words
      my_batch[i] = complement_word(my_batch[i]);
    }
  }
}