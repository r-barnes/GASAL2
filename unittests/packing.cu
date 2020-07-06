#include "doctest.h"
#include <gasal2/gasal_align.h>

#include <thrust/device_vector.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

__global__ void pack_data(
	const uint32_t *const unpacked,
	uint32_t *const packed,
	const uint64_t N
);

__global__ void	new_reversecomplement_kernel(
  uint32_t       *const packed_batch,
  const uint32_t *const batch_lengths,
  const uint32_t *const batch_offsets,
  const uint8_t  *const op,
  const uint32_t        batch_size
);

__host__ __device__ uint32_t complement_word(const uint32_t packed_word);
__host__ __device__ uint32_t reverse_word(uint32_t word);
__host__ __device__ uint8_t count_word_trailing_n(uint32_t word);
__global__ void test_DEV_GET_SUB_SCORE_LOCAL(bool *d_good);


/*
a=['A','G','C','T']
import random
''.join(random.choices(a,k=8))
*/

const std::unordered_map<int,char> trans{{1,'A'}, {3, 'C'}, {7, 'G'}, {4, 'T'}, {14, 'N'}, {0, '-'}};
const std::string bases = "GCTAN";



std::string Unloadword(uint32_t word){
  std::string unpacked;
  unpacked += trans.at((word>>28)&0xF);
  unpacked += trans.at((word>>24)&0xF);
  unpacked += trans.at((word>>20)&0xF);
  unpacked += trans.at((word>>16)&0xF);
  unpacked += trans.at((word>>12)&0xF);
  unpacked += trans.at((word>> 8)&0xF);
  unpacked += trans.at((word>> 4)&0xF);
  unpacked += trans.at((word>> 0)&0xF);
  return unpacked;
}



uint32_t LoadWord(const std::string &nibbles){
  REQUIRE(nibbles.size()==8);
  uint32_t packed = 0;
  packed |= ((nibbles[0]&0xF)<<28);
  packed |= ((nibbles[1]&0xF)<<24);
  packed |= ((nibbles[2]&0xF)<<20);
  packed |= ((nibbles[3]&0xF)<<16);
  packed |= ((nibbles[4]&0xF)<<12);
  packed |= ((nibbles[5]&0xF)<< 8);
  packed |= ((nibbles[6]&0xF)<< 4);
  packed |= ((nibbles[7]&0xF)<< 0);
  return packed;
}



std::string RandomWord(){
  std::string input;
  for(int i=0;i<8;i++)
    input += bases.at(rand()%bases.size());
  return input;
}



std::string RandomInput(){
  int length = rand()%2048;
  length -= length%8;
  CHECK(length%8==0);
  std::string ret;
  for(int i=0;i<length;i++)
    ret += bases.at(rand()%bases.size());
  return ret;
}



struct Batch {
  std::string unpacked_data;
  std::vector<uint32_t> lengths;
  std::vector<uint32_t> offsets;
  int size;
};

struct GPUBatch {
  thrust::device_vector<uint32_t> packed_data;
  thrust::device_vector<uint32_t> lengths;
  thrust::device_vector<uint32_t> offsets;
  int size;
};

Batch RandomBatch(const int size){
  Batch batch;
  batch.size = size;
  batch.offsets.push_back(0);
  for(int i=0;i<size;i++){
    const auto seq = RandomInput();
    batch.unpacked_data += seq;
    batch.lengths.push_back(seq.size());
    batch.offsets.push_back(batch.offsets.back()+seq.size());
  }
  batch.offsets.pop_back();
  return batch;
}



thrust::device_vector<uint32_t> pack_string_onto_device(const std::string &data){
  CHECK(data.size()%8==0);

  thrust::device_vector<char> unpacked(data.begin(), data.end());
  thrust::device_vector<uint32_t> packed(data.size()/4/2); //Two characters squeezed into the space of one

  const uint32_t BLOCKDIM = 128;
  const uint32_t N_BLOCKS = 30;

  pack_data<<<N_BLOCKS, BLOCKDIM>>>(
    (uint32_t*)thrust::raw_pointer_cast(unpacked.data()),
    thrust::raw_pointer_cast(packed.data()),
    data.size()/4 //Four characters per uint32_t
  );

  const auto err = cudaGetLastError();
  CHECK(err==cudaSuccess);

  CHECKCUDAERROR(cudaDeviceSynchronize());

  return packed;
}



GPUBatch gpuify_batch(const Batch &batch){
  GPUBatch gbatch;
  gbatch.packed_data = pack_string_onto_device(batch.unpacked_data);
  gbatch.lengths = batch.lengths;
  gbatch.offsets = batch.offsets;
  gbatch.size    = batch.size;
  return gbatch;
}



TEST_CASE("Packing"){
  for(int i=0;i<100;i++){
    const auto unpacked = RandomInput();
    const auto packed = pack_string_onto_device(unpacked);

    std::vector<uint32_t> host_packed(packed.size());

    CHECKCUDAERROR(cudaMemcpy(host_packed.data(), thrust::raw_pointer_cast(packed.data()), packed.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::string packed_result;
    for(const auto &x: host_packed)
      packed_result += Unloadword(x);

    CHECK(unpacked==packed_result);
  }
}



TEST_CASE("Complement"){
  const auto do_op_on_batch = [](GPUBatch &gbatch, const char op){
    const auto ops = thrust::device_vector<uint8_t>(gbatch.size, op);

    const uint32_t BLOCKDIM = 128;
    const uint32_t N_BLOCKS = 30;
    new_reversecomplement_kernel<<<N_BLOCKS, BLOCKDIM>>>(
      thrust::raw_pointer_cast(gbatch.packed_data.data()),
      thrust::raw_pointer_cast(gbatch.lengths.data()),
      thrust::raw_pointer_cast(gbatch.offsets.data()),
      thrust::raw_pointer_cast(ops.data()),
      gbatch.size
    );

    const auto err = cudaGetLastError();
    CHECK(err==cudaSuccess);

    CHECKCUDAERROR(cudaDeviceSynchronize());

    std::vector<uint32_t> host_packed(gbatch.packed_data.size());
    CHECKCUDAERROR(cudaMemcpy(host_packed.data(), thrust::raw_pointer_cast(gbatch.packed_data.data()), gbatch.packed_data.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::string unpacked_result;
    for(const auto &x: host_packed)
      unpacked_result += Unloadword(x);
    return unpacked_result;
  };

  // '>' = Forward, natural
  for(int i=0;i<30;i++){
    const auto batch = RandomBatch(200);
    auto gbatch = gpuify_batch(batch);
    const auto unpacked_result = do_op_on_batch(gbatch, '>');
    CHECK(batch.unpacked_data==unpacked_result);
  }

  // '/' = Forward, complemented
  for(int i=0;i<30;i++){
    const auto batch = RandomBatch(200);
    auto gbatch = gpuify_batch(batch);
    const auto complemented = do_op_on_batch(gbatch, '/');
    CHECK(batch.unpacked_data!=complemented);
    //Check that we can undo the complement
    const auto unpacked_result_undo = do_op_on_batch(gbatch, '/');
    CHECK(batch.unpacked_data==unpacked_result_undo);
    //Verify the complement by undoing the complement using simple methods
    for(int i=0;i<complemented.size();i+=8){
      const auto word_comp = Unloadword(complement_word(LoadWord(complemented.substr(i,8))));
      CHECK(word_comp==batch.unpacked_data.substr(i,8));
    }
  }

  // * - '<' = Reverse, natural
  //TODO: Enable this code after fixing up the reversal kernel
/*  for(int i=0;i<30;i++){
    const auto batch = RandomBatch(200);
    auto gbatch = gpuify_batch(batch);
    auto reversed = do_op_on_batch(gbatch, '<');
    CHECK(batch.unpacked_data!=reversed);
    //Check that we can undo the reverse
    const auto unpacked_result_undo = do_op_on_batch(gbatch, '/');
    CHECK(batch.unpacked_data==unpacked_result_undo);
    //Verify the reverse by undoing the reverse using simple methods
    for(int i=0;i<batch.size;i++){
      auto original_sub = batch.unpacked_data.substr(batch.offsets[i], batch.lengths[i]);
      const auto reversed_sub = reversed.substr(batch.offsets[i], batch.lengths[i]);

      int n_count = 0;
      for(auto i=original_sub.rbegin();i!=original_sub.rend() && *i=='N';i++,n_count++){}

      // std::cerr<<original_sub<<"\n"<<reversed_sub<<"\n\n";
      std::reverse(original_sub.begin(), original_sub.end()-n_count);
      CHECK(original_sub==reversed_sub);
    }*/

  //TODO: Add tests
  // '+' = Reverse, complemented


}





TEST_CASE("Complement word single"){
  uint32_t packed = 0;
  packed |= (('G'&0xF)<<28);
  packed |= (('C'&0xF)<<24);
  packed |= (('T'&0xF)<<20);
  packed |= (('T'&0xF)<<16);
  packed |= (('G'&0xF)<<12);
  packed |= (('T'&0xF)<< 8);
  packed |= (('N'&0xF)<< 4);
  packed |= (('A'&0xF)<< 0);

  const auto complement = complement_word(packed);

  CHECK( ((complement>>28)&0xF) == ('C' & 0xF));
  CHECK( ((complement>>24)&0xF) == ('G' & 0xF));
  CHECK( ((complement>>20)&0xF) == ('A' & 0xF));
  CHECK( ((complement>>16)&0xF) == ('A' & 0xF));
  CHECK( ((complement>>12)&0xF) == ('C' & 0xF));
  CHECK( ((complement>> 8)&0xF) == ('A' & 0xF));
  CHECK( ((complement>> 4)&0xF) == ('N' & 0xF));
  CHECK( ((complement>> 0)&0xF) == ('T' & 0xF));
}



TEST_CASE("Complement word randomized"){
  //Complementing the complement should give us the original
  for(int i=0;i<100;i++){
    const std::string input = RandomWord();
    const auto complemented = Unloadword(complement_word(complement_word(LoadWord(input))));
    CHECK(input==complemented);
  }
}



TEST_CASE("Reverse word"){
  //Test that reversing inside the system matches a known reverse
  for(int i=0;i<100;i++){
    const std::string input = RandomWord();
    auto reversed = Unloadword(reverse_word(LoadWord(input)));
    std::reverse(reversed.begin(), reversed.end());
    CHECK(input==reversed);
  }
}



TEST_CASE("Trailing Ns"){
  for(int i=0;i<100;i++){
    std::string input = RandomWord();

    int trailing = 0;
    for(int i=7;i>=0;i--){
      if(input.at(i)=='N')
        trailing++;
      else
        break;
    }

    const auto cuda_trailing = count_word_trailing_n(LoadWord(input));

    CHECK_MESSAGE(cuda_trailing==trailing, input);
  }
}



TEST_CASE("Score matching"){
  gasal_subst_scores sub_scores;
  sub_scores.match      = 1;
  sub_scores.mismatch   = 4;
  sub_scores.gap_open   = 6;
  sub_scores.gap_extend = 1;
  gasal_copy_subst_scores(sub_scores);

  bool h_good=false;
  bool *d_good;
  cudaMalloc(&d_good, sizeof(bool));
  test_DEV_GET_SUB_SCORE_LOCAL<<<1,1>>>(d_good);
  cudaMemcpy(&h_good, d_good, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(d_good);
  CHECK(h_good==true);
}