#include "doctest.h"
#include <kernels/pack_rc_seqs.cuh>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

/*
a=['A','G','C','T']
import random
''.join(random.choices(a,k=8))
*/

const std::unordered_map<int,char> trans{{1,'A'}, {3, 'C'}, {7, 'G'}, {4, 'T'}, {14, 'N'}, {0, '-'}};
const std::string bases = "GCTAN";

#define CHECKCUDAERROR(error) \
    {\
      const auto err=(error); \
      if (err!=cudaSuccess) { \
        fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(err), err,  __LINE__, __FILE__); \
      }\
      REQUIRE(err==cudaSuccess); \
    }



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



std::string RandomWord(){
  std::string input;
  for(int i=0;i<8;i++)
    input += bases.at(rand()%bases.size());
  return input;
}



TEST_CASE("Packing"){
  const std::string unpacked_query_seed = "GAACTGCCGAGAAGTCACAGAAGGGACTGTGG";
  std::string unpacked_query;
  for(int i=0;i<100;i++)
    unpacked_query += unpacked_query_seed;

  CHECK(unpacked_query.size()%8==0);

  const auto unpacked_size = unpacked_query.size()/4;
  const auto packed_size   = unpacked_size/2;

  uint32_t *unpacked_query_dev;
  CHECKCUDAERROR(cudaMalloc(&unpacked_query_dev, unpacked_size*sizeof(uint32_t)));
  CHECKCUDAERROR(cudaMemcpy(unpacked_query_dev, (uint32_t*)unpacked_query.data(), unpacked_size*sizeof(uint32_t), cudaMemcpyHostToDevice));

  uint32_t *packed_query_dev;
  CHECKCUDAERROR(cudaMalloc(&packed_query_dev, packed_size*sizeof(uint32_t)));

  const uint32_t BLOCKDIM = 128;
  const uint32_t N_BLOCKS = 30;

  pack_data<<<N_BLOCKS, BLOCKDIM>>>(
    unpacked_query_dev,
    packed_query_dev,
    unpacked_size
  );

  const auto err = cudaGetLastError();
  CHECK(err==cudaSuccess);

  CHECKCUDAERROR(cudaDeviceSynchronize());

  std::vector<uint32_t> packed_query (packed_size);

  CHECKCUDAERROR(cudaMemcpy(packed_query.data(),  packed_query_dev,  packed_size*sizeof(uint32_t), cudaMemcpyDeviceToHost));

  std::string packed_result;
  for(const auto &x: packed_query)
    packed_result += Unloadword(x);

  CHECK(unpacked_query==packed_result);
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
