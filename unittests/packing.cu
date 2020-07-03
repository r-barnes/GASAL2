#include "doctest.h"
#include <kernels/pack_rc_seqs.cuh>

#include <string>
#include <unordered_map>
#include <vector>

/*
a=['A','G','C','T']
import random
''.join(random.choices(a,k=8))
*/

#define CHECKCUDAERROR(error) \
    {\
      const auto err=(error); \
      if (err!=cudaSuccess) { \
        fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(err), err,  __LINE__, __FILE__); \
      }\
      REQUIRE(err==cudaSuccess); \
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

  const std::unordered_map<int,char> trans{{1,'A'}, {3, 'C'}, {7, 'G'}, {4, 'T'}, {0, '-'}};

  std::string packed_result;
  for(const auto &x: packed_query){
    packed_result.push_back(trans.at((x>>28)&0xF));
    packed_result.push_back(trans.at((x>>24)&0xF));
    packed_result.push_back(trans.at((x>>20)&0xF));
    packed_result.push_back(trans.at((x>>16)&0xF));
    packed_result.push_back(trans.at((x>>12)&0xF));
    packed_result.push_back(trans.at((x>> 8)&0xF));
    packed_result.push_back(trans.at((x>> 4)&0xF));
    packed_result.push_back(trans.at((x>> 0)&0xF));
  }

  CHECK(unpacked_query==packed_result);
}



TEST_CASE("Complement word"){
  uint32_t packed = 0;
  packed |= (('G'&0xF)<<28);
  packed |= (('C'&0xF)<<24);
  packed |= (('T'&0xF)<<20);
  packed |= (('T'&0xF)<<16);
  packed |= (('G'&0xF)<<12);
  packed |= (('T'&0xF)<< 8);
  packed |= (('A'&0xF)<< 4);
  packed |= (('A'&0xF)<< 0);

  const auto complement = complement_word(packed);

  CHECK( ((complement>>28)&0xF) == ('C' & 0xF));
  CHECK( ((complement>>24)&0xF) == ('G' & 0xF));
  CHECK( ((complement>>20)&0xF) == ('A' & 0xF));
  CHECK( ((complement>>16)&0xF) == ('A' & 0xF));
  CHECK( ((complement>>12)&0xF) == ('C' & 0xF));
  CHECK( ((complement>> 8)&0xF) == ('A' & 0xF));
  CHECK( ((complement>> 4)&0xF) == ('T' & 0xF));
  CHECK( ((complement>> 0)&0xF) == ('T' & 0xF));
}



TEST_CASE("Reverse word"){
  uint32_t packed = 0;
  packed |= (('A'&0xF)<<28);
  packed |= (('B'&0xF)<<24);
  packed |= (('C'&0xF)<<20);
  packed |= (('D'&0xF)<<16);
  packed |= (('E'&0xF)<<12);
  packed |= (('F'&0xF)<< 8);
  packed |= (('G'&0xF)<< 4);
  packed |= (('H'&0xF)<< 0);

  const auto reversed = reverse_word(packed);

  CHECK( ((reversed>>28)&0xF) == ('H' & 0xF));
  CHECK( ((reversed>>24)&0xF) == ('G' & 0xF));
  CHECK( ((reversed>>20)&0xF) == ('F' & 0xF));
  CHECK( ((reversed>>16)&0xF) == ('E' & 0xF));
  CHECK( ((reversed>>12)&0xF) == ('D' & 0xF));
  CHECK( ((reversed>> 8)&0xF) == ('C' & 0xF));
  CHECK( ((reversed>> 4)&0xF) == ('B' & 0xF));
  CHECK( ((reversed>> 0)&0xF) == ('A' & 0xF));
}