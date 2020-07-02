#include "doctest.h"
#include <kernels/pack_rc_seqs.cuh>

#include <string>
#include <unordered_map>
#include <vector>

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