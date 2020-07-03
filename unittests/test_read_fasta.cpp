#include "doctest.h"
#include <gasal2/read_fasta.h>

#include <stdexcept>

TEST_CASE("No fasta file"){
  CHECK_THROWS_AS(ReadFasta("not-a-file"), std::runtime_error);
}

//TODO: Test for modifiers reading correctly
TEST_CASE("Read Fasta"){
  FastaInput fasta;
  CHECK_NOTHROW(fasta=ReadFasta("test.fasta"));

  CHECK(fasta.sequences.size()==200);
  CHECK(fasta.modifiers.size()==200);
  CHECK(fasta.headers.size()==200);
  CHECK(fasta.maximum_sequence_length==150);
  CHECK(fasta.sequences.at(5)=="TTGAGACCAGCTTGGGCAACATAGCGAGACACCGTCTCTCCAAAAAAATAACAAATAGTGGGGCGTGATGGCGCGCTCCTGTAGTCTCAGCTACTTGGGCGGTCGCGATGGGAGGATCGATCGAGTCTGGGAGGTCGAGGCTGCAGTGAG");
}

TEST_CASE("Read Pair"){
  CHECK_NOTHROW(ReadFastaQueryTargetPair("test.fasta", "test.fasta"));
}