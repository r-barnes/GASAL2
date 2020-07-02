#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Timer {
 public:
  Timer() = default;
  Timer(const std::string &name) : name(name) {}

  void clear() {elapsed_time = std::chrono::milliseconds(0);}
  void start() {start_time = std::chrono::high_resolution_clock::now(); }
  void restart() { clear(); start(); }

  void stop() {
    const auto this_duration = std::chrono::high_resolution_clock::now() - start_time;
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(this_duration);
  }

  void print() const {
    std::cout << name << " = " << elapsed_time.count() << " msec"   << std::endl;
  }

  double getTime() const { return elapsed_time.count();}

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  const std::string name;
  std::chrono::milliseconds elapsed_time;
};
