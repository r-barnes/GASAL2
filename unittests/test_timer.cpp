#include "doctest.h"
#include <gasal2/Timer.h>

#include <chrono>
#include <thread>

TEST_CASE("Timer Start Stop"){
  Timer timer;
  timer.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  timer.stop();

  CHECK(std::abs(timer.getTime()-200)<10);
}
