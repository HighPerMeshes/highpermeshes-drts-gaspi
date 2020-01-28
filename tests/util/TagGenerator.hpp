#ifndef TEST_TAGGENERATOR_HPP
#define TEST_TAGGENERATOR_HPP

#include <iostream>

class TagGenerator
{
  public:
    static int Next() { 
        static int tag { 0 };
        std::cout << tag << std::endl;
        return tag++; 
    }

};

#endif