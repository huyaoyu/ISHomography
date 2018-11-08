#ifndef __COMMON_COMMON_HPP__
#define __COMMON_COMMON_HPP__

#include <exception>
#include <boost/exception/all.hpp>

struct exception_base : virtual std::exception, virtual boost::exception { };

typedef boost::error_info<struct tag_info_string, std::string> ExceptionInfoString;

typedef double real;

#endif // __COMMON_COMMON_HPP__