#pragma once

#pragma warning(disable : 4996)

#include <iostream>
#include <iomanip>
#include <cctype>
#include <string>
#include <algorithm>
#include <ctime>
#include "termcolor.hpp"

class MyConsole
{
public:
    MyConsole(bool verbose_tag = true)
        : verbose(verbose_tag)
    {
    }
    void set_verbose(bool verbose_tag)
    {
        verbose = verbose_tag;
    }
    void set_width(int w)
    {
        fieldwidth = std::max(80, w);
    }
    void print_seperator()
    {
        if (!verbose)
            return;
        std::cout << termcolor::bright_white << std::setfill('-') << std::setw(fieldwidth) << termcolor::reset << std::endl;
    }
    void print_title(const std::string &title, char fillchar = '-')
    {
        if (!verbose)
            return;
        size_t fill_size = fieldwidth > title.size() + 1 ? fieldwidth - title.size() - 1 : 0;
        std::cout << termcolor::bright_white << std::setw(fill_size / 2) << std::setfill(fillchar) << ' '
                  << termcolor::bright_blue << str_toupper(title) << ' '
                  << termcolor::bright_white << std::setw(fill_size - fill_size / 2)
                  << termcolor::reset << std::setfill(fillchar) << std::endl;
    }
    void print_comment(const std::string &title)
    {
        if (!verbose)
            return;
        size_t fill_size = fieldwidth > title.size() + 4 ? fieldwidth - title.size() - 4 : 0;

        std::cout << "| "
                  << termcolor::yellow << title
                  << std::setw(fill_size) << std::setfill(' ')
                  << termcolor::reset << " |" << std::endl;
    }
    void print_warning(const std::string &title)
    {
        if (!verbose)
            return;
        size_t fill_size = fieldwidth > title.size() + 4 ? fieldwidth - title.size() - 4 : 0;

        std::cout << "| "
                  << termcolor::red << title
                  << std::setw(fill_size) << std::setfill(' ')
                  << termcolor::reset << " |" << std::endl;
    }
    void print_string(const std::string &head, const std::string &content, const std::string &tail = std::string())
    {
        if (!verbose)
            return;
        size_t fill_size = fieldwidth > 11 + head.size() ? fieldwidth - 11 - head.size() : 0;
        if (tail.empty())
            std::cout << "| " << termcolor::cyan << head << ": "
                      << termcolor::white << std::right << std::setfill(' ')
                      << std::setw(fill_size) << shorten(content) << termcolor::reset << " |" << std::endl;
        else
        {

            std::cout << "| " << termcolor::cyan << head << ": "
                      << termcolor::white << std::right << std::setfill(' ')
                      << std::setw(fill_size) << shorten(content) + " " + tail << termcolor::reset << " |" << std::endl;
        }
    }
    void print_number(const std::string &head, const int content, const std::string &tail = std::string())
    {
        print_string(head, std::to_string(content), tail);
    }
    void print_number(const std::string &head, const float content, const std::string &tail = std::string())
    {
        char buffer[50];
        sprintf(buffer, "%.2f", content);
        print_string(head, buffer, tail);
    }
    void print_number(const std::string &head, const double content, const std::string &tail = std::string())
    {
        char buffer[50];
        sprintf(buffer, "%.2f", content);
        print_string(head, std::to_string(content), tail);
    }
    void print_boolean(const std::string &head, const bool content)
    {
        print_string(head, content ? "True" : "False");
    }
    void print_program_start(const std::string &programname)
    {
        if (!verbose)
            return;
        std::time_t result_time = std::time(nullptr);
        std::string title = str_toupper(programname) + " started on " + std::asctime(std::localtime(&result_time));
        std::cout << std::endl;
        print_title(rtrim(title), '=');
    }
    void print_program_end(const std::string &programname)
    {
        if (!verbose)
            return;
        std::time_t result_time = std::time(nullptr);
        std::string title = str_toupper(programname) + " ended on " + std::asctime(std::localtime(&result_time));
        print_title(rtrim(title), '=');
        std::cout << std::endl;
    }

protected:
    std::string str_toupper(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                       { return std::toupper(c); });
        return s;
    }
    std::string &ltrim(std::string &str)
    {
        auto it2 = std::find_if(str.begin(), str.end(), [](char ch)
                                { return !std::isspace(static_cast<unsigned char>(ch)); });
        str.erase(str.begin(), it2);
        return str;
    }

    std::string &rtrim(std::string &str)
    {
        auto it1 = std::find_if(str.rbegin(), str.rend(), [](char ch)
                                { return !std::isspace(static_cast<unsigned char>(ch)); });
        str.erase(it1.base(), str.end());
        return str;
    }

    std::string &trim(std::string &str)
    {
        return ltrim(rtrim(str));
    }

    std::string shorten(const std::string &str)
    {
        std::string s = str;
        if (str.size() > 35)
            s = str.substr(0, 15) + "..." + str.substr(str.size() - 15, str.size());
        return trim(s);
    }

protected:
    bool verbose;
    size_t fieldwidth = 80;
};