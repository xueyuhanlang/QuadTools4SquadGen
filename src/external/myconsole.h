#pragma once

#include <iostream>
#include <iomanip>
#include <cctype>
#include <string>
#include <string_view>
#include <array>
#include <algorithm>
#include <ctime>
#include <cstdio>
#include "termcolor.hpp"

class MyConsole
{
public:
    explicit MyConsole(bool verbose_tag = true) noexcept
        : verbose_(verbose_tag) {}

    void set_verbose(bool verbose_tag) noexcept { verbose_ = verbose_tag; }
    void set_width(int w) noexcept { field_width_ = std::max(80, w); }

    void print_separator() const
    {
        if (!verbose_) return;
        std::cout << termcolor::bright_white
                  << std::setfill('-') << std::setw(field_width_)
                  << termcolor::reset << '\n';
    }

    void print_title(std::string_view title, char fillchar = '-') const
    {
        if (!verbose_) return;
        const int title_len  = static_cast<int>(title.size());
        const int fill_size  = field_width_ > title_len + 1 ? field_width_ - title_len - 1 : 0;
        std::cout << termcolor::bright_white
                  << std::setw(fill_size / 2) << std::setfill(fillchar) << ' '
                  << termcolor::bright_blue << str_toupper(title) << ' '
                  << termcolor::bright_white << std::setw(fill_size - fill_size / 2)
                  << std::setfill(fillchar) << termcolor::reset << '\n';
    }

    void print_comment(std::string_view title) const
    {
        if (!verbose_) return;
        const int fill_size = field_width_ > static_cast<int>(title.size()) + 4
                            ? field_width_ - static_cast<int>(title.size()) - 4 : 0;
        std::cout << "| " << termcolor::yellow << title
                  << std::setw(fill_size) << std::setfill(' ')
                  << termcolor::reset << " |\n";
    }

    void print_warning(std::string_view title) const
    {
        if (!verbose_) return;
        const int fill_size = field_width_ > static_cast<int>(title.size()) + 4
                            ? field_width_ - static_cast<int>(title.size()) - 4 : 0;
        std::cout << "| " << termcolor::red << title
                  << std::setw(fill_size) << std::setfill(' ')
                  << termcolor::reset << " |\n";
    }

    void print_string(std::string_view head, std::string_view content, std::string_view tail = {}) const
    {
        if (!verbose_) return;
        const int fill_size = field_width_ > 11 + static_cast<int>(head.size())
                            ? field_width_ - 11 - static_cast<int>(head.size()) : 0;
        const std::string shortened = shorten(content);
        const std::string display   = tail.empty() ? shortened : shortened + ' ' + std::string(tail);
        std::cout << "| " << termcolor::cyan << head << ": "
                  << termcolor::white << std::right << std::setfill(' ')
                  << std::setw(fill_size) << display
                  << termcolor::reset << " |\n";
    }

    void print_number(std::string_view head, int content, std::string_view tail = {}) const
    {
        print_string(head, std::to_string(content), tail);
    }

    void print_number(std::string_view head, float content, std::string_view tail = {}) const
    {
        std::array<char, 50> buf{};
        std::snprintf(buf.data(), buf.size(), "%.2f", content);
        print_string(head, buf.data(), tail);
    }

    void print_number(std::string_view head, double content, std::string_view tail = {}) const
    {
        std::array<char, 50> buf{};
        std::snprintf(buf.data(), buf.size(), "%.2f", content);
        print_string(head, buf.data(), tail);
    }

    void print_boolean(std::string_view head, bool content) const
    {
        print_string(head, content ? "True" : "False");
    }

    void print_program_start(std::string_view programname) const
    {
        if (!verbose_) return;
        std::array<char, 64> time_buf{};
        format_current_time(time_buf);
        const std::string title = std::string(str_toupper(programname)) + " started on " + time_buf.data();
        std::cout << '\n';
        print_title(title, '=');
    }

    void print_program_end(std::string_view programname) const
    {
        if (!verbose_) return;
        std::array<char, 64> time_buf{};
        format_current_time(time_buf);
        const std::string title = std::string(str_toupper(programname)) + " ended on " + time_buf.data();
        print_title(title, '=');
        std::cout << '\n';
    }

private:
    [[nodiscard]] static std::string str_toupper(std::string_view s)
    {
        std::string result(s);
        std::transform(result.begin(), result.end(), result.begin(),
                       [](unsigned char c) { return std::toupper(c); });
        return result;
    }

    [[nodiscard]] static std::string shorten(std::string_view str)
    {
        // Trim whitespace helper
        auto trim_sv = [](std::string_view s) -> std::string_view {
            const auto start = s.find_first_not_of(" \t\r\n");
            if (start == std::string_view::npos) return {};
            return s.substr(start, s.find_last_not_of(" \t\r\n") - start + 1);
        };

        if (str.size() > 35) {
            const std::string s = std::string(str.substr(0, 15)) + "..."
                                + std::string(str.substr(str.size() - 15));
            return std::string(trim_sv(s));
        }
        return std::string(trim_sv(str));
    }

    static void format_current_time(std::array<char, 64>& buf) noexcept
    {
        const std::time_t t = std::time(nullptr);
        std::tm tm_buf{};
#ifdef _MSC_VER
        localtime_s(&tm_buf, &t);
#else
        localtime_r(&t, &tm_buf);
#endif
        std::strftime(buf.data(), buf.size(), "%c", &tm_buf);
    }

    bool verbose_;
    int  field_width_ = 80;
};