#pragma once

#include <string>
#include <string_view>
#include <ctime>
#include <filesystem>
#include <algorithm>
#include <array>
#include <cstdio>

namespace fs = std::filesystem;

[[nodiscard]] inline std::string GetFileExtension(const std::string& fileName)
{
    std::string ext = fs::path(fileName).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    // Strip the leading dot returned by fs::path::extension()
    if (!ext.empty() && ext.front() == '.')
        ext.erase(0, 1);
    return ext;
}

[[nodiscard]] inline std::string GetFileNameWithoutExtension(const std::string& fileName)
{
    return fs::path(fileName).stem().string();
}

[[nodiscard]] inline std::string GetFileNameWithoutDirInfo(const std::string& fileName)
{
    return fs::path(fileName).filename().string();
}

[[nodiscard]] inline std::string GetFileDirectory(const std::string& fileName)
{
    const fs::path p(fileName);
    return p.has_parent_path() ? p.parent_path().string()
                               : fs::current_path().string();
}

[[nodiscard]] inline std::string GetDirectoryShortName(const std::string& dirName)
{
    fs::path p(dirName);
    // Strip trailing separator so filename() returns the last component
    if (p.filename().empty())
        p = p.parent_path();
    return p.filename().string();
}

[[nodiscard]] inline std::string path_concatenate(const std::string& dir, const std::string& file)
{
    return (fs::path(dir) / fs::path(file)).string();
}

[[nodiscard]] inline std::string zfill(int number, std::size_t width)
{
    const bool negative = number < 0;
    std::string digits  = std::to_string(negative ? -number : number);
    const std::size_t target = negative ? width - 1 : width;
    if (digits.size() < target)
        digits.insert(0, target - digits.size(), '0');
    return negative ? '-' + digits : digits;
}

[[nodiscard]] inline std::string GetTimeString()
{
    const std::time_t t = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _MSC_VER
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    std::array<char, 32> buf{};
    std::strftime(buf.data(), buf.size(), "%Y%m%d-%H-%M", &tm_buf);
    return buf.data();
}