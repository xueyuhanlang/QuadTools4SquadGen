#pragma once

#pragma warning(disable : 4996)

#include <string>
#include <ctime>
#include <filesystem>
#include <algorithm>

// inline std::string GetFileExtension(const std::string &FileName)
// {
//     std::filesystem::path filePath(FileName);
//     std::string ext = filePath.extension().string();
//     std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c)
//                    { return std::tolower(c); }); //
//     return ext;
// }

inline std::string GetFileExtension(const std::string &FileName)
{
    if (FileName.find_last_of(".") != std::string::npos)
    {
        auto ext = FileName.substr(FileName.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c)
                       { return std::tolower(c); }); //
        return ext;
    }
    return "";
}

inline std::string GetFileNameWithoutExtension(const std::string &FileName)
{
    std::filesystem::path filePath(FileName);
    return filePath.stem().string();
}

inline std::string GetFileNameWithoutDirInfo(const std::string &FileName)
{
    std::filesystem::path filePath(FileName);
    return filePath.filename().string();
}

inline std::string GetFileDirectory(const std::string &FileName)
{
    std::filesystem::path filePath(FileName);
    if (!filePath.has_parent_path())
        return std::filesystem::current_path().string();
    else
        return filePath.parent_path().string();
}

inline std::string GetDirectoryShortName(const std::string &DirName)
{
    auto lastpos = DirName.find_last_of("/\\");

    if (lastpos == DirName.size() - 1)
    {
        auto s = DirName.substr(0, lastpos);
        auto prevpos = s.find_last_of("/\\");
        if (prevpos != std::string::npos)
            return s.substr(prevpos + 1, std::string::npos);
        else
            return s;
    }
    else if (lastpos != std::string::npos)
    {
        return DirName.substr(lastpos + 1, std::string::npos);
    }
    return DirName;
}

inline std::string path_concatenate(const std::string &dir, const std::string &file)
{
    std::filesystem::path dir_path(dir);
    std::filesystem::path file_path(file);
    return (dir_path / file_path).string();
}

inline std::string zfill(int number, std::size_t width)
{
    std::string str = std::to_string(number);

    // Handle negative numbers: keep the minus sign at the front
    if (str[0] == '-')
    {
        std::string digits = str.substr(1);
        if (digits.length() < width - 1)
        {
            digits = std::string(width - 1 - digits.length(), '0') + digits;
        }
        return "-" + digits;
    }
    else
    {
        if (str.length() < width)
        {
            str = std::string(width - str.length(), '0') + str;
        }
        return str;
    }
}

inline std::string GetTimeString()
{
    time_t now = time(0);
    tm *ltm = localtime(&now);
    return std::to_string(1900 + ltm->tm_year) +
           (ltm->tm_mon < 9 ? "0" : "") + std::to_string(1 + ltm->tm_mon) +
           (ltm->tm_mday < 10 ? "0" : "") + std::to_string(ltm->tm_mday) + "-" +
           (ltm->tm_hour < 10 ? "0" : "") + std::to_string(ltm->tm_hour) + "-" +
           (ltm->tm_min < 10 ? "0" : "") + std::to_string(ltm->tm_min);
}