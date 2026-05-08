#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <sys/stat.h>
#include <fcntl.h>

#define PROT_READ 1
#define MAP_PRIVATE 2
#define MAP_FAILED ((void*)-1)

inline void * mmap(void * addr, size_t length, int prot, int flags, int fd, off_t offset) {
    HANDLE hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE) return MAP_FAILED;
    
    // For reading, we use PAGE_READONLY
    HANDLE hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapping == NULL) return MAP_FAILED;
    
    void * view = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, length);
    CloseHandle(hMapping); // The mapping object can be closed as soon as a view is mapped.
    
    if (view == NULL) return MAP_FAILED;
    return view;
}

inline int munmap(void * addr, size_t length) {
    return UnmapViewOfFile(addr) ? 0 : -1;
}

// Windows equivalents for POSIX file functions
#define open _open
#define fstat _fstat
#define stat _stat
#define close _close
#define O_RDONLY _O_RDONLY

// Windows doesn't have S_ISREG macro in its simplified sys/stat.h
#ifndef S_ISREG
#define S_ISREG(mode) (((mode) & _S_IFMT) == _S_IFREG)
#endif

#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
