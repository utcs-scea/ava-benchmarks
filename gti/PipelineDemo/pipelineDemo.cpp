
/***********************************************************************
*
* Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
* See LICENSE file in the project root for full license information.
*
************************************************************************/
#ifdef _MSC_VER
#include <Windows.h>
#include <locale> 
#include <codecvt>
#endif

#include <iostream>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <memory>
#include <vector>
#include <GTILib.h>
//#include <GtiLog.h>
#define LOG(IO) std::cout

#ifdef _MSC_VER
#else
#include <glob.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef LINUX
#include <unistd.h>
#include <sys/time.h>
unsigned long GetTickCount64_us()
{
    struct timeval tv;
    if(gettimeofday(&tv, NULL) != 0)
            return 0;
    return (tv.tv_sec * 1000000) + (tv.tv_usec);
}
#elif defined ANDROID
#include <time.h>
unsigned long GetTickCount64_us()
{
    struct timespec ts;
    if(clock_gettime(CLOCK_MONOTONIC,&ts) != 0) {
        return 0;
    }
    return (ts.tv_sec * 1000000) + (ts.tv_nsec/1000);
}
#endif

#ifdef WIN32
#include <windows.h>
unsigned long GetTickCount64_us()
{
    return GetTickCount64() * 1000;
}
#endif

using namespace std;

#ifdef _MSC_VER

wstring s2ws(const std::string& str)
{
	using convert_typeX = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_typeX, wchar_t> converterX;

	return converterX.from_bytes(str);
}

string ws2s(const std::wstring& wstr)
{
	using convert_typeX = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_typeX, wchar_t> converterX;

	return converterX.to_bytes(wstr);
}

vector<string> glob(const std::string& folder)
{
	vector<string> names;
	wstring search_path = s2ws(folder) + L"\\*.*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if (wstring(fd.cFileName).compare(L".") == 0) continue;
			if (wstring(fd.cFileName).compare(L"..") == 0) continue;
			string fullname = folder + "\\" + ws2s(fd.cFileName);
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fullname);
			}
			else {
				auto v = glob(fullname);
				names.insert(names.end(), v.begin(), v.end());
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}
#else
vector<string> glob(const std::string& folder) {
	glob_t glob_result;
	vector<string> filenames;
	memset(&glob_result, 0, sizeof(glob_result));

	string folderPattern = folder + "*";
	int return_value = glob(folderPattern.c_str(), GLOB_TILDE | GLOB_MARK, NULL, &glob_result);
	if (return_value != 0) goto _lexit;

	for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
		struct stat s;
		if (stat(glob_result.gl_pathv[i], &s) == 0) {
			if (s.st_mode & S_IFREG)    filenames.push_back(string(glob_result.gl_pathv[i]));
			else if (s.st_mode &S_IFDIR) {
				auto v = glob(string(glob_result.gl_pathv[i]));
				if (v.size()) filenames.insert(filenames.end(), v.begin(), v.end());
			}
		}
	}
_lexit:
	globfree(&glob_result);
	return filenames;
}

#endif




void printResult(GtiTensor *tensorOut){
	if ((tensorOut->format == TENSOR_FORMAT_BINARY)||
	    (tensorOut->format == TENSOR_FORMAT_BINARY_INTEGER) ||
	    (tensorOut->format == TENSOR_FORMAT_BINARY_FLOAT) ){
		ofstream dump("testdump.bin", ios::out | ios::binary | ios::ate | ios::app);
		dump.write((const char*)tensorOut->buffer, tensorOut->size);
		dump.close();
	}
	else{
	   if(tensorOut->tag) cout <<(char *)tensorOut->tag <<endl;
	   cout << (char *)tensorOut->buffer << endl;
	}
}

void evaluateCb(GtiModel *model, const char*layer, GtiTensor *tensorOut) {
    LOG(INFO)<<"Callback tag: "<<tensorOut->tag<<". Layer: "<<layer;
    if(tensorOut->format == TENSOR_FORMAT_JSON) printResult(tensorOut);
}


int main(int argc, const char * argv[]){
    if(argc<3) {LOG(ERROR)<<"Usage: "<<argv[0]<<" "<<"model_file image_dir"; return -1;}      
    int loops=1;
    if(argc>=4) loops=atoi(argv[4]);

    vector<string> fileArray=glob(string(argv[2]));
    if(fileArray.empty()) {LOG(ERROR)<<"could not find files under dir: "<<argv[2]; return -1;}
    size_t seq=0;
    /* the model file path should be absolute when running locally via ava. */
	GtiModel *model = GtiCreateModel(argv[1]);
	if (0 == model) { LOG(ERROR) << "could not open model: " << argv[1]; return -1; }

    for(auto &i:fileArray){
        LOG(INFO)<<"file: "<<i<<endl;
		std::ifstream fin(i, ios::in | ios::binary );
        if(!fin.is_open()){LOG(ERROR)<<"could not open file "<<i; continue;}
		fin.seekg(0,std::ios::end);
		size_t filesize=fin.tellg();
		fin.seekg(0,std::ios::beg);
		unique_ptr<char[]> buffer(new char[filesize]);
		fin.read((char *)buffer.get(), filesize);
		fin.close();

		GtiTensor tensor;
		tensor.height=tensor.width=1;
		tensor.depth=(int)filesize;
		tensor.buffer=buffer.get();
        //tensor.tag=(void *)i.c_str();
        seq++;
        long startTime = GetTickCount64_us();
        //int r = GtiEvaluateWithCallback(model,&tensor, evaluateCb);
        GtiTensor *tensorOut = GtiEvaluate(model,&tensor);
        long endTime = GetTickCount64_us();
        //if(0==r) {LOG(ERROR)<<"evaluate error,exit";break;}
        if(NULL==tensorOut) {LOG(ERROR)<<"evaluate error,exit";break;}
        else {

            char label[100];
            const char *p;
            cout<<(char *)tensorOut->buffer<<endl;
            p = std::strstr((char *)tensorOut->buffer + 20, (const char *)"label");
            p = std::strchr((const char *)p+7, ' ');
            for(int k = 0; k < 100; k++) {
                if(*p == '\"') {
                    label[k] = 0;
                    break;
                }
                label[k] = *p++;
            }
            cout<<"\nRESULT:" << label <<endl;

            float inferenceTime = (endTime - startTime)/1000.0;
            float fps = (1000.0)/inferenceTime;
            cout << "Image " << i << ", inference time = " << inferenceTime  << " ms,  FPS = " << fps << endl <<endl;
        }
    }
	GtiDestroyModel(model);

    return 0;
} 
