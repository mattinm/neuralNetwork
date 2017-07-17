
#ifndef __ConvNet_IDX__
#define __ConvNet_IDX__

#include <fstream>
#include <vector>
#include <string>
#include <exception>
#include "opencv2/imgproc/imgproc.hpp" //used for showing images being read in from IDX
#include "opencv2/highgui/highgui.hpp"


#define IDX_UCHAR  0x08
#define IDX_CHAR   0x09
#define IDX_SHORT  0x0B
#define IDX_INT    0x0C
#define IDX_FLOAT  0x0D
#define IDX_DOUBLE 0x0E

class InconsistentDimsException: public std::exception{
  virtual const char* what() const throw(){
    return "The object given has inconsistent dimensions with this IDX.";
  }
};

class BadTypeException: public std::exception{
  virtual const char* what() const throw(){
    return "The type given is not supported.";
  }
};

class BadIDXFormatException: public std::exception{
  virtual const char* what() const throw(){
    return "IDX format appears to be incorrect.";
  }
};

class NotEnoughDataException: public std::exception{
  virtual const char* what() const throw(){
    return "There doesn't appear to be enough data in the IDX file.";
  }
};

class NonImageIDXException: public std::exception{
  virtual const char* what() const throw(){
    return "You are trying to add an image to an IDX that is not 3 dimensions.";
  }
};
class IDXIndexOutOfBoundsException: public std::exception{
  virtual const char* what() const throw(){
    return "The index requested is out of allocated bounds.";
  }
};


template <typename T>
class IDX{
private:
	template <typename OutType, typename DUMMY = void> 
	struct TypeWriter{
		static BadIDXFormatException badIDXFormatException;
		static void write(std::ofstream& out) { throw badIDXFormatException; }
	};

	template <typename DUMMY> 
	struct TypeWriter<unsigned char, DUMMY>{
		static void write(std::ofstream& out) { out.put(IDX_UCHAR); }
	};
		template <typename DUMMY> 
	struct TypeWriter<char, DUMMY>{
		static void write(std::ofstream& out) { out.put(IDX_CHAR); }
	};
		template <typename DUMMY> 
	struct TypeWriter<short, DUMMY>{
		static void write(std::ofstream& out) { out.put(IDX_SHORT); }
	};
		template <typename DUMMY> 
	struct TypeWriter<int, DUMMY>{
		static void write(std::ofstream& out) { out.put(IDX_INT); }
	};
		template <typename DUMMY> 
	struct TypeWriter<float, DUMMY>{
		static void write(std::ofstream& out) { out.put(IDX_FLOAT); }
	};
		template <typename DUMMY> 
	struct TypeWriter<double, DUMMY>{
		static void write(std::ofstream& out) { out.put(IDX_DOUBLE); }
	};
public:
	//constructors and destructors
	IDX();
	IDX(const char* filename);
	~IDX();
	void destroy();

	//operators and data manip
	IDX<T>& operator+=(const IDX<T>& other);
	std::vector<T>& operator[](int x);
	void erase(int x);
	void erase(std::vector<int> v);
	void getFlatData(std::vector<T>& dest);
	void getData(std::vector<std::vector<T> >& dest);
	std::vector<std::vector<T> >* data();

	//load, write, append
	void load(const char* filename);
	void append(const char* filename);
	void append(const IDX<T>& other);
	void write(const char* filename);
	void write(const std::string& filename);
	template<typename OutType>
	void write(const char* filename);
	template<typename OutType>
	void write(const std::string& filename);

	//dimensions and metadata
	void getDims(std::vector<int32_t>& dest) const;
	int32_t getNumData() const;
	void printMetaData() const;

	void addMat(const cv::Mat& mat);

private:
//exceptions
	InconsistentDimsException inconsistentDimsException;
	BadTypeException badTypeDimsException;
	BadIDXFormatException badIDXFormatException;
	NotEnoughDataException notEnoughDataException;
	NonImageIDXException nonImageIDXException;
	IDXIndexOutOfBoundsException idxIndexOutOfBoundsException;

//variables
	unsigned char type = 0, num_dims;
	int32_t num_data;
	long flatsize = 1;
	std::vector<int32_t> dims;
	std::string filename;

	std::vector<std::vector<T> > _data;

//functions
	template<typename OtherType>
	void loadData(std::ifstream& in);

	void checkType(uint8_t type);
	bool checkDimsWithMe(const IDX& other);

	template<typename ReadType>
	ReadType read(std::ifstream& in);
	int readBigEndianInt(std::ifstream& in);

	template<typename OutType>
	void writeType(std::ofstream& out);
};

#endif