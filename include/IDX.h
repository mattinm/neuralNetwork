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

template <typename T>
class IDX{
public:
	IDX();
	~IDX();
	IDX(const char* filename);

	//operators
	IDX& operator+=(const IDX& other);
	std::vector<T>& operator[](int x);

	//methods
	void load(const char* filename);
	void append(const char* filename);
	void append(const IDX& other);
	void write(const char* filename);
	void write(const char* filename, unsigned char type);
	void getDims(std::vector<int32_t>& dest) const;
	int32_t getNumData() const;
	void destroy();
	void printMetaData() const;

	//getData - template

	void addMat(const cv::Mat& mat);

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
	    return "You are trying to add an image to an IDX that is not 3 dimensions.";
	  }
	};

private:
//exceptions
	InconsistentDimsException inconsistentDimsException;
	BadTypeException badTypeDimsException;
	BadIDXFormatException badIDXFormatException;
	NotEnoughDataException notEnoughDataException;
	NonImageIDXException nonImageIDXException;
	IDXIndexOutOfBoundsException IDXIndexOutOfBoundsException;

//variables
	unsigned char type = 0, num_dims;
	int32_t num_data;
	long flatsize = 1;
	std::vector<int32_t> dims;
	std::string filename;

	std::vector<std::vector<T> > data;

//functions
	template<typename OtherType>
	void loadData(std::ifstream& in);

	// template<typename MyType>
	// void writeData(std::ofstream& out, uint8_t outType);

	template<typename OutType>
	void writeData(std::ofstream& out);

	void checkType(uint8_t type);

	template<typename MyType>
	void appendData(const IDX& other);

	template<typename OtherType, typename MyType>
	void appendData(const IDX& other);

	// void addMatData(cv::Mat& mat);


	template<typename ReadType>
	ReadType read(std::ifstream& in);
	int readBigEndianInt(std::ifstream& in);
	bool checkDimsWithMe(const IDX& other);
};
