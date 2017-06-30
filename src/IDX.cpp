#include "IDX.h"
#include <sstream>
#include <iomanip>

using namespace std;
using namespace cv;

IDX::IDX() {}

IDX::IDX(const char* filename)
{
	load(filename);
}

IDX::~IDX()
{
	destroy();
}

void IDX::destroy()
{
	data.clear();
	data.shrink_to_fit();
	num_data = 0;
}

template<typename T>
vector<T>& operator[](int x)
{
	if(x < 0 || num_data <= x)
		throw IDXIndexOutOfBoundsException();
	if(type == 0x08)
		loadData<unsigned char>(in);
	else if(type == 0x09)
		loadData<char>(in);
	else if(type == 0x0B)
		loadData<short>(in);
	else if(type == 0x0C)
		loadData<int>(in);
	else if(type == 0x0D)
		loadData<float>(in);
	else if(type == 0x0E)
		loadData<double>(in);
}


template<typename T>
void IDX::load(const char* filename)
{
	if(type != 0)
		destroy();
	this->filename = string(filename);
	ifstream in(filename);

	//magic number
	if(read<unsigned char>(in) + read<unsigned char>(in) != 0)
		throw badIDXFormatException;

	type = read<unsigned char>(in);
	num_dims = read<unsigned char>(in);

	//dimensions. Assume first dim is number of items, rest of dims is size of (num_dims-1)-dimensional items
	num_data = readBigEndianInt(in);
	dims.resize(num_dims - 1);
	flatsize = 1;
	for(int i = 0; i < num_dims - 1; i++)
	{
		dims[i] = readBigEndianInt(in);
		flatsize *= dims[i];
	}

	//read in data
	if(type == 0x08)
		loadData<T, unsigned char>(in);
	else if(type == 0x09)
		loadData<T, char>(in);
	else if(type == 0x0B)
		loadData<T, short>(in);
	else if(type == 0x0C)
		loadData<T, int>(in);
	else if(type == 0x0D)
		loadData<T, float>(in);
	else if(type == 0x0E)
		loadData<T, double>(in);
	else
	{
		printf("Unknown type! Exiting. \"%s\"!\n",filename);
		exit(0);
	}
}

void IDX::append(const char* filename)
{
	append(IDX(filename));
}
IDX& IDX::operator+=(const IDX& other)
{
	append(other);
	return *this;
}
void IDX::append(const IDX& other)
{
	checkType(type);
	checkType(other.type);
	if(!checkDimsWithMe(other))
		throw inconsistentDimsException;

	//add to num_data
	num_data += other.num_data;

	//read in data
	if(type == 0x08)
		appendData<unsigned char>(other);
	else if(type == 0x09)
		appendData<char>(other);
	else if(type == 0x0B)
		appendData<short>(other);
	else if(type == 0x0C)
		appendData<int>(other);
	else if(type == 0x0D)
		appendData<float>(other);
	else if(type == 0x0E)
		appendData<double>(other);
}

template<typename MyType>
void IDX::appendData(const IDX& other)
{
	if(other.type == 0x08)
		appendData<unsigned char, MyType>(other);
	else if(other.type == 0x09)
		appendData<char, MyType>(other);
	else if(other.type == 0x0B)
		appendData<short, MyType>(other);
	else if(other.type == 0x0C)
		appendData<int, MyType>(other);
	else if(other.type == 0x0D)
		appendData<float, MyType>(other);
	else if(other.type == 0x0E)
		appendData<double, MyType>(other);
}

template<typename OtherType, typename MyType>
void IDX::appendData(const IDX& other)
{
	vector<vector<MyType> > *myData = (vector<vector<MyType> > *)data;
	vector<vector<OtherType> > *otherData = (vector<vector<OtherType> > *)other.data;

	size_t oldSize = myData->size();
	myData->resize(num_data); //should never be smaller than before
	for(int m = oldSize, o = 0; m < num_data; m++, o++)
	{
		myData->at(m).resize(flatsize);
		for(int j = 0; j < flatsize; j++)
			myData->at(m)[j] = otherData->at(o)[j];
	}
}


void IDX::getDims(vector<int32_t>& dest) const
{
	dest = dims;
}
int32_t IDX::getNumData() const
{
	return num_data;
}

bool IDX::checkDimsWithMe(const IDX& other)
{
	bool good = dims.size() == other.dims.size();
	if(!good) 
		return false;
	for(int i = 0; i < dims.size(); i++)
		if(dims[i] != other.dims[i])
			return false;
	return true;
}

template<typename T, typename OtherType>
void IDX::loadData(ifstream& in)
{
	data.resize(num_data)

	//get each data item
	for(int i = 0; i < num_data; i++)
	{
		data[i].resize(flatsize);
		for(int j = 0; j < flatsize; j++)
			data[i][j] = (T)read<OtherType>(in);
		if(in.eof())
			throw notEnoughDataException;
	}
}

void IDX::write(const char* filename)
{
	checkType(this->type);
	write(filename,this->type);
}

template<typename T>
void IDX::write(const char* filename, unsigned char type)
{
	//check type for validity, also if this->type is not valid it means an IDX hasn't been loaded
	checkType(type);

	//open ofstream
	ofstream out(filename, ios::trunc);

	//write magic number (0, 0, type, num_dims)
	out.put(0);
	out.put(0);
	out.put(type);
	out.put(num_dims);

	//write dims as Big Endian Int
	for(int j = 24; j >= 0; j-=8)
		out.put(num_data >> j);
	for(size_t i = 0; i < dims.size(); i++)
		for(int j = 24; j >= 0; j-=8)
			out.put(dims[i] >> j);

	//write data
	if(type == IDX_UCHAR)
		writeData<unsigned char>(out,type);
	else if(type == IDX_CHAR)
		writeData<char>(out,type);
	else if(type == IDX_SHORT)
		writeData<short>(out,type);
	else if(type == IDX_INT)
		writeData<int>(out,type);
	else if(type == IDX_FLOAT)
		writeData<float>(out,type);
	else if(type == IDX_DOUBLE)
		writeData<double>(out,type);
	//should never be an else due to check at the top
}

template<typename T>
void addMat(const Mat& mat)
{
	unsigned char chans = 1 + (mat.type() >> CV_CN_SHIFT);
	if(dims.size() != 3)
		throw nonImageIDXException;
	if(chans != dims[2] || mat.rows != dims[1] || mat.cols != dims[0])
		throw inconsistentDimsException;

	data.push_back(vector<vector<T> >(flatsize));

	int i = 0;
	for(int y = 0; y < mat.rows; y++)
		for(int x = 0; x < mat.cols; x++)
		{
			const Vec3b& pix = mat.at(x,y);
			data.back()[i++] = (T)pix[2];
			data.back()[i++] = (T)pix[1];
			data.back()[i++] = (T)pix[0];
		}

	// //write data
	// if(this->type == 0x08)
	// 	addMatData<char>(mat);
	// else if(this->type == 0x09)
	// 	addMatData<unsigned char>(mat);
	// else if(this->type == 0x0B)
	// 	addMatData<short>(mat);
	// else if(this->type == 0x0C)
	// 	addMatData<int>(mat);
	// else if(this->type == 0x0D)
	// 	addMatData<float>(mat);
	// else if(this->type == 0x0E)
	// 	addMatData<double>(mat);
}

// template<typename T>
// void addMatData(const Mat& mat)
// {
// 	data.push_back(vector<vector<T> >(flatsize));

// 	int i = 0;
// 	for(int y = 0; y < mat.rows; y++)
// 		for(int x = 0; x < mat.cols; x++)
// 		{
// 			const Vec3b& pix = mat.at(x,y);
// 			data.back()[i++] = (T)pix[2];
// 			data.back()[i++] = (T)pix[1];
// 			data.back()[i++] = (T)pix[0];
// 		}
// }

// template<typename T, typename OutType>
// void IDX::writeData(ofstream& out, uint8_t outType)
// {
// 	//write data
// 	if(outType == 0x08)
// 		writeData<char, OutType>(out);
// 	else if(outType == 0x09)
// 		writeData<unsigned char, OutType>(out);
// 	else if(outType == 0x0B)
// 		writeData<short, OutType>(out);
// 	else if(outType == 0x0C)
// 		writeData<int, OutType>(out);
// 	else if(outType == 0x0D)
// 		writeData<float, OutType>(out);
// 	else if(outType == 0x0E)
// 		writeData<double, OutType>(out);
// }

template<typename OutType>
void IDX::writeData(ofstream& out)
{
	for(int i = 0; i < data.size(); i++)
		for(int j = 0; j < data[i].size(); j++)
		{
			OutType dat = (OutType)data[i][j];
			out.write(reinterpret_cast<const char *>(&dat),sizeof(OutType));
		}
}

void IDX::checkType(uint8_t type)
{
	if(type == 0x08 || type ==0x09 || (0x0B <= type && type <= 0x0E))
		return;
	throw badTypeDimsException;
}

void IDX::printMetaData() const
{
	stringstream ss;
	ss << "Filename: " << filename << endl;
	ss << "Type: 0x" << setfill('0') << setw(2) << std::hex << (int)type << std::dec << endl;
	ss << "Num Data: " << num_data << endl;
	ss << "Dims:" << endl;
	for(int i = 0; i < dims.size(); i++)
		ss << "\t" << dims[i] << endl;
	printf("%s\n", ss.str().c_str());
}


template<typename ReadType>
ReadType IDX::read(ifstream& in)
{
	ReadType num;
	in.read((char*)&num,sizeof(ReadType));
	return num;
}
int IDX::readBigEndianInt(ifstream& in)
{
	int32_t out = 0;
	for(int i=3; i >= 0; i--)
		out |= (read<uint8_t>(in) << 8*(i));
	return out;
}