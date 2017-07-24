#include "IDX.h"
#include <sstream>
#include <iomanip>

using namespace std;
using namespace cv;

template<typename T>
IDX<T>::IDX() {}

template<typename T>
IDX<T>::IDX(const char* filename)
{
	load(filename);
}

template<typename T>
IDX<T>::~IDX()
{
	destroy();
}

template<typename T>
void IDX<T>::destroy()
{
	_data.clear();
	_data.shrink_to_fit();
	num_data = 0;
}

template<typename T>
vector<T>& IDX<T>::operator[](int x)
{
	if(x < 0 || num_data <= x)
		throw idxIndexOutOfBoundsException;

	return _data[x];
}

template<typename T>
void IDX<T>::erase(int x)
{
	if(x < 0 || num_data <= x)
		throw idxIndexOutOfBoundsException;

	_data.erase(_data.begin() + x);
	num_data--;
}

template<typename T>
void IDX<T>::erase(vector<int> v)
{
	//sort in descending order
	sort(v.begin(), v.end(), 
		[](const int& first, const int& second) -> bool
		{
			return first > second;
		});

	//erasing in descending order should preserve index values
	for(int i = 0; i < v.size(); i++)
		this->erase(v[i]);
}

template<typename T>
void IDX<T>::getFlatData(vector<T>& dest)
{
	dest.resize(flatsize * _data.size());
	int d = 0;
	for(int i = 0; i < _data.size(); i++)
		for(int j = 0; j < flatsize; j++)
			dest[d++] = _data[i][j];
}

template<typename T>
void IDX<T>::getData(vector<vector<T> >& dest)
{
	dest = _data;
}

template<typename T>
vector<vector<T> >* IDX<T>::data()
{
	return &_data;
}

template<typename T>
void IDX<T>::load(const char* filename)
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
	flatsize = 1;

	dims.resize(num_dims - 1);
	for(int i = 0; i < num_dims - 1; i++)
	{
		dims[i] = readBigEndianInt(in);
		flatsize *= dims[i];
	}



	//read in _data
	if(type == IDX_UCHAR)
		loadData<unsigned char>(in);
	else if(type == IDX_CHAR)
		loadData<char>(in);
	else if(type == IDX_SHORT)
		loadData<short>(in);
	else if(type == IDX_INT)
		loadData<int>(in);
	else if(type == IDX_FLOAT)
		loadData<float>(in);
	else if(type == IDX_DOUBLE)
		loadData<double>(in);
	else
	{
		printf("Unknown type! Exiting. \"%s\"!\n",filename);
		exit(0);
	}
}

template<typename T>
template<typename OtherType>
void IDX<T>::loadData(ifstream& in)
{
	_data.resize(num_data);

	//get each _data item
	for(int i = 0; i < num_data; i++)
	{
		_data[i].resize(flatsize);
		for(int j = 0; j < flatsize; j++)
			_data[i][j] = (T)read<OtherType>(in);
		if(in.eof())
			throw notEnoughDataException;
	}
}

template<typename T>
void IDX<T>::append(const char* filename)
{
	append(IDX<T>(filename));
}

template<typename T>
IDX<T>& IDX<T>::operator+=(const IDX<T>& other)
{
	append(other);
	return *this;
}

template<typename T>
void IDX<T>::append(const IDX<T>& other)
{
	checkType(type);
	checkType(other.type);
	if(!checkDimsWithMe(other))
		throw inconsistentDimsException;


	//add to num_data
	num_data += other.num_data;

	size_t oldSize = _data.size();
	_data.resize(num_data);
	for(size_t m = oldSize, o = 0; m < num_data; m++, o++)
	{
		_data[m].resize(flatsize);
		for(size_t j = 0; j < flatsize; j++)
			_data[m][j] = other._data[o][j];
	}
}

template<typename T>
void IDX<T>::getDims(vector<int32_t>& dest) const
{
	dest = dims;
}

template<typename T>
int32_t IDX<T>::getNumData() const
{
	return num_data;
}

template<typename T>
bool IDX<T>::checkDimsWithMe(const IDX<T>& other)
{
	bool good = dims.size() == other.dims.size();
	if(!good) 
		return false;
	for(int i = 0; i < dims.size(); i++)
		if(dims[i] != other.dims[i])
			return false;
	return true;
}

template<typename T>
void IDX<T>::write(const char* filename)
{
	write<T>(filename);
}

template<typename T>
void IDX<T>::write(const string& filename)
{
	write<T>(filename.c_str());
}

template<typename T>
template<typename OutType>
void IDX<T>::write(const string& filename)
{
	write<OutType>(filename.c_str());
}

template<typename T>
template<typename OutType>
void IDX<T>::write(const char* filename)
{
	//if this->type is not valid it means an IDX hasn't been loaded
	checkType(type);

	//open ofstream
	ofstream out(filename, ios::trunc);

	//write magic number (0, 0, type, num_dims)
	out.put(0);
	out.put(0);
	writeType<OutType>(out); // specializations of the template write the correct one
	out.put(num_dims);

	//write dims as Big Endian Int
	for(int j = 24; j >= 0; j-=8)
		out.put(num_data >> j);
	for(size_t i = 0; i < num_dims-1; i++)
		for(int j = 24; j >= 0; j-=8)
			out.put(dims[i] >> j);

	//write _data
	for(int i = 0; i < _data.size(); i++)
		for(int j = 0; j < _data[i].size(); j++)
		{
			OutType dat = (OutType)_data[i][j];
			out.write(reinterpret_cast<const char *>(&dat),sizeof(OutType));
		}
}

template<typename T>
template<typename OutType>
void IDX<T>::writeType(ofstream& out)
{
	TypeWriter<OutType>::write(out);
}

template<typename T>
void IDX<T>::addMat(const Mat& mat)
{
	unsigned char chans = 1 + (mat.type() >> CV_CN_SHIFT);
	if(dims.size() != 3)
		throw nonImageIDXException;
	if(chans != dims[2] || mat.rows != dims[1] || mat.cols != dims[0])
		throw inconsistentDimsException;

	_data.push_back(vector<T>(flatsize));

	int i = 0;
	for(int y = 0; y < mat.rows; y++)
		for(int x = 0; x < mat.cols; x++)
		{
			const Vec3b& pix = mat.at<Vec3b>(x,y);
			_data.back()[i++] = (T)pix[2];
			_data.back()[i++] = (T)pix[1];
			_data.back()[i++] = (T)pix[0];
		}
	num_data++;
}

template<typename T>
void IDX<T>::checkType(uint8_t type)
{
	if(type == 0x08 || type ==0x09 || (0x0B <= type && type <= 0x0E))
		return;
	throw badTypeDimsException;
}

template<typename T>
void IDX<T>::printMetaData() const
{
	stringstream ss;
	ss << "Filename: " << filename << endl;
	ss << "Type: 0x" << setfill('0') << setw(2) << std::hex << (int)type << std::dec << endl;
	ss << "Num Data: " << num_data << endl;
	if(dims.size() > 0)
	{
		ss << "Dims:" << endl;
		for(int i = 0; i < dims.size(); i++)
			ss << "\t" << dims[i] << endl;
	}
	else
		ss << "Dims: each item is one element" << endl;
	printf("%s\n", ss.str().c_str());
}

template<typename T>
template<typename ReadType>
ReadType IDX<T>::read(ifstream& in)
{
	ReadType num;
	in.read((char*)&num,sizeof(ReadType));
	return num;
}
template<typename T>
int IDX<T>::readBigEndianInt(ifstream& in)
{
	int32_t out = 0;
	for(int i=3; i >= 0; i--)
		out |= (read<uint8_t>(in) << 8*(i));
	return out;
}

template class IDX<unsigned char>;
template class IDX<char>;
template class IDX<short>;
template class IDX<int>;
template class IDX<float>;
template class IDX<double>;

template void IDX<unsigned char>::write<unsigned char>(const string& filename);
template void IDX<unsigned char>::write<unsigned char>(const char* filename);
template void IDX<unsigned char>::write<char>(const string& filename);
template void IDX<unsigned char>::write<char>(const char* filename);
template void IDX<unsigned char>::write<short>(const string& filename);
template void IDX<unsigned char>::write<short>(const char* filename);
template void IDX<unsigned char>::write<int>(const string& filename);
template void IDX<unsigned char>::write<int>(const char* filename);
template void IDX<unsigned char>::write<float>(const string& filename);
template void IDX<unsigned char>::write<float>(const char* filename);
template void IDX<unsigned char>::write<double>(const string& filename);
template void IDX<unsigned char>::write<double>(const char* filename);

template void IDX<char>::write<unsigned char>(const string& filename);
template void IDX<char>::write<unsigned char>(const char* filename);
template void IDX<char>::write<char>(const string& filename);
template void IDX<char>::write<char>(const char* filename);
template void IDX<char>::write<short>(const string& filename);
template void IDX<char>::write<short>(const char* filename);
template void IDX<char>::write<int>(const string& filename);
template void IDX<char>::write<int>(const char* filename);
template void IDX<char>::write<float>(const string& filename);
template void IDX<char>::write<float>(const char* filename);
template void IDX<char>::write<double>(const string& filename);
template void IDX<char>::write<double>(const char* filename);

template void IDX<short>::write<unsigned char>(const string& filename);
template void IDX<short>::write<unsigned char>(const char* filename);
template void IDX<short>::write<char>(const string& filename);
template void IDX<short>::write<char>(const char* filename);
template void IDX<short>::write<short>(const string& filename);
template void IDX<short>::write<short>(const char* filename);
template void IDX<short>::write<int>(const string& filename);
template void IDX<short>::write<int>(const char* filename);
template void IDX<short>::write<float>(const string& filename);
template void IDX<short>::write<float>(const char* filename);
template void IDX<short>::write<double>(const string& filename);
template void IDX<short>::write<double>(const char* filename);

template void IDX<int>::write<unsigned char>(const string& filename);
template void IDX<int>::write<unsigned char>(const char* filename);
template void IDX<int>::write<char>(const string& filename);
template void IDX<int>::write<char>(const char* filename);
template void IDX<int>::write<short>(const string& filename);
template void IDX<int>::write<short>(const char* filename);
template void IDX<int>::write<int>(const string& filename);
template void IDX<int>::write<int>(const char* filename);
template void IDX<int>::write<float>(const string& filename);
template void IDX<int>::write<float>(const char* filename);
template void IDX<int>::write<double>(const string& filename);
template void IDX<int>::write<double>(const char* filename);

template void IDX<float>::write<unsigned char>(const string& filename);
template void IDX<float>::write<unsigned char>(const char* filename);
template void IDX<float>::write<char>(const string& filename);
template void IDX<float>::write<char>(const char* filename);
template void IDX<float>::write<short>(const string& filename);
template void IDX<float>::write<short>(const char* filename);
template void IDX<float>::write<int>(const string& filename);
template void IDX<float>::write<int>(const char* filename);
template void IDX<float>::write<float>(const string& filename);
template void IDX<float>::write<float>(const char* filename);
template void IDX<float>::write<double>(const string& filename);
template void IDX<float>::write<double>(const char* filename);

template void IDX<double>::write<unsigned char>(const string& filename);
template void IDX<double>::write<unsigned char>(const char* filename);
template void IDX<double>::write<char>(const string& filename);
template void IDX<double>::write<char>(const char* filename);
template void IDX<double>::write<short>(const string& filename);
template void IDX<double>::write<short>(const char* filename);
template void IDX<double>::write<int>(const string& filename);
template void IDX<double>::write<int>(const char* filename);
template void IDX<double>::write<float>(const string& filename);
template void IDX<double>::write<float>(const char* filename);
template void IDX<double>::write<double>(const string& filename);
template void IDX<double>::write<double>(const char* filename);