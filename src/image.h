#ifndef IMAGE_H_
#define IMAGE_H_

#include <string>
#include <memory>
#include <vector>

template <class T>
class Image {
public:
	typedef std::vector<std::vector<std::vector<T> > > ImageVoxels;
	typedef std::vector<std::vector<T> > ImagePixels;
	Image(std::vector<int> dims) :
		dims_(dims) {
			imageData_ = createCube(dims[0], dims[1], dims[2]);
		}
	T& operator() (const size_t& i, const size_t& j, const size_t& k);
	T operator() (const size_t& i, const size_t& j, const size_t& k) const;
	std::vector<int> getDims() {return dims_;}
	int getDim() {return dims_.size();}
	T getPixelAt(float x, float y, float z);
	T getPixelAt(int x, int y, int z);
private:
	ImageVoxels imageData_;
	std::vector<int> dims_;
	ImageVoxels createCube(size_t dim0, size_t dim1, size_t dim2);
};

template <class T>
typename Image<T>::ImageVoxels Image<T>::createCube(size_t dim0, size_t dim1, size_t dim2) {
	using std::vector;
	return ImageVoxels(dim0, vector<vector<T> >(dim1, vector<T>(dim2)));
}

template <class T>
T& Image<T>::operator() (const size_t& i, const size_t& j, const size_t& k) {
	return imageData_[i][j][k];
}

template <class T>
T Image<T>::operator() (const size_t& i, const size_t& j, const size_t& k) const {
	return imageData_[i][j][k];
}

#endif