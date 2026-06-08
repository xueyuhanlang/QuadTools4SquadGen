#pragma once

#include "Mesh3D.h"
#include <omp.h>

enum KNN_ENGINE
{
	ANN = 0,
	NANOFLANN = 1
};

template <typename T>
class PointCloud
{
public:
	PointCloud(const std::vector<TinyVector<T, 3>> &points)
	{
		pts.resize(points.size());
		double xmin = DBL_MAX, ymin = DBL_MAX, zmin = DBL_MAX;
		double xmax = -DBL_MAX, ymax = -DBL_MAX, zmax = -DBL_MAX;
		for (auto k = 0; k < (ptrdiff_t)points.size(); k++)
		{
			xmin = std::min(xmin, points[k][0]), ymin = std::min(ymin, points[k][1]), zmin = std::min(zmin, points[k][2]);
			xmax = std::max(xmax, points[k][0]), ymax = std::max(ymax, points[k][1]), zmax = std::max(zmax, points[k][2]);
		}
		double xlen = xmax - xmin, ylen = ymax - ymin, zlen = zmax - zmin;
		double max_len = std::max(std::max(xlen, std::max(ylen, zlen)), 1.0e-10);
#pragma omp parallel for
		for (auto k = 0; k < (ptrdiff_t)points.size(); k++)
		{
			pts[k][0] = (points[k][0] - xmin) / max_len;
			pts[k][1] = (points[k][1] - ymin) / max_len;
			pts[k][2] = (points[k][2] - zmin) / max_len;
		}
	}

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the dim'th component of the idx'th point in the class:
	inline double kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		return pts[idx][(int)dim];
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	template <class BBOX>
	bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }

	const TinyVector<T, 3> &operator[](const size_t idx) { return pts[idx]; }

protected:
	std::vector<TinyVector<T, 3>> pts;
};

template <typename Real>
ptrdiff_t MergeSamePoints(const std::vector<TinyVector<Real, 3>> &points,
						  std::vector<ptrdiff_t> &merge2uniqueID_map,
						  std::vector<ptrdiff_t> &back2overlapID_map,
						  double DIST_THRES = (Real)1.0e-10, KNN_ENGINE knn_engine = NANOFLANN);

template <typename Real>
void scale_and_PCA(const std::vector<TinyVector<Real, 3>> &sample_points,
				   TinyVector<Real, 3> rotation[3],
				   TinyVector<Real, 3> inverse_rotation[3],
				   TinyVector<Real, 3> &normalization_center,
				   Real &normalization_scale);

template <typename Real>
void scale_and_PCA(MeshLib::Mesh3D<Real> *mesh,
				   TinyVector<Real, 3> inverse_rotation[3],
				   TinyVector<Real, 3> &normalization_center,
				   Real &normalization_scale, const int num_samples = 2048);