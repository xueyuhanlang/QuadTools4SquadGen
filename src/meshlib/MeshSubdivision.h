#ifndef MESHSUBDIVISION_H
#define MESHSUBDIVISION_H

#include "Mesh3D.h"
#include <set>

/** \defgroup MeshSubdivision Subdivision schemes */

namespace MeshLib
{
	//! Mesh Subdivision \ingroup MeshSubdivision
	/*!
	 *	The class implements several popular subdivision schemes and some splitting schemes.
	 *
	 *	Please refer to subdivision papers for details.
	 */
	template <typename Real>
	class MeshSubdivision
	{
	private:
		Mesh3D<Real> *m_pmesh; //<! the input mesh

	public:
		//! constructor
		MeshSubdivision(Mesh3D<Real> *_mesh)
			: m_pmesh(_mesh)
		{
		}
		//! destructor
		~MeshSubdivision()
		{
		}
		// subdivision schemes

		//! Catmull Clark
		Mesh3D<Real> *Catmull_Clark(std::vector<Real> *vert_color = 0, std::vector<Real> *new_vert_color = 0);

		//! quadrilaterial splitting
		Mesh3D<Real> *SplitQuad(std::vector<Real> *vert_color = 0, std::vector<Real> *new_vert_color = 0);

		//! quadrilaterial splitting
		Mesh3D<Real> *SplitQuad4CDF(std::vector<std::set<ptrdiff_t>> &vertex_belonging_clusters,
									std::unordered_map<ptrdiff_t, std::set<int>> &vertex_to_edgearc_ids);

		//! 1 to 4 triangular splitting
		Mesh3D<Real> *SplitTri(std::vector<std::vector<ptrdiff_t>> *vert_color_comb = 0);

		//! Dualization
		Mesh3D<Real> *Dual();
	};
} // end of namespace

// typedef MeshLib::MeshSubdivision<float> MeshSubdivision3f;
typedef MeshLib::MeshSubdivision<double> MeshSubdivision3d;

#endif // MESHSUBDIVISION_H
