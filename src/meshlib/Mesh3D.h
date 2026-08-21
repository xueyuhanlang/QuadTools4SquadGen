#ifndef MESH3D_H
#define MESH3D_H

#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <queue>
#include <cstdlib>

#include "TinyVector.h"

/** \defgroup MeshCore Mesh data structure */

namespace MeshLib
{
	// declare classes for the compiler
	template <typename Real>
	class HE_vert;
	template <typename Real>
	class HE_face;
	template <typename Real>
	class HE_edge;

	//! vertex class \ingroup MeshCore
	/*!
	 *	The basic vertex class for half-edge structure.
	 */
	template <typename Real>
	class HE_vert
	{
	public:
		TinyVector<Real, 3> pos;	//!< 3D coordinate
		HE_edge<Real> *edge;		//!< one of the half-edges_list emanating from the vertex
		TinyVector<Real, 3> normal; //!< vertex normal
		ptrdiff_t id;				//!< index
		unsigned int degree;		//!< the degree of vertex
		bool tag;					//!< tag for programming easily
		//! constructor
		HE_vert(const TinyVector<Real, 3> &v)
			: pos(v), edge(nullptr), id(-1), degree(0), tag(false)
		{
		}
		// destructor
		~HE_vert() = default;
	};

	//! edge class \ingroup MeshCore
	/*!
	 *	The basic edge class for half-edge structure.
	 */
	template <typename Real>
	class HE_edge
	{
	public:
		HE_vert<Real> *vert; //!< vertex at the end of the half-edge
		HE_edge<Real> *pair; //!< oppositely oriented adjacent half-edge
		HE_face<Real> *face; //!< face the half-edge borders
		HE_edge<Real> *next; //!< next half-edge around the face
		HE_edge<Real> *prev; //!< prev half-edge around the face
		ptrdiff_t id;		 //!< index
		bool tag;			 //!< tag for programming easily
		//! constructor
		HE_edge()
			: vert(nullptr), pair(nullptr), face(nullptr), next(nullptr), prev(nullptr), id(-1), tag(false)
		{
		}
		//! destructor
		~HE_edge() = default;

		//! compute the middle point
		[[nodiscard]] inline TinyVector<Real, 3> GetMidPoint() const
		{
			return (Real)0.5 * (vert->pos + pair->vert->pos);
		}
		//! compute the length
		[[nodiscard]] inline Real GetLength() const
		{
			return (vert->pos - pair->vert->pos).Length();
		}
	};

	//! face class \ingroup MeshCore
	/*!
	 *	The basic face class for half-edge structure.
	 */
	template <typename Real>
	class HE_face
	{
	public:
		HE_edge<Real> *edge;					//!< one of the half-edges_list bordering the face
		unsigned int valence;					//!< the number of edges_list
		TinyVector<Real, 3> normal;				//!< face normal
		int id;									//!< index
		bool tag;								//!< tag for programming easily
		std::vector<ptrdiff_t> texture_indices; //! texture indices
		std::vector<ptrdiff_t> normal_indices;	//! texture indices
		int groupid;
		//! constructor
		HE_face()
			: edge(nullptr), id(-1), tag(false), groupid(-1)
		{
		}
		//! destructor
		~HE_face() = default;
		//! compute the barycenter
		[[nodiscard]] inline TinyVector<Real, 3> GetCentroid() const
		{
			TinyVector<Real, 3> V(0, 0, 0);
			HE_edge<Real> *he = edge;
			int i = 0;
			do
			{
				V += he->vert->pos;
				he = he->next;
				i++;
			} while (he != edge);
			return V / Real(i);
		}
		//! compute the area
		[[nodiscard]] inline Real GetArea() const
		{
			TinyVector<Real, 3> V(0, 0, 0);
			HE_edge<Real> *he = edge;
			do
			{
				V += he->vert->pos.Cross(he->next->vert->pos);
				he = he->next;
			} while (he != edge);
			return V.Length() / 2;
		}
		//! whether texture_indices exists
		[[nodiscard]] bool has_texture_map() const
		{
			return (!texture_indices.empty()) && (texture_indices.size() == static_cast<size_t>(valence));
		}
		//! whether normal_indices exists
		[[nodiscard]] bool has_normal_map() const
		{
			return (!normal_indices.empty()) && (normal_indices.size() == static_cast<size_t>(valence));
		}
	};

	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	bool CompareEdgeID(HE_edge<Real> *he1, HE_edge<Real> *he2)
	{
		return he1->id < he2->id;
	}

	template <typename Real>
	bool CompareVertexID(HE_vert<Real> *hv1, HE_vert<Real> *hv2)
	{
		return hv1->id < hv2->id;
	}

	template <typename Real>
	bool CompareFaceID(HE_face<Real> *hf1, HE_face<Real> *hf2)
	{
		return hf1->id < hf2->id;
	}

	// Hash function for vertex pairs
	template <typename Real>
	struct VertexPairHash
	{
		size_t operator()(const std::pair<HE_vert<Real> *, HE_vert<Real> *> &p) const noexcept
		{
			size_t h1 = std::hash<void *>{}(p.first);
			size_t h2 = std::hash<void *>{}(p.second);
			return h1 ^ (h2 << 1);
		}
	};
	//////////////////////////////////////////////////////////////////////////

	//! Mesh3D class: Half edge data structure \ingroup MeshCore
	/*!
	 * a half-edge based mesh data structure
	 * For understanding half-edge structure,
	 * please read the article in http://www.flipcode.com/articles/article_halfedge.shtml
	 */
	template <typename Real>
	class Mesh3D
	{
	public:
		// type definition
		using VERTEX_LIST = std::vector<HE_vert<Real> *>;
		using FACE_LIST = std::vector<HE_face<Real> *>;
		using EDGE_LIST = std::vector<HE_edge<Real> *>;

		using PTR_VERTEX_LIST = VERTEX_LIST *;
		using PTR_FACE_LIST = FACE_LIST *;
		using PTR_EDGE_LIST = EDGE_LIST *;

		using VERTEX_ITER = typename VERTEX_LIST::iterator;
		using FACE_ITER = typename FACE_LIST::iterator;
		using EDGE_ITER = typename EDGE_LIST::iterator;

		using VERTEX_RITER = typename VERTEX_LIST::reverse_iterator;
		using FACE_RITER = typename FACE_LIST::reverse_iterator;
		using EDGE_RITER = typename EDGE_LIST::reverse_iterator;
		using PAIR_VERTEX = std::pair<HE_vert<Real> *, HE_vert<Real> *>;

	protected:
		// mesh data

		PTR_VERTEX_LIST vertices_list; //!< store vertices
		PTR_EDGE_LIST edges_list;	   //!< store edges
		PTR_FACE_LIST faces_list;	   //!< store faces

		std::vector<TinyVector<float, 2>> texture_array;
		std::vector<TinyVector<float, 3>> normal_array;
		// mesh type

		bool m_closed;	 //!< indicate whether the mesh is closed
		bool m_quad;	 //!< indicate whether the mesh is quadrilateral
		bool m_tri;		 //!< indicate whether the mesh is triangular
		bool m_hex;		 //!< indicate whether the mesh is hexagonal
		bool m_pentagon; //!< indicate whether the mesh is pentagonal

		//! associate two end vertices with its edge: only useful in creating mesh
		std::unordered_map<PAIR_VERTEX, HE_edge<Real> *, VertexPairHash<Real>> m_edgemap;

		// mesh info

		int m_num_components; //!< number of components
		int m_num_boundaries; //!< number of boundaries
		int m_genus;		  //!< the genus value

		bool m_encounter_non_manifold;

	public:
		//! values for the bounding box
		Real xmax, xmin, ymax, ymin, zmax, zmin;

		//! store all the boundary vertices, each vector corresponds to one boundary
		std::vector<std::vector<HE_vert<Real> *>> boundaryvertices;
		std::vector<std::string> groupname;

		//! constructor
		Mesh3D(void);

		//! destructor
		~Mesh3D(void);

		//! get the pointer of vertices list
		[[nodiscard]] inline PTR_VERTEX_LIST get_vertices_list() const
		{
			return vertices_list;
		}

		//! get the pointer of edges list
		[[nodiscard]] inline PTR_EDGE_LIST get_edges_list() const
		{
			return edges_list;
		}

		//! get the pointer of faces list
		[[nodiscard]] inline PTR_FACE_LIST get_faces_list() const
		{
			return faces_list;
		}

		//! get the total number of vertices
		[[nodiscard]] inline ptrdiff_t get_num_of_vertices() const
		{
			return vertices_list ? static_cast<ptrdiff_t>(vertices_list->size()) : 0;
		}

		//! get the total number of faces
		[[nodiscard]] inline ptrdiff_t get_num_of_faces() const
		{
			return faces_list ? static_cast<ptrdiff_t>(faces_list->size()) : 0;
		}

		//! get the total number of half-edges
		[[nodiscard]] inline ptrdiff_t get_num_of_edges() const
		{
			return edges_list ? static_cast<ptrdiff_t>(edges_list->size()) : 0;
		}

		//! get the pointer of the id-th vertex
		[[nodiscard]] inline HE_vert<Real> *get_vertex(ptrdiff_t id) const
		{
			return id >= get_num_of_vertices() || id < 0 ? nullptr : (*vertices_list)[id];
		}

		//! get the pointer of the id-th edge
		[[nodiscard]] inline HE_edge<Real> *get_edge(ptrdiff_t id) const
		{
			return id >= get_num_of_edges() || id < 0 ? nullptr : (*edges_list)[id];
		}

		//! get the pointer of the id-th face
		[[nodiscard]] inline HE_face<Real> *get_face(ptrdiff_t id) const
		{
			return id >= get_num_of_faces() || id < 0 ? nullptr : (*faces_list)[id];
		}

		//! get the number of components
		[[nodiscard]] inline int get_num_of_components() const
		{
			return m_num_components;
		}

		//! get the number of boundaries
		[[nodiscard]] inline int get_num_of_boundaries() const
		{
			return m_num_boundaries;
		}

		//! get the genus
		[[nodiscard]] inline int genus() const
		{
			return m_genus;
		}

		//! check whether the mesh is valid
		[[nodiscard]] inline bool is_valid() const
		{
			if (get_num_of_vertices() == 0 || get_num_of_faces() == 0)
			{
				return false;
			}
			return true;
		}

		//! check whether the mesh is closed
		[[nodiscard]] inline bool is_closed() const
		{
			return m_closed;
		}

		//! check whether the mesh is triangular
		[[nodiscard]] inline bool is_tri() const
		{
			return m_tri;
		}

		//! check whether the mesh is quadrilateral
		[[nodiscard]] inline bool is_quad() const
		{
			return m_quad;
		}

		//! check whether the mesh is hexgaonal
		[[nodiscard]] inline bool is_hex() const
		{
			return m_hex;
		}

		//! check whether the mesh is pentagonal
		[[nodiscard]] inline bool is_pentagon() const
		{
			return m_pentagon;
		}

		//! insert a vertex
		/*!
		 *	\param v a 3d point
		 *	\return a pointer to the created vertex
		 */
		HE_vert<Real> *insert_vertex(const TinyVector<Real, 3> &v);

		//! insert a face
		/*!
		 *	\param vec_hv the vertices list of a face
		 *	\param texture the pointer of texture vector
		 *	\param normal the normal of texture vector
		 *	\return a pointer to the created face
		 */
		HE_face<Real> *insert_face(VERTEX_LIST &vec_hv, std::vector<ptrdiff_t> *texture = nullptr, std::vector<ptrdiff_t> *normal = nullptr);

		//! check whether the vertex is on border
		[[nodiscard]] bool is_on_boundary(HE_vert<Real> *hv) const;
		//! check whether the face is on border
		[[nodiscard]] bool is_on_boundary(HE_face<Real> *hf) const;
		//! check whether the edge is on border
		[[nodiscard]] bool is_on_boundary(HE_edge<Real> *he) const;

		// FILE IO

		//! load a 3D mesh from an OFF format file
		bool load_off(const char *fins);
		//! export the current mesh to an OFF format file
		void write_off(const char *fouts);
		//! load a 3D mesh from an OBJ format file
		bool load_obj(const char *fins);
		//! export the current mesh to an OBJ format file
		void write_obj(const char *fouts);
		//! export to a VTK format file
		void write_vtk(const char *fouts);

		//! update mesh:
		/*!
		 *	call it when you have created the mesh
		 */
		void update_mesh();

		//! update normal
		/*!
		 *	compute all the normals of vertices and faces
		 */
		void update_normal(bool onlyupdate_facenormal = false);

		//! compute the bounding box
		void compute_boundingbox();

		//! get a copy of the current
		Mesh3D<Real> *make_copy();

		//! return a face-orientation-changed mesh
		Mesh3D<Real> *reverse_orientation();

		//! init edge tags
		/*!
		 *	for a pair of edges, only one of them is tagged to be true.
		 */
		void init_edge_tag();

		//! reset all the vertices' tag
		void reset_vertices_tag(bool tag_status);
		//! reset all the faces' tag
		void reset_faces_tag(bool tag_status);
		//! reset all the edges' tag
		void reset_edges_tag(bool tag_status);
		//! reset all tag exclude edges' tag2
		void reset_all_tag(bool tag_status);

		//! translate the mesh with tran_V
		void translate(const TinyVector<Real, 3> &tran_V);
		//! scale the mesh
		void scale(Real factorx, Real factory, Real factorz);

		//! check whether there is any non-manifold case
		[[nodiscard]] inline bool is_encounter_nonmanifold() const
		{
			return m_encounter_non_manifold;
		}

		//! set texture array
		void set_texture_array(const std::vector<TinyVector<float, 2>> &textures)
		{
			texture_array.assign(textures.begin(), textures.end());
		}
		//! set normal array
		void set_normal_array(const std::vector<TinyVector<float, 3>> &normals)
		{
			normal_array.assign(normals.begin(), normals.end());
		}
		//! get texture array
		[[nodiscard]] std::vector<TinyVector<float, 2>> &get_texture_array()
		{
			return texture_array;
		}
		//! get texture array
		[[nodiscard]] const std::vector<TinyVector<float, 2>> &get_texture_array() const
		{
			return texture_array;
		}
		//! get normal array
		[[nodiscard]] std::vector<TinyVector<float, 3>> &get_normal_array()
		{
			return normal_array;
		}
		//! get normal array
		[[nodiscard]] const std::vector<TinyVector<float, 3>> &get_normal_array() const
		{
			return normal_array;
		}

		//! swap edge
		/*!
		 *	\param triedge an edge between two triangular faces
		 */
		void swap_edge(HE_edge<Real> *triedge, bool update_vertex_normal = true);

	private:
		//! insert an edge
		HE_edge<Real> *insert_edge(HE_vert<Real> *vstart, HE_vert<Real> *vend);

		//! clear all the data
		void clear_data();

		//! clear vertices
		void clear_vertices();
		//! clear edges
		void clear_edges();
		//! clear faces
		void clear_faces();

		//! check whether the mesh is closed
		void check_closed();
		//! check the mesh type
		void check_meshtype();

		//! compute all the normals of faces
		void compute_faces_list_normal();
		//! compute the normal of a face
		void compute_perface_normal(HE_face<Real> *hf);
		//! compute all the normals of vertices
		void compute_vertices_list_normal();
		//! compute the normal of a vertex
		void compute_pervertex_normal(HE_vert<Real> *hv);

		//! compute the number of components
		void compute_num_components();
		//! compute the number of boundaries
		void compute_num_boundaries();
		//! compute the genus
		void compute_genus();

		//! handle the boundary half edges specially
		void set_nextedge_for_border_vertices();

		//! remove the vertices which have no connection to others.
		void remove_hanged_vertices();

		//! align edges's id
		/*!
		set mesh's edge(vertex(startid), vertex(endid)->id = edgeid.
		only used in make_copy
		*/
		void copy_edge_id(ptrdiff_t edgeid, ptrdiff_t startid, ptrdiff_t endid, Mesh3D<Real> *mesh);
	};
} // end of namespace

// typedef MeshLib::Mesh3D<float> Mesh3f;
using Mesh3d = MeshLib::Mesh3D<double>;

#endif // MESH3D_H
