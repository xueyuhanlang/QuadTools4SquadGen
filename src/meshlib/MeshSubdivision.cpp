#include "MeshSubdivision.h"
#include "looputil.h"

namespace MeshLib
{
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	Mesh3D<Real> *MeshSubdivision<Real>::Catmull_Clark(std::vector<Real> *vert_color, std::vector<Real> *new_vert_color)
	{
		if (m_pmesh == NULL)
		{
			return NULL;
		}

		Mesh3D<Real> *New_mesh = new Mesh3D<Real>;

		if (new_vert_color)
		{
			new_vert_color->resize(0);
			new_vert_color->reserve(m_pmesh->get_num_of_vertices() + m_pmesh->get_num_of_faces() + m_pmesh->get_num_of_edges() / 2);
		}

		typename Mesh3D<Real>::PTR_VERTEX_LIST vertices_list = m_pmesh->get_vertices_list();
		typename Mesh3D<Real>::VERTEX_ITER viter = vertices_list->begin();

		ptrdiff_t num_v = m_pmesh->get_num_of_vertices();

		for (; viter != vertices_list->end(); viter++)
		{
			auto hv = New_mesh->insert_vertex((*viter)->pos);
			if (new_vert_color)
			{
				new_vert_color->emplace_back((*vert_color)[hv->id]);
			}
		}

		// add face point
		typename Mesh3D<Real>::PTR_FACE_LIST faces_list = m_pmesh->get_faces_list();
		typename Mesh3D<Real>::FACE_ITER fiter = faces_list->begin();

		for (; fiter != faces_list->end(); fiter++)
		{
			HE_edge<Real> *edge = (*fiter)->edge;
			TinyVector<Real, 3> vv(0, 0, 0);
			int degree = 0;
			do
			{
				vv += edge->vert->pos;
				degree++;
				edge = edge->next;
			} while (edge != (*fiter)->edge);

			TinyVector<Real, 3> nvv = vv / (Real)degree;
			New_mesh->insert_vertex(nvv);

			if (new_vert_color)
			{
				auto min_color = (*vert_color)[edge->pair->vert->id], max_color = (*vert_color)[edge->pair->vert->id];
				do
				{
					min_color = std::min(min_color, (*vert_color)[edge->vert->id]);
					max_color = std::max(max_color, (*vert_color)[edge->vert->id]);
					edge = edge->next;
				} while (edge != (*fiter)->edge);
				new_vert_color->emplace_back((min_color + max_color) / 2);
			}
		}

		// add edge point
		typename Mesh3D<Real>::PTR_EDGE_LIST edges_list = m_pmesh->get_edges_list();
		typename Mesh3D<Real>::EDGE_ITER eiter = edges_list->begin();

		std::unordered_map<HE_edge<Real> *, HE_vert<Real> *> m_map_edge_vert;

		for (; eiter != edges_list->end(); eiter++)
		{
			HE_edge<Real> *edge = *eiter;

			if (m_map_edge_vert[edge] == NULL && m_map_edge_vert[edge->pair] == NULL)
			{
				TinyVector<Real, 3> vv = edge->pair->vert->pos + edge->vert->pos;

				int n = 2;
				if (!m_pmesh->is_on_boundary(edge))
				{
					if (edge->face)
					{
						HE_vert<Real> *hv = New_mesh->get_vertex(edge->face->id + num_v);
						vv += hv->pos;
						n++;
					}
					if (edge->pair->face)
					{
						HE_vert<Real> *hv = New_mesh->get_vertex(edge->pair->face->id + num_v);
						vv += hv->pos;
						n++;
					}
				}

				TinyVector<Real, 3> nvv = vv / (Real)n;
				HE_vert<Real> *hv = New_mesh->insert_vertex(nvv);

				m_map_edge_vert[edge] = hv;
				m_map_edge_vert[edge->pair] = hv;

				if (new_vert_color)
				{
					new_vert_color->emplace_back(((*vert_color)[edge->vert->id] + (*vert_color)[edge->pair->vert->id]) / 2);
				}
			}
		}

		// add updated original point
		vertices_list = m_pmesh->get_vertices_list();
		viter = vertices_list->begin();

		int id = 0;
		for (; viter != vertices_list->end(); viter++, id++)
		{
			if ((*viter)->degree == 2)
			{
				HE_vert<Real> *cv = New_mesh->get_vertex(id);
				continue;
			}

			// compute Q: Q is the average of the new face points surrounding the old vertex
			// Compute R:: R is the average of the midpoints of the edges that share the old vertex,

			// Q 2R S(n-3)
			// --- + ---- + --------
			//	 n n n

			// S is the old vertex point, and
			// n is the number of edges that share the old vertex.

			if (!m_pmesh->is_on_boundary(*viter))
			{
				HE_edge<Real> *edge = (*viter)->edge;
				TinyVector<Real, 3> q(0, 0, 0);
				TinyVector<Real, 3> r(0, 0, 0);
				int facedegree = 0;
				int edgedegree = 0;
				do
				{
					HE_face<Real> *hf = edge->pair->face;
					if (hf)
					{
						HE_vert<Real> *hv = New_mesh->get_vertex(hf->id + num_v);
						q += hv->pos;
						facedegree++;
					}

					r += (edge->vert->pos + edge->pair->vert->pos) * (Real)0.5;

					edge = edge->pair->next;
					edgedegree++;
				} while (edge != (*viter)->edge);

				q /= (Real)facedegree;
				r /= (Real)edgedegree;

				TinyVector<Real, 3> nvv = (q + (Real)2.0 * r + (Real)(edgedegree - 3) * (*viter)->pos) / (Real)edgedegree;

				HE_vert<Real> *cv = New_mesh->get_vertex(id);
				cv->pos = nvv;
			}
			else
			{
				HE_vert<Real> *hv = *viter;
				HE_edge<Real> *edge = hv->edge;

				HE_vert<Real> *v1 = edge->vert;

				do
				{
					edge = edge->pair->next;
				} while (edge->pair->next != hv->edge);

				HE_vert<Real> *v2 = edge->vert;
				TinyVector<Real, 3> nvv = (Real)0.125 * (v1->pos + v2->pos) + (Real)0.75 * hv->pos;
				HE_vert<Real> *cv = New_mesh->get_vertex(id);
				cv->pos = nvv;
			}
		}

		// add face
		faces_list = m_pmesh->get_faces_list();
		fiter = faces_list->begin();

		for (; fiter != faces_list->end(); fiter++)
		{
			HE_vert<Real> *hv_face = New_mesh->get_vertex((*fiter)->id + num_v);
			HE_edge<Real> *edge = (*fiter)->edge;

			do
			{
				typename Mesh3D<Real>::VERTEX_LIST mvlist;

				// assert(hv_face);
				// assert(m_map_edge_vert[edge]);
				// assert(New_mesh->get_vertex(edge->vert->id));
				// assert(m_map_edge_vert[edge->next]);

				mvlist.emplace_back(hv_face);
				mvlist.emplace_back(m_map_edge_vert[edge]);
				mvlist.emplace_back(New_mesh->get_vertex(edge->vert->id));
				mvlist.emplace_back(m_map_edge_vert[edge->next]);
				New_mesh->insert_face(mvlist);
				edge = edge->next;
			} while (edge != (*fiter)->edge);
		}

		m_map_edge_vert.clear();
		New_mesh->update_mesh();

		return New_mesh;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	Mesh3D<Real> *MeshSubdivision<Real>::SplitQuad(std::vector<Real> *vert_color, std::vector<Real> *new_vert_color)
	{
		Mesh3D<Real> *New_mesh = new Mesh3D<Real>;

		if (new_vert_color)
		{
			new_vert_color->resize(0);
			new_vert_color->reserve(m_pmesh->get_num_of_vertices() + m_pmesh->get_num_of_faces() + m_pmesh->get_num_of_edges() / 2);
		}
		// add original point
		typename Mesh3D<Real>::PTR_VERTEX_LIST vertices_list = m_pmesh->get_vertices_list();
		typename Mesh3D<Real>::VERTEX_ITER viter = vertices_list->begin();

		ptrdiff_t num_v = m_pmesh->get_num_of_vertices();

		for (; viter != vertices_list->end(); viter++)
		{
			auto hv = New_mesh->insert_vertex((*viter)->pos);
			if (new_vert_color)
			{
				new_vert_color->emplace_back((*vert_color)[hv->id]);
			}
		}
		// add face point
		typename Mesh3D<Real>::PTR_FACE_LIST faces_list = m_pmesh->get_faces_list();
		typename Mesh3D<Real>::FACE_ITER fiter = faces_list->begin();

		for (; fiter != faces_list->end(); fiter++)
		{
			HE_edge<Real> *edge = (*fiter)->edge;
			TinyVector<Real, 3> vv(0, 0, 0);
			int degree = 0;
			do
			{
				vv += edge->vert->pos;
				degree++;
				edge = edge->next;
			} while (edge != (*fiter)->edge);

			TinyVector<Real, 3> nvv = vv / (Real)degree;
			New_mesh->insert_vertex(nvv);

			if (new_vert_color)
			{
				auto min_color = (*vert_color)[edge->pair->vert->id], max_color = (*vert_color)[edge->pair->vert->id];
				do
				{
					min_color = std::min(min_color, (*vert_color)[edge->vert->id]);
					max_color = std::max(max_color, (*vert_color)[edge->vert->id]);
					edge = edge->next;
				} while (edge != (*fiter)->edge);
				new_vert_color->emplace_back((min_color + max_color) / 2);
			}
		}

		// add edge point
		typename Mesh3D<Real>::PTR_EDGE_LIST edges_list = m_pmesh->get_edges_list();
		typename Mesh3D<Real>::EDGE_ITER eiter = edges_list->begin();

		std::unordered_map<HE_edge<Real> *, HE_vert<Real> *> m_map_edge_vert;

		for (; eiter != edges_list->end(); eiter++)
		{
			HE_edge<Real> *edge = *eiter;
			if (m_map_edge_vert[edge] == NULL && m_map_edge_vert[edge->pair] == NULL)
			{
				TinyVector<Real, 3> nvv = (Real)0.5 * (edge->pair->vert->pos + edge->vert->pos);
				HE_vert<Real> *hv = New_mesh->insert_vertex(nvv);
				m_map_edge_vert[edge] = hv;
				m_map_edge_vert[edge->pair] = hv;

				if (new_vert_color)
				{
					new_vert_color->emplace_back(((*vert_color)[edge->vert->id] + (*vert_color)[edge->pair->vert->id]) / 2);
				}
			}
		}

		// add face
		faces_list = m_pmesh->get_faces_list();
		fiter = faces_list->begin();

		for (; fiter != faces_list->end(); fiter++)
		{
			HE_vert<Real> *hv_face = New_mesh->get_vertex((*fiter)->id + num_v);
			HE_edge<Real> *edge = (*fiter)->edge;

			do
			{
				typename Mesh3D<Real>::VERTEX_LIST mvlist;
				mvlist.emplace_back(hv_face);
				mvlist.emplace_back(m_map_edge_vert[edge]);
				mvlist.emplace_back(New_mesh->get_vertex(edge->vert->id));
				mvlist.emplace_back(m_map_edge_vert[edge->next]);
				New_mesh->insert_face(mvlist);
				edge = edge->next;
			} while (edge != (*fiter)->edge);
		}
		m_map_edge_vert.clear();
		New_mesh->update_mesh();
		return New_mesh;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	Mesh3D<Real> *MeshSubdivision<Real>::SplitQuad4CDF(std::vector<std::set<ptrdiff_t>> &vertex_belonging_clusters,
													   std::unordered_map<ptrdiff_t, std::set<int>> &vertex_to_edgearc_ids)
	{
		Mesh3D<Real> *New_mesh = new Mesh3D<Real>;
		// add original point
		typename Mesh3D<Real>::PTR_VERTEX_LIST vertices_list = m_pmesh->get_vertices_list();
		typename Mesh3D<Real>::VERTEX_ITER viter = vertices_list->begin();

		ptrdiff_t num_v = m_pmesh->get_num_of_vertices();

		for (; viter != vertices_list->end(); viter++)
		{
			auto hv = New_mesh->insert_vertex((*viter)->pos);
		}
		// add face point
		typename Mesh3D<Real>::PTR_FACE_LIST faces_list = m_pmesh->get_faces_list();
		typename Mesh3D<Real>::FACE_ITER fiter = faces_list->begin();

		for (; fiter != faces_list->end(); fiter++)
		{
			HE_edge<Real> *edge = (*fiter)->edge;
			TinyVector<Real, 3> vv(0, 0, 0);
			int degree = 0;
			do
			{
				vv += edge->vert->pos;
				degree++;
				edge = edge->next;
			} while (edge != (*fiter)->edge);

			TinyVector<Real, 3> nvv = vv / (Real)degree;
			New_mesh->insert_vertex(nvv);

			// std::set<ptrdiff_t> face_belonging_clusters;
			std::set<ptrdiff_t> face_belonging_clusters_intersection = vertex_belonging_clusters[edge->pair->vert->id];
			std::set<ptrdiff_t> face_belonging_clusters_union;
			do
			{
				auto &v_clusters = vertex_belonging_clusters[edge->vert->id];
				// std::set<ptrdiff_t> unionset;
				// std::set_union(face_belonging_clusters.begin(), face_belonging_clusters.end(),
				// 			   v_clusters.begin(), v_clusters.end(),
				// 			   std::inserter(unionset, unionset.begin()));
				// face_belonging_clusters = unionset;

				std::set<ptrdiff_t> intersection;
				std::set_intersection(face_belonging_clusters_intersection.begin(), face_belonging_clusters_intersection.end(),
									  v_clusters.begin(), v_clusters.end(),
									  std::inserter(intersection, intersection.begin()));
				face_belonging_clusters_intersection = intersection;

				face_belonging_clusters_union.insert(v_clusters.begin(), v_clusters.end());

				edge = edge->next;
			} while (edge != (*fiter)->edge);

			if (!face_belonging_clusters_intersection.empty())
				vertex_belonging_clusters.emplace_back(face_belonging_clusters_intersection);
			else
			{ // should not happen
				vertex_belonging_clusters.emplace_back(face_belonging_clusters_union);
			}
			// if (face_belonging_clusters_intersection.size() == 0)
			// {
			// 	std::cerr << "Warning: face not belonging to any cluster!" << std::endl;
			// }
		}
		// add edge point
		typename Mesh3D<Real>::PTR_EDGE_LIST edges_list = m_pmesh->get_edges_list();
		typename Mesh3D<Real>::EDGE_ITER eiter = edges_list->begin();

		std::unordered_map<HE_edge<Real> *, HE_vert<Real> *> m_map_edge_vert;

		for (; eiter != edges_list->end(); eiter++)
		{
			HE_edge<Real> *edge = *eiter;
			if (m_map_edge_vert[edge] == NULL && m_map_edge_vert[edge->pair] == NULL)
			{
				TinyVector<Real, 3> nvv = (Real)0.5 * (edge->pair->vert->pos + edge->vert->pos);
				HE_vert<Real> *hv = New_mesh->insert_vertex(nvv);
				m_map_edge_vert[edge] = hv;
				m_map_edge_vert[edge->pair] = hv;

				// set intersection
				std::set<ptrdiff_t> edge_belonging_clusters;
				std::set_intersection(vertex_belonging_clusters[edge->vert->id].begin(), vertex_belonging_clusters[edge->vert->id].end(),
									  vertex_belonging_clusters[edge->pair->vert->id].begin(), vertex_belonging_clusters[edge->pair->vert->id].end(),
									  std::inserter(edge_belonging_clusters, edge_belonging_clusters.begin()));
				if (edge_belonging_clusters.empty())
					std::set_union(vertex_belonging_clusters[edge->vert->id].begin(), vertex_belonging_clusters[edge->vert->id].end(),
								   vertex_belonging_clusters[edge->pair->vert->id].begin(), vertex_belonging_clusters[edge->pair->vert->id].end(),
								   std::inserter(edge_belonging_clusters, edge_belonging_clusters.begin()));
				vertex_belonging_clusters.emplace_back(edge_belonging_clusters);

				if (edge_belonging_clusters.size() == 0)
				{
					std::cerr << "Warning: edge not belonging to any cluster!" << std::endl;
				}

				if (vertex_to_edgearc_ids.find(edge->vert->id) != vertex_to_edgearc_ids.end() &&
					vertex_to_edgearc_ids.find(edge->pair->vert->id) != vertex_to_edgearc_ids.end())
				{
					// if (edge->face && edge->pair->face)
					// {
					// 	auto sharp_angle = compute_dihedral_angle(edge);
					// 	if (sharp_angle > 150)
					// 	{
					// 		continue;
					// 	}
					// }
					std::set<int> edgearc_id;
					std::set_intersection(vertex_to_edgearc_ids[edge->vert->id].begin(), vertex_to_edgearc_ids[edge->vert->id].end(),
										  vertex_to_edgearc_ids[edge->pair->vert->id].begin(), vertex_to_edgearc_ids[edge->pair->vert->id].end(),
										  std::inserter(edgearc_id, edgearc_id.begin()));
					if (edgearc_id.size() > 0)
					{
						vertex_to_edgearc_ids[hv->id] = edgearc_id;
					}
				}
			}
		}

		// add face
		faces_list = m_pmesh->get_faces_list();
		fiter = faces_list->begin();

		for (; fiter != faces_list->end(); fiter++)
		{
			HE_vert<Real> *hv_face = New_mesh->get_vertex((*fiter)->id + num_v);
			HE_edge<Real> *edge = (*fiter)->edge;

			do
			{
				typename Mesh3D<Real>::VERTEX_LIST mvlist;
				mvlist.emplace_back(hv_face);
				mvlist.emplace_back(m_map_edge_vert[edge]);
				mvlist.emplace_back(New_mesh->get_vertex(edge->vert->id));
				mvlist.emplace_back(m_map_edge_vert[edge->next]);
				New_mesh->insert_face(mvlist);
				edge = edge->next;
			} while (edge != (*fiter)->edge);
		}
		m_map_edge_vert.clear();
		New_mesh->update_mesh();
		return New_mesh;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	Mesh3D<Real> *MeshSubdivision<Real>::SplitTri(std::vector<std::vector<ptrdiff_t>> *vert_color_comb)
	{
		if (m_pmesh->is_tri() == false)
		{
			return NULL;
		}

		if (vert_color_comb)
		{
			vert_color_comb->resize(0);
			vert_color_comb->reserve(m_pmesh->get_num_of_vertices() + m_pmesh->get_num_of_edges() / 2);
		}

		m_pmesh->init_edge_tag();

		Mesh3D<Real> *newmesh = new Mesh3D<Real>;

		std::unordered_map<HE_edge<Real> *, HE_vert<Real> *> edgemap;

		ptrdiff_t count = m_pmesh->get_num_of_vertices();
		for (ptrdiff_t i = 0; i < count; i++)
		{
			HE_vert<Real> *hv = m_pmesh->get_vertex(i);
			newmesh->insert_vertex(hv->pos);
			if (vert_color_comb)
			{
				vert_color_comb->push_back({hv->id});
			}
		}

		typename Mesh3D<Real>::PTR_EDGE_LIST edges_list = m_pmesh->get_edges_list();
		typename Mesh3D<Real>::EDGE_ITER eiter = edges_list->begin();

		for (; eiter != edges_list->end(); eiter++)
		{
			HE_edge<Real> *he = *eiter;
			if (he->tag)
			{
				auto V = (Real)0.5 * (he->vert->pos + he->pair->vert->pos);
				HE_vert<Real> *hv = newmesh->insert_vertex(V);
				edgemap[he] = hv;
				edgemap[he->pair] = hv;
				if (vert_color_comb)
				{
					vert_color_comb->push_back({he->vert->id, he->pair->vert->id});
				}
			}
		}

		for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
		{
			HE_face<Real> *hf = m_pmesh->get_face(i);
			HE_edge<Real> *he = hf->edge;
			typename Mesh3D<Real>::VERTEX_LIST vlist2;
			do
			{
				typename Mesh3D<Real>::VERTEX_LIST vlist;
				vlist.emplace_back(newmesh->get_vertex(he->vert->id));
				vlist.emplace_back(edgemap[he->next]);
				vlist.emplace_back(edgemap[he]);
				newmesh->insert_face(vlist);

				vlist2.emplace_back(edgemap[he->next]);
				he = he->next;
			} while (he != hf->edge);
			newmesh->insert_face(vlist2);
		}

		newmesh->update_mesh();
		return newmesh;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	Mesh3D<Real> *MeshSubdivision<Real>::Dual()
	{
		typename Mesh3D<Real>::PTR_FACE_LIST faces_list = m_pmesh->get_faces_list();
		typename Mesh3D<Real>::FACE_ITER fiter = faces_list->begin();

		Mesh3D<Real> *m_dualmesh = new Mesh3D<Real>;

		for (; fiter != faces_list->end(); fiter++)
		{
			HE_face<Real> *hf = *fiter;
			m_dualmesh->insert_vertex(hf->GetCentroid());
		}

		typename Mesh3D<Real>::PTR_VERTEX_LIST vertices_list = m_pmesh->get_vertices_list();
		typename Mesh3D<Real>::VERTEX_ITER viter = vertices_list->begin();
		bool hasface = false;
		for (; viter != vertices_list->end(); viter++)
		{
			HE_vert<Real> *hv = *viter;

			typename Mesh3D<Real>::VERTEX_LIST vlist;

			if (m_pmesh->is_on_boundary(hv))
			{
				continue;
			}

			HE_edge<Real> *he = hv->edge;

			do
			{
				vlist.emplace_back(m_dualmesh->get_vertex(he->face->id));

				he = he->pair->next;
			} while (he != hv->edge);

			std::reverse(vlist.begin(), vlist.end());

			m_dualmesh->insert_face(vlist);
			hasface = true;
		}

		if (!hasface)
		{
			delete m_dualmesh;
			return m_pmesh;
		}

		m_dualmesh->update_mesh();

		return m_dualmesh;
	}
	//////////////////////////////////////////////////////////////////////////
	//----------------------------------------------------------------------------
	// explicit instantiation
	//----------------------------------------------------------------------------
	template class MeshSubdivision<double>;
} // end of namespace