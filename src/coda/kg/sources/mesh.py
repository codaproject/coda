import csv
from coda.kg.sources import KGSourceExporter

class MeshExporter(KGSourceExporter):
    name = "mesh_hierarchy"
    def __init__(self):
        super().__init__()

        from indra.databases import mesh_client
        self.mesh_client = mesh_client

    def export(self):
        from indra.ontology.bio import bio_ontology
        edges = set()
        nodes = set()

        for mesh_id, mesh_name in self.mesh_client.mesh_id_to_name.items():
            # TODO: we could consider adding other parts of MeSH as well
            # that could have some relevance for CODA, e.g., Anatomy (A)
            # analytical tecniques (E), etc.
            is_dis = self.is_disease("MESH", mesh_id)
            is_pat = self.is_pathogen("MESH", mesh_id)
            is_geo = self.is_geoloc("MESH", mesh_id)

            if not any([is_dis, is_pat, is_geo]):
                continue

            # TODO: we could potentially add more informative labels
            # like "Disease", "Pathogen", "Geoloc" instead of just "mesh"
            nodes.add(
                (
                    f"mesh:{mesh_id}",
                    mesh_name,
                    "mesh"
                )
            )

            parent_ids = list(bio_ontology.child_rel("MESH", mesh_id, {"isa"}))
            new_edges = set()

            for _, parent in parent_ids:
                if is_dis and not self.is_disease("MESH", parent):
                    continue
                if is_pat and not self.is_pathogen("MESH", parent):
                    continue
                if is_geo and not self.is_geoloc("MESH", parent):
                    continue

                new_edges.add(
                    (
                        f"mesh:{mesh_id}",
                        f"mesh:{parent}",
                        "isa"
                    )
                )

            edges |= new_edges

        node_header = ['id:ID', 'name', ':LABEL']
        edge_header = [':START_ID', ':END_ID', ':TYPE']

        with open(self.edges_file, 'w') as fh:
            writer = csv.writer(fh, delimiter='\t')
            writer.writerows([edge_header] + sorted(list(edges)))

        with open(self.nodes_file, 'w') as fh:
            writer = csv.writer(fh, delimiter='\t')
            writer.writerows([node_header] + sorted(list(nodes)))

    def is_geoloc(self, x_db, x_id):
        if x_db == 'MESH':
            return self.mesh_client.mesh_isa(x_id, 'D005842')
        return False

    def is_pathogen(self, x_db, x_id):
        if x_db == 'MESH':
            return (
                self.mesh_client.mesh_isa(x_id, 'D001419') or
                self.mesh_client.mesh_isa(x_id, 'D014780')
            )
        return False

    def is_disease(self, x_db, x_id):
        if x_db == 'MESH':
            return self.mesh_client.is_disease(x_id)
        return False


if __name__ == "__main__":
    exporter = MeshExporter()
    exporter.export()