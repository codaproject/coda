import csv
from indra.databases import mesh_client
from indra.ontology.bio import bio_ontology
from coda.kg.sources import KGSourceExporter

class MeshExporter(KGSourceExporter):
    name = "mesh_hierarchy"

    def export(self):
        edges = set()
        nodes = set()

        for mesh_id, mesh_name in mesh_client.mesh_id_to_name.items():
            is_dis = self.is_disease("MESH", mesh_id)
            is_pat = self.is_pathogen("MESH", mesh_id)
            is_geo = self.is_geoloc("MESH", mesh_id)

            if not any([is_dis, is_pat, is_geo]):
                continue

            if is_dis:
                node_type = "disease"
            elif is_pat:
                node_type = "pathogen"
            else:
                node_type = "geoloc"

            nodes.add(
                (
                    f"MESH:{mesh_id}",
                    mesh_name,
                    node_type + ';entity'
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
                        f"MESH:{mesh_id}",
                        f"MESH:{parent}",
                        "isa"
                    )
                )

            edges |= new_edges

        node_header = ['curie:ID', 'name:string', ':LABEL']
        edge_header = [':START_ID', ':END_ID', ':TYPE']

        with open(self.edges_file, 'w') as fh:
            writer = csv.writer(fh, delimiter='\t')
            writer.writerows([edge_header] + sorted(list(edges)))

        with open(self.nodes_file, 'w') as fh:
            writer = csv.writer(fh, delimiter='\t')
            writer.writerows([node_header] + sorted(list(nodes)))

        # ---- write nodes ----
        # with open(self.nodes_file, "w") as fh:
        #     writer = csv.writer(fh, delimiter="\t")
        #     writer.writerow(["id:ID", "name", ":LABEL"])
        #     for node_id, name, label in sorted(nodes):
        #         writer.writerow([node_id, name, label])

        # ---- write edges ----
        # with open(self.edges_file, "w") as fh:
        #     writer = csv.writer(fh, delimiter="\t")
        #     writer.writerow([":START_ID", ":END_ID", ":TYPE"])
        #     for start, end, rel in sorted(edges):
        #         writer.writerow([start, end, rel])

    def is_geoloc(self, x_db, x_id):
        if x_db == 'MESH':
            return mesh_client.mesh_isa(x_id, 'D005842')
        return False

    def is_pathogen(self, x_db, x_id):
        if x_db == 'MESH':
            return (
                mesh_client.mesh_isa(x_id, 'D001419') or
                mesh_client.mesh_isa(x_id, 'D014780')
            )
        return False

    def is_disease(self, x_db, x_id):
        if x_db == 'MESH':
            return mesh_client.is_disease(x_id)
        return False


if __name__ == "__main__":
    exporter = MeshExporter()
    exporter.export()