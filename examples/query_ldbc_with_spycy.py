#!/usr/bin/env python3
import csv
import logging
import os
import sys
from pathlib import Path
from time import time
from typing import Dict

import networkx as nx

from spycy.graph import NetworkXGraph
from spycy.spycy import CypherExecutor


def load_ldbc(input_dir: Path) -> nx.MultiDiGraph:
    """Load ldbc data into NetworkX"""

    data_dir = input_dir / "graphs/csv/bi/composite-merged-fk/initial_snapshot"

    edge_dirs = []
    node_dirs = []

    for dir_ in os.listdir(data_dir):
        for typed_entity in os.listdir(data_dir / dir_):
            if "_" in typed_entity:
                edge_dirs.append(data_dir / dir_ / typed_entity)
            else:
                node_dirs.append(data_dir / dir_ / typed_entity)

    output = nx.MultiDiGraph()

    typed_id_to_id: Dict[str, Dict[int, int]] = {}

    def process_node_dir(output: nx.MultiDiGraph, label: str, node_data: Path):
        with node_data.open() as f:
            for row in csv.DictReader(f, delimiter="|"):
                type_ = row.get("type", label)
                if type_ not in typed_id_to_id:
                    typed_id_to_id[type_] = {}
                data = {"labels": list(set([label, type_])), "properties": row}
                node_id = len(output.nodes)
                output.add_node(node_id, **data)
                typed_id_to_id[type_][int(row["id"])] = node_id

    def process_edge_dir(output: nx.MultiDiGraph, label: str, edge_data: Path):
        src_type, etype, dst_type = label.split("_")
        etype_parts = [""]
        for c in etype:
            if c.isupper():
                etype_parts.append("")
            etype_parts[-1] += c

        src_key = src_type
        dst_key = dst_type
        if src_key == dst_key:
            src_key += "1"
            dst_key += "2"
        src_key += "Id"
        dst_key += "Id"

        etype = "_".join(etype_parts).upper()
        with edge_data.open() as f:
            for row in csv.DictReader(f, delimiter="|"):
                edge_prop_data = {
                    k: v for k, v in row.items() if k not in [src_type, dst_type]
                }
                data = {"type": etype, "properties": edge_prop_data}
                start = typed_id_to_id[src_type][int(row[src_key])]
                end = typed_id_to_id[dst_type][int(row[dst_key])]
                output.add_edge(start, end, **data)

    for node_type in node_dirs:
        logging.info("loading %s" % node_type.stem)
        for data in os.listdir(node_type):
            if data.endswith(".csv"):
                process_node_dir(output, node_type.stem, node_type / data)

    for edge_type in edge_dirs:
        logging.info("loading %s" % edge_type.stem)
        for data in os.listdir(edge_type):
            if data.endswith(".csv"):
                process_edge_dir(output, edge_type.stem, edge_type / data)

    return output


if __name__ == "__main__":
    # Generate ldbc snb dataset by following the instructions at https://github.com/ldbc/ldbc_snb_datagen_spark
    if len(sys.argv) != 3:
        print("Usage: ./query_ldbc_with_spycy.py <path to ldbc snb outdir> <query>")
    input_dir = Path(sys.argv[1])
    query = sys.argv[2]

    logging.root.setLevel(0)
    start = time()
    ldbc = load_ldbc(input_dir)
    logging.info("LOADED %s nodes" % len(ldbc.nodes))
    logging.info("LOADED %s edges" % len(ldbc.edges))
    exe = CypherExecutor(graph=NetworkXGraph(_graph=ldbc))
    logging.info(f"Loaded the graph in {time() - start}s")

    start = time()
    print(exe.exec(query))
    logging.info(f"Ran the query in {time() - start}s")
