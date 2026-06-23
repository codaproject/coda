#!/usr/bin/env bash

set -euo pipefail

neo4j console &
neo4j_pid=$!

stop_neo4j() {
  kill -TERM "${neo4j_pid}" 2>/dev/null || true
  wait "${neo4j_pid}" 2>/dev/null || true
}
trap stop_neo4j EXIT
trap 'exit 0' INT TERM

echo "Waiting for Neo4j..."
until curl --fail --silent --output /dev/null "http://127.0.0.1:7474"; do
  if ! kill -0 "${neo4j_pid}" 2>/dev/null; then
    wait "${neo4j_pid}"
    exit $?
  fi
  sleep 3
done

echo "Creating vector indexes..."
python /sw/vector_index.py

echo "Neo4j is ready."
wait "${neo4j_pid}"
