FROM ubuntu:24.04


WORKDIR /sw

# Install basic dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get -y install build-essential \
    && apt-get install -y git wget curl nano zip unzip bzip2 gcc graphviz \
        graphviz-dev pkg-config python3-pip cmake libxml2-dev swig \
        software-properties-common dirmngr apt-transport-https \
        gnupg gnupg2 ca-certificates lsb-release ubuntu-keyring

# Install neo4j
RUN curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key | \
        gpg --dearmor -o /usr/share/keyrings/neo4j.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable latest" | \
        tee -a /etc/apt/sources.list.d/neo4j.list \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get update -y \
    && apt-get install -y neo4j=1:2025.10.1

# Retrieves jar file needed to install apoc
# FIXME: get these for the right neo4j version
# RUN wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.4.0.30/apoc-4.4.0.30-all.jar -O /var/lib/neo4j/plugins/apoc-4.4.0.30-all.jar

# Set custom configuration to enable apoc and allow connections
# FIXME: we need to add an apoc.conf
#RUN echo "dbms.security.procedures.unrestricted=apoc.*" >> /etc/neo4j/neo4j.conf
#RUN echo "apoc.export.file.enabled=true" >> /etc/neo4j/neo4j.conf
RUN sed -i 's/#server.default_listen_address/server.default_listen_address/' /etc/neo4j/neo4j.conf
RUN sed -i 's/#dbms.security.auth_enabled/dbms.security.auth_enabled/' /etc/neo4j/neo4j.conf

# On newer Ubuntu versions we need to explicitly allow
# install of Python packages at the system level
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# Fixes ERROR: Cannot uninstall 'blinker'. It is a distutils installed project
# and thus we cannot accurately determine which files belong to it which would lead to only a partial
# uninstall.
RUN python -m pip install --ignore-installed blinker && \
    python -m pip install git+https://github.com/gyorilab/gilda.git && \
    python -c "import nltk;nltk.download('stopwords');nltk.download('punkt_tab')" && \
    python -m gilda.resources


# Ingest graph content into neo4j
COPY kg /sw/kg

RUN neo4j-admin database import full \
    --delimiter='TAB' \
    --skip-duplicate-nodes=true \
    --skip-bad-relationships=true \
    --relationships /sw/kg/icd10_edges.tsv \
    --nodes /sw/kg/icd10_nodes.tsv \
    --relationships /sw/kg/acme_edges.tsv \
    --nodes /sw/kg/acme_nodes.tsv \
    --relationships /sw/kg/icd11_edges.tsv \
    --nodes /sw/kg/icd11_nodes.tsv \
    --relationships /sw/kg/who_va_edges.tsv \
    --nodes /sw/kg/who_va_nodes.tsv \
    --relationships /sw/kg/phmrc_edges.tsv \
    --nodes /sw/kg/phmrc_nodes.tsv \
    --relationships /sw/kg/probbase_edges.tsv \
    --nodes /sw/kg/probbase_nodes.tsv \
    --relationships /sw/kg/hpo_edges.tsv \
    --nodes /sw/kg/hpo_nodes.tsv \
    --bad-tolerance=100000 \
      || (echo "=== IMPORT FAILED, SHOWING /sw/import.report ===" && cat /sw/import.report && exit 1)


ENV DOCKERIZED="TRUE"
ENV NEO4J_URL="bolt://localhost:7687"

COPY startup.sh  /sw/startup.sh

ENTRYPOINT ["/bin/bash",  "/sw/startup.sh"]
