#! /bin/bash
cd ..

cd antlr4
mvn -DskipTests install
cd ..

cd openCypher
mvn clean install
./tools/grammar/src/main/shell/launch.sh Antlr4 grammar/cypher.xml > Cypher.g4
cd ..
