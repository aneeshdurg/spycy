#! /bin/bash
cd antlr4
mvn -DskipTests install
cd ..

cd openCypher
mvn clean install
./tools/grammar/src/main/shell/launch.sh Antlr4 grammar/cypher.xml > Cypher.g4
cd ..

mkdir -p gen
cd gen
cp ../openCypher/Cypher.g4 .
ANTLR_JAR="tool/target/antlr4-4.9-4-SNAPSHOT-complete.jar"
java -jar ../antlr4/$ANTLR_JAR -Dlanguage=Python3 ./Cypher.g4
