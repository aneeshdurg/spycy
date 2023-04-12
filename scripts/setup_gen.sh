#! /bin/bash
cd ..

cd spycy/gen
# cp ../openCypher/Cypher.g4 .
cat tck.g4 >> Cypher.g4
ANTLR_JAR="../../antlr4/tool/target/antlr4-4.12.1-SNAPSHOT.jar"
java -jar $ANTLR_JAR -Dlanguage=Python3 ./Cypher.g4
