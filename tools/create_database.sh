DB=$1
python database.py $DB --create
sudo cp import/neo4j_$DB.csv /var/lib/neo4j/import/
cd /var/lib/neo4j
sudo service neo4j stop
sudo -u neo4j neo4j-admin database import full --nodes=import/neo4j_$DB.csv --overwrite-destination
sudo service neo4j start